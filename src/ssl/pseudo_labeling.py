import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class PseudoLabelResult:
    pseudo_probs: torch.Tensor      # [N, K, T]
    confidence: torch.Tensor        # [N]
    selected_mask: torch.Tensor     # [N]
    pred_event: torch.Tensor        # [N], 0-based model event index
    pred_time_bin: torch.Tensor     # [N]


def deephit_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert DeepHit logits [N, K, T] to joint probabilities [N, K, T]."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [N, K, T], got {tuple(logits.shape)}")

    n, k, t = logits.shape
    return F.softmax(logits.reshape(n, k * t), dim=1).reshape(n, k, t)


@torch.no_grad()
def generate_deephit_pseudo_labels(
    logits: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    min_confidence: float = 0.7,
    censor_time_bin: Optional[torch.Tensor] = None,
    enforce_censoring_consistency: bool = True,
) -> PseudoLabelResult:
    """
    Generate soft pseudo-labels from DeepHit teacher logits.

    Confidence:
    - sum probabilities over time
    - choose most likely event
    - take max probability over time for that event

    Censoring consistency:
    - for censored samples, only keep pseudo-labels where:
        predicted event time bin > censoring time bin
    """
    probs = deephit_logits_to_probs(logits)

    n, _, _ = probs.shape
    event_probs = probs.sum(dim=2)
    pred_event = event_probs.argmax(dim=1)

    batch_idx = torch.arange(n, device=probs.device)
    probs_for_event = probs[batch_idx, pred_event, :]

    confidence, pred_time_bin = probs_for_event.max(dim=1)
    selected_mask = confidence >= min_confidence

    if candidate_mask is not None:
        if candidate_mask.shape[0] != n:
            raise ValueError(
                f"candidate_mask must have shape [N], got {tuple(candidate_mask.shape)}"
            )
        candidate_mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
        selected_mask = selected_mask & candidate_mask

    if enforce_censoring_consistency and censor_time_bin is not None:
        if censor_time_bin.shape[0] != n:
            raise ValueError(
                f"censor_time_bin must have shape [N], got {tuple(censor_time_bin.shape)}"
            )

        censor_time_bin = censor_time_bin.to(device=logits.device, dtype=torch.long)

        censor_consistent_mask = pred_time_bin > censor_time_bin
        selected_mask = selected_mask & censor_consistent_mask

    return PseudoLabelResult(
        pseudo_probs=probs.detach(),
        confidence=confidence.detach(),
        selected_mask=selected_mask.detach(),
        pred_event=pred_event.detach(),
        pred_time_bin=pred_time_bin.detach(),
    )


def make_censored_candidate_mask(
    event: torch.Tensor,
    train_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return mask for censored samples, optionally restricted to train nodes."""
    candidate_mask = event == 0

    if train_mask is not None:
        if train_mask.shape[0] != event.shape[0]:
            raise ValueError(
                f"train_mask must have shape [N], got {tuple(train_mask.shape)}"
            )
        train_mask = train_mask.to(device=event.device, dtype=torch.bool)
        candidate_mask = candidate_mask & train_mask

    return candidate_mask


def deephit_pseudo_label_loss(
    student_logits: torch.Tensor,
    pseudo_probs: torch.Tensor,
    selected_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL loss between student predictions and soft pseudo-label distributions."""
    if student_logits.ndim != 3:
        raise ValueError(f"Expected student_logits [N, K, T], got {tuple(student_logits.shape)}")

    if pseudo_probs.shape != student_logits.shape:
        raise ValueError(
            f"pseudo_probs shape {tuple(pseudo_probs.shape)} does not match "
            f"student_logits shape {tuple(student_logits.shape)}"
        )

    if selected_mask.shape[0] != student_logits.shape[0]:
        raise ValueError(
            f"selected_mask must have shape [N], got {tuple(selected_mask.shape)}"
        )

    selected_mask = selected_mask.to(device=student_logits.device, dtype=torch.bool)

    if selected_mask.sum() == 0:
        return student_logits.sum() * 0.0

    student_probs = deephit_logits_to_probs(student_logits)

    student_selected = student_probs[selected_mask].clamp_min(eps)
    pseudo_selected = pseudo_probs[selected_mask].to(student_logits.device).clamp_min(eps)

    loss = pseudo_selected * (pseudo_selected.log() - student_selected.log())
    loss = loss.sum(dim=(1, 2)).mean()

    return loss


torch.no_grad()
def generate_pseudo_labels_from_model(
    model: torch.nn.Module,
    data,
    candidate_mask: Optional[torch.Tensor] = None,
    min_confidence: float = 0.7,
    device: Optional[torch.device] = None,
    enforce_censoring_consistency: bool = True,
) -> PseudoLabelResult:
    """Generate DeepHit pseudo-labels from a GNN teacher model."""
    was_training = model.training
    model.eval()

    if device is not None:
        data = data.to(device)
        model = model.to(device)

    logits = model(data.x, data.edge_index)

    censor_time_bin = None
    if enforce_censoring_consistency:
        if not hasattr(data, "time_bin"):
            raise ValueError(
                "Censoring-consistency filtering requires data.time_bin, "
                "but the graph data object does not contain it."
            )
        censor_time_bin = data.time_bin

    pseudo = generate_deephit_pseudo_labels(
        logits=logits,
        candidate_mask=candidate_mask,
        min_confidence=min_confidence,
        censor_time_bin=censor_time_bin,
        enforce_censoring_consistency=enforce_censoring_consistency,
    )

    if was_training:
        model.train()

    return pseudo


def create_static_teacher(model: torch.nn.Module) -> torch.nn.Module:
    """Create a frozen copy of a trained student model."""
    teacher = copy.deepcopy(model)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


def update_ema_teacher(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    ema_decay: float = 0.99,
) -> None:
    """Update teacher weights with exponential moving average."""
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(ema_decay).add_(
                student_param.data,
                alpha=1.0 - ema_decay,
            )