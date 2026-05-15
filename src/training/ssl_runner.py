import copy

import torch

from src.losses import DeepHitLoss
from src.evaluation import c_index, event_specific_c_index
from src.ssl.pseudo_labeling import (
    create_static_teacher,
    generate_pseudo_labels_from_model,
    make_censored_candidate_mask,
    deephit_pseudo_label_loss,
)


def _deephit_expected_time_risk(logits: torch.Tensor) -> torch.Tensor:
    """Return binary DeepHit risk as negative expected event time."""
    n, k, t = logits.shape

    probs = torch.softmax(logits.reshape(n, k * t), dim=1).reshape(n, k, t)

    time_idx = torch.arange(t, device=logits.device).float()
    time_probs = probs.sum(dim=1)

    expected_time = (time_probs * time_idx).sum(dim=1)
    risk = -expected_time

    return risk


def train_gnn_deephit_ssl_static_teacher(
    model,
    data,
    optimizer,
    n_epochs: int = 50,
    alpha: float = 1.0,
    beta: float = 0.0,
    sigma: float = 0.1,
    pseudo_weight: float = 0.2,
    min_confidence: float = 0.7,
    patience: int = 10,
    min_delta: float = 0.0001,
    device: str = "cpu",
):
    """
    Train GraphSAGEDeepHit with static-teacher pseudo-labeling.

    Current validation:
    - binary / any-event C-index
    """
    data = data.to(device)
    model = model.to(device)

    teacher = create_static_teacher(model).to(device)
    deephit_loss = DeepHitLoss(alpha=alpha, beta=beta, sigma=sigma)

    candidate_mask = make_censored_candidate_mask(
        event=data.event,
        train_mask=data.train_mask,
    )

    pseudo = generate_pseudo_labels_from_model(
        model=teacher,
        data=data,
        candidate_mask=candidate_mask,
        min_confidence=min_confidence,
        device=device,
    )

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("-inf")
    bad_epochs = 0

    selected_pseudo = int(pseudo.selected_mask.sum().item())
    print(f"Selected pseudo-labels: {selected_pseudo}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)

        train_logits = logits[data.train_mask]
        train_time = data.time[data.train_mask]
        train_event = data.event[data.train_mask]
        train_time_bin = data.time_bin[data.train_mask]

        supervised_loss = deephit_loss(
            logits=train_logits,
            time=train_time,
            event=train_event,
            time_bin=train_time_bin,
        )

        pseudo_loss = deephit_pseudo_label_loss(
            student_logits=logits,
            pseudo_probs=pseudo.pseudo_probs,
            selected_mask=pseudo.selected_mask,
        )

        loss = supervised_loss + pseudo_weight * pseudo_loss

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            risk = _deephit_expected_time_risk(logits)

            val_time = data.time[data.val_mask].detach().cpu().numpy()
            val_event = (data.event[data.val_mask] != 0).long().detach().cpu().numpy()
            val_risk = risk[data.val_mask].detach().cpu().numpy()

            val_cindex = c_index(
                time=val_time,
                event=val_event,
                risk=val_risk,
            )

        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "supervised_loss": float(supervised_loss.item()),
            "pseudo_loss": float(pseudo_loss.item()),
            "val_cindex": float(val_cindex),
            "selected_pseudo": selected_pseudo,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['loss']:.4f} | "
            f"sup={row['supervised_loss']:.4f} | "
            f"pseudo={row['pseudo_loss']:.4f} | "
            f"val_cindex={row['val_cindex']:.4f} | "
            f"pseudo_n={row['selected_pseudo']}"
        )

        if val_cindex > best_val + min_delta:
            best_val = val_cindex
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    return model, history, pseudo