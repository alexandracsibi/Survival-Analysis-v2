import copy
from pathlib import Path

import torch
from torch_geometric.loader import NeighborLoader

from src.losses import DeepHitLoss
from src.evaluation import c_index, event_specific_c_index
from src.ssl.pseudo_labeling import (
    create_static_teacher,
    generate_pseudo_labels_from_model,
    make_censored_candidate_mask,
    deephit_pseudo_label_loss,
)
from src.ssl.label_masks import make_limited_label_masks


def _deephit_expected_time_risk(logits: torch.Tensor) -> torch.Tensor:
    """Return binary DeepHit risk as negative expected event time."""
    n, k, t = logits.shape

    probs = torch.softmax(logits.reshape(n, k * t), dim=1).reshape(n, k, t)

    time_idx = torch.arange(t, device=logits.device).float()
    time_probs = probs.sum(dim=1)

    expected_time = (time_probs * time_idx).sum(dim=1)
    risk = -expected_time

    return risk

def _create_static_teacher_from_checkpoint(
    model,
    teacher_checkpoint_path=None,
    device="cpu",
):
    teacher = create_static_teacher(model)

    if teacher_checkpoint_path is not None:
        teacher_checkpoint_path = Path(teacher_checkpoint_path)
        state_dict = torch.load(teacher_checkpoint_path, map_location="cpu", weights_only=True)
        teacher.load_state_dict(state_dict)

    teacher = teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


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
    teacher_checkpoint_path=None,
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

    if device == "cuda":
        print("CUDA allocated GB:", torch.cuda.memory_allocated() / 1024**3)

    teacher = _create_static_teacher_from_checkpoint(
        model=model,
        teacher_checkpoint_path=teacher_checkpoint_path,
        device=device,
    )
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
                risk_score=val_risk,
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

def train_gnn_deephit_ssl_static_teacher_sampled(
    model,
    data,
    optimizer,
    n_epochs: int = 50,
    batch_size: int = 4096,
    num_neighbors: list[int] = [5, 5],
    alpha: float = 1.0,
    beta: float = 0.0,
    sigma: float = 0.1,
    pseudo_weight: float = 0.2,
    min_confidence: float = 0.02,
    teacher_checkpoint_path=None,
    semi_supervised=None,
    patience: int = 10,
    min_delta: float = 0.0001,
    device: str = "cpu",
):
    data = data.to(device)
    model = model.to(device)

    teacher = _create_static_teacher_from_checkpoint(
        model=model,
        teacher_checkpoint_path=teacher_checkpoint_path,
        device=device,
    )

    deephit_loss = DeepHitLoss(alpha=alpha, beta=beta, sigma=sigma)

    if semi_supervised is not None and semi_supervised.get("enabled", False):
        supervised_mask, unlabeled_train_mask, label_stats = make_limited_label_masks(
            event=data.event,
            train_mask=data.train_mask,
            labeled_event_fraction=semi_supervised.get("labeled_event_fraction", 0.2),
            min_labeled_events=semi_supervised.get("min_labeled_events", 0),
            keep_censored_in_supervised=semi_supervised.get("keep_censored_in_supervised", True),
            seed=semi_supervised.get("seed", 42),
        )

        print("Limited-label stats:", label_stats)

        candidate_mask = make_censored_candidate_mask(
            event=data.event,
            train_mask=unlabeled_train_mask,
        )
    else:
        supervised_mask = data.train_mask

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

    selected_pseudo = int(pseudo.selected_mask.sum().item())
    print(f"Selected pseudo-labels: {selected_pseudo}")

    train_loader = NeighborLoader(
        data,
        input_nodes=supervised_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
    )

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("-inf")
    bad_epochs = 0

    for epoch in range(1, n_epochs + 1):
        model.train()

        total_loss = 0.0
        total_sup_loss = 0.0
        total_pseudo_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch.x, batch.edge_index)

            seed_n = batch.batch_size

            supervised_loss = deephit_loss(
                logits[:seed_n],
                batch.time_bin[:seed_n],
                batch.event[:seed_n],
            )

            global_node_ids = batch.n_id[:seed_n]

            pseudo_loss = deephit_pseudo_label_loss(
                student_logits=logits[:seed_n],
                pseudo_probs=pseudo.pseudo_probs[global_node_ids],
                selected_mask=pseudo.selected_mask[global_node_ids],
            )

            loss = supervised_loss + pseudo_weight * pseudo_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sup_loss += supervised_loss.item()
            total_pseudo_loss += pseudo_loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        sup_loss = total_sup_loss / max(n_batches, 1)
        pseudo_loss_value = total_pseudo_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            risk = _deephit_expected_time_risk(logits)

            val_time = data.time[data.val_mask].detach().cpu().numpy()
            event_binary = (data.event[data.val_mask] != 0).long().detach().cpu().numpy()
            val_risk = risk[data.val_mask].detach().cpu().numpy()

            val_cindex = c_index(
                time=val_time,
                event=event_binary,
                risk_score=val_risk,
            )

        row = {
            "epoch": epoch,
            "loss": float(train_loss),
            "supervised_loss": float(sup_loss),
            "pseudo_loss": float(pseudo_loss_value),
            "limited_labels": bool(semi_supervised is not None and semi_supervised.get("enabled", False)),
            "n_labeled_events": label_stats["n_labeled_events"] if semi_supervised is not None and semi_supervised.get("enabled", False) else None,
            "n_supervised": label_stats["n_supervised"] if semi_supervised is not None and semi_supervised.get("enabled", False) else None,
            "val_cindex": float(val_cindex),
            "selected_pseudo": selected_pseudo,
            "batches": n_batches,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"sup={sup_loss:.4f} | "
            f"pseudo={pseudo_loss_value:.4f} | "
            f"batches={n_batches} | "
            f"val_cindex={val_cindex:.4f} | "
            f"pseudo_n={selected_pseudo}"
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