import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

from src.losses import CoxPHLoss, DeepHitLoss
from src.evaluation import c_index, event_specific_c_index
from src.ssl.label_masks import make_limited_label_masks
from src.training.risk_utils import deephit_expected_time_risk_torch

def predict_gnn_deephit_risk(model, data, mask, device):
    """
    Binary / any-event GNN-DeepHit risk:
    risk = - expected event time bin
    """
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        logits = logits[mask]
        risk = deephit_expected_time_risk_torch(logits, event_id="any")

    return risk.detach().cpu().numpy()


def predict_gnn_deephit_event_risk(model, data, mask, event_id, device):
    """
    Event-specific GNN-DeepHit risk.

    event_id uses original labels: 1, 2, ..., K
    """
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        logits = logits[mask]
        risk = deephit_expected_time_risk_torch(logits, event_id=event_id)

    return risk.detach().cpu().numpy()

def train_gnn_cox(
    model,
    data,
    device="cpu",
    lr=1e-3,
    weight_decay=1e-4,
    epochs=20,
    early_stopping=None,
):
    model = model.to(device)
    data = data.to(device)

    loss_fn = CoxPHLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = []

    best_score = None
    best_state_dict = None
    best_epoch = None
    patience_counter = 0

    if early_stopping is None:
        early_stopping = {"enabled": False}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        log_risk = model(data.x, data.edge_index)

        train_event_binary = (data.event[data.train_mask] != 0).long()

        loss = loss_fn(
            log_risk[data.train_mask],
            data.time[data.train_mask],
            train_event_binary,
        )

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            log_risk = model(data.x, data.edge_index)

            val_risk = log_risk[data.val_mask].detach().cpu().numpy()
            val_time = data.time[data.val_mask].detach().cpu().numpy()
            val_event = data.event[data.val_mask].detach().cpu().numpy()
            val_event_binary = (val_event != 0).astype("int64")

            val_cindex = c_index(
                time=val_time,
                event=val_event_binary,
                risk_score=val_risk,
            )

        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "val_cindex": val_cindex,
        }

        history.append(row)

        if early_stopping.get("enabled", False):
            monitor = early_stopping.get("monitor", "val_cindex")
            mode = early_stopping.get("mode", "max")
            patience = early_stopping.get("patience", 5)
            min_delta = early_stopping.get("min_delta", 0.0001)

            current_score = row[monitor]

            if best_score is None:
                improved = True
            elif mode == "max":
                improved = current_score > best_score + min_delta
            elif mode == "min":
                improved = current_score < best_score - min_delta
            else:
                raise ValueError(f"Unknown early stopping mode: {mode}")

            if improved:
                best_score = current_score
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={loss.item():.4f} | "
            f"val_cindex={val_cindex:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model from epoch {best_epoch}.")

    return model, history

def train_gnn_cox_sampled(
    model,
    data,
    device="cpu",
    lr=1e-3,
    weight_decay=1e-4,
    epochs=20,
    batch_size=4096,
    num_neighbors=[5, 5],
    early_stopping=None,
    semi_supervised=None,
):
    model = model.to(device)
    data = data.to(device)

    loss_fn = CoxPHLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if early_stopping is None:
        early_stopping = {"enabled": False}

    if semi_supervised is not None and semi_supervised.get("enabled", False):
        supervised_mask, _, label_stats = make_limited_label_masks(
            event=data.event,
            train_mask=data.train_mask,
            labeled_event_fraction=semi_supervised.get("labeled_event_fraction", 0.2),
            min_labeled_events=semi_supervised.get("min_labeled_events", 0),
            keep_censored_in_supervised=semi_supervised.get("keep_censored_in_supervised", True),
            seed=semi_supervised.get("seed", 42),
        )
        print("Limited-label stats:", label_stats)
    else:
        supervised_mask = data.train_mask
        label_stats = None

    train_loader = NeighborLoader(
        data,
        input_nodes=supervised_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
    )

    history = []

    best_score = None
    best_state_dict = None
    best_epoch = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            log_risk = model(batch.x, batch.edge_index).view(-1)

            seed_n = batch.batch_size

            event_binary = (batch.event[:seed_n] != 0).long()

            loss = loss_fn(
                log_risk[:seed_n],
                batch.time[:seed_n],
                event_binary,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            full_log_risk = model(data.x, data.edge_index).view(-1)

            val_time = data.time[data.val_mask].detach().cpu().numpy()
            val_event = (data.event[data.val_mask] != 0).long().detach().cpu().numpy()
            val_risk = full_log_risk[data.val_mask].detach().cpu().numpy()

            val_cindex = c_index(
                time=val_time,
                event=val_event,
                risk_score=val_risk,
            )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_cindex": float(val_cindex),
            "batches": n_batches,
            "limited_labels": label_stats is not None,
            "n_labeled_events": label_stats["n_labeled_events"] if label_stats is not None else None,
            "n_supervised": label_stats["n_supervised"] if label_stats is not None else None,
        }

        history.append(row)

        if early_stopping.get("enabled", False):
            monitor = early_stopping.get("monitor", "val_cindex")
            mode = early_stopping.get("mode", "max")
            patience = early_stopping.get("patience", 5)
            min_delta = early_stopping.get("min_delta", 0.0001)

            current_score = row[monitor]

            if best_score is None:
                improved = True
            elif mode == "max":
                improved = current_score > best_score + min_delta
            elif mode == "min":
                improved = current_score < best_score - min_delta
            else:
                raise ValueError(f"Unknown early stopping mode: {mode}")

            if improved:
                best_score = current_score
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"batches={n_batches} | "
            f"val_cindex={val_cindex:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model from epoch {best_epoch}.")

    return model, history

def train_gnn_deephit(
    model,
    data,
    device="cpu",
    lr=1e-3,
    weight_decay=1e-4,
    epochs=20,
    alpha=1.0,
    beta=0.0,
    sigma=0.1,
    early_stopping=None,
):
    model = model.to(device)
    data = data.to(device)

    print("After moving to device:")
    if device == "cuda":
        print("CUDA allocated GB:", torch.cuda.memory_allocated() / 1024**3)
        
    loss_fn = DeepHitLoss(alpha=alpha, beta=beta, sigma=sigma)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = []

    best_score = None
    best_state_dict = None
    best_epoch = None
    patience_counter = 0

    if early_stopping is None:
        early_stopping = {"enabled": False}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)

        loss = loss_fn(
            logits=logits[data.train_mask],
            event=data.event[data.train_mask],
            time_bin=data.time_bin[data.train_mask],
        )

        loss.backward()
        optimizer.step()

        model.eval()

        val_event = data.event[data.val_mask].detach().cpu().numpy()
        val_time = data.time[data.val_mask].detach().cpu().numpy()

        unique_events = sorted(np.unique(val_event).tolist())
        event_ids = [e for e in unique_events if e != 0]

        row = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
        }

        if len(event_ids) <= 1:
            risk = predict_gnn_deephit_risk(
                model=model,
                data=data,
                mask=data.val_mask,
                device=device,
            )

            event_binary = (val_event != 0).astype("int64")

            val_cindex = c_index(
                time=val_time,
                event=event_binary,
                risk_score=risk,
            )

            row["val_cindex"] = val_cindex
            print_metric = f"val_cindex={val_cindex:.4f}"

        else:
            event_cindices = []

            for event_id in event_ids:
                risk = predict_gnn_deephit_event_risk(
                    model=model,
                    data=data,
                    mask=data.val_mask,
                    event_id=event_id,
                    device=device,
                )

                score = event_specific_c_index(
                    time=val_time,
                    event=val_event,
                    risk_score=risk,
                    event_id=event_id,
                )

                row[f"val_event_{event_id}_cindex"] = score
                event_cindices.append(score)

            row["val_mean_cindex"] = float(np.mean(event_cindices))
            print_metric = (
                f"val_mean_cindex={row['val_mean_cindex']:.4f} | "
                + " | ".join(
                    [
                        f"val_event_{event_id}_cindex={row[f'val_event_{event_id}_cindex']:.4f}"
                        for event_id in event_ids
                    ]
                )
            )

        history.append(row)

        if early_stopping.get("enabled", False):
            monitor = early_stopping.get("monitor", "val_cindex")
            mode = early_stopping.get("mode", "max")
            patience = early_stopping.get("patience", 5)
            min_delta = early_stopping.get("min_delta", 0.0001)

            current_score = row[monitor]

            if best_score is None:
                improved = True
            elif mode == "max":
                improved = current_score > best_score + min_delta
            elif mode == "min":
                improved = current_score < best_score - min_delta
            else:
                raise ValueError(f"Unknown early stopping mode: {mode}")

            if improved:
                best_score = current_score
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={loss.item():.4f} | "
            f"{print_metric}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model from epoch {best_epoch}.")

    return model, history

def train_gnn_deephit_sampled(
    model,
    data,
    device="cpu",
    lr=1e-3,
    weight_decay=1e-4,
    epochs=20,
    batch_size=2048,
    num_neighbors=[10, 10],
    alpha=1.0,
    beta=0.0,
    sigma=0.1,
    early_stopping=None,
    semi_supervised=None,
):
    model = model.to(device)
    data = data.to(device)

    loss_fn = DeepHitLoss(alpha=alpha, beta=beta, sigma=sigma)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if early_stopping is None:
        early_stopping = {"enabled": False}

    if semi_supervised is not None and semi_supervised.get("enabled", False):
        supervised_mask, _, label_stats = make_limited_label_masks(
            event=data.event,
            train_mask=data.train_mask,
            labeled_event_fraction=semi_supervised.get("labeled_event_fraction", 0.2),
            min_labeled_events=semi_supervised.get("min_labeled_events", 0),
            keep_censored_in_supervised=semi_supervised.get("keep_censored_in_supervised", True),
            seed=semi_supervised.get("seed", 42),
        )
        print("Limited-label stats:", label_stats)
    else:
        supervised_mask = data.train_mask
        label_stats = None

    train_loader = NeighborLoader(
        data,
        input_nodes=supervised_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
    )

    history = []

    best_score = None
    best_state_dict = None
    best_epoch = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            logits = model(batch.x, batch.edge_index)

            # NeighborLoader puts target train nodes first
            seed_n = batch.batch_size

            loss = loss_fn(
                logits=logits[:seed_n],
                event=batch.event[:seed_n],
                time_bin=batch.time_bin[:seed_n],
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        model.eval()

        val_event = data.event[data.val_mask].detach().cpu().numpy()
        val_time = data.time[data.val_mask].detach().cpu().numpy()

        unique_events = sorted(np.unique(val_event).tolist())
        event_ids = [e for e in unique_events if e != 0]

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "limited_labels": label_stats is not None,
            "n_labeled_events": label_stats["n_labeled_events"] if label_stats is not None else None,
            "n_supervised": label_stats["n_supervised"] if label_stats is not None else None,
        }

        # full-graph validation for now
        if len(event_ids) <= 1:
            risk = predict_gnn_deephit_risk(
                model=model,
                data=data,
                mask=data.val_mask,
                device=device,
            )

            event_binary = (val_event != 0).astype("int64")

            val_cindex = c_index(
                time=val_time,
                event=event_binary,
                risk_score=risk,
            )

            row["val_cindex"] = val_cindex
            print_metric = f"val_cindex={val_cindex:.4f}"

        else:
            event_cindices = []

            for event_id in event_ids:
                risk = predict_gnn_deephit_event_risk(
                    model=model,
                    data=data,
                    mask=data.val_mask,
                    event_id=event_id,
                    device=device,
                )

                score = event_specific_c_index(
                    time=val_time,
                    event=val_event,
                    risk_score=risk,
                    event_id=event_id,
                )

                row[f"val_event_{event_id}_cindex"] = score
                event_cindices.append(score)

            row["val_mean_cindex"] = float(np.mean(event_cindices))
            print_metric = (
                f"val_mean_cindex={row['val_mean_cindex']:.4f} | "
                + " | ".join(
                    [
                        f"val_event_{event_id}_cindex={row[f'val_event_{event_id}_cindex']:.4f}"
                        for event_id in event_ids
                    ]
                )
            )

        history.append(row)

        if early_stopping.get("enabled", False):
            monitor = early_stopping.get("monitor", "val_cindex")
            mode = early_stopping.get("mode", "max")
            patience = early_stopping.get("patience", 5)
            min_delta = early_stopping.get("min_delta", 0.0001)

            current_score = row[monitor]

            if best_score is None:
                improved = True
            elif mode == "max":
                improved = current_score > best_score + min_delta
            elif mode == "min":
                improved = current_score < best_score - min_delta
            else:
                raise ValueError(f"Unknown early stopping mode: {mode}")

            if improved:
                best_score = current_score
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"batches={n_batches} | "
            f"{print_metric}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model from epoch {best_epoch}.")

    return model, history