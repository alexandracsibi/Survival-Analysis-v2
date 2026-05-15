import numpy as np
import torch

from src.losses import CoxPHLoss, DeepHitLoss
from src.evaluation import c_index, event_specific_c_index

def predict_gnn_deephit_risk(model, data, mask, device):
    """
    Binary GNN-DeepHit risk:
    risk = - expected event time bin
    """
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        logits = logits[mask]

        probs = torch.softmax(logits.view(logits.shape[0], -1), dim=1)
        probs = probs.view_as(logits)

        event_probs = probs[:, 0, :]

        time_bins = torch.arange(
            event_probs.shape[1],
            device=device,
            dtype=torch.float32,
        )

        expected_time = (event_probs * time_bins).sum(dim=1)
        risk = -expected_time

    return risk.detach().cpu().numpy()


def predict_gnn_deephit_event_risk(model, data, mask, event_id, device):
    """
    Event-specific GNN-DeepHit risk.
    """
    model.eval()

    event_channel = event_id - 1

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        logits = logits[mask]

        probs = torch.softmax(logits.view(logits.shape[0], -1), dim=1)
        probs = probs.view_as(logits)

        event_probs = probs[:, event_channel, :]

        time_bins = torch.arange(
            event_probs.shape[1],
            device=device,
            dtype=torch.float32,
        )

        expected_time = (event_probs * time_bins).sum(dim=1)
        risk = -expected_time

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
            logits[data.train_mask],
            data.time_bin[data.train_mask],
            data.event[data.train_mask],
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