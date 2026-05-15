import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.losses import CoxPHLoss
from src.evaluation import c_index, event_specific_c_index
from src.losses import CoxPHLoss, DeepHitLoss


def make_dataloader(X, time, event, batch_size=512, shuffle=True):
    event_binary = (event != 0).astype("int64")

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(time, dtype=torch.float32),
        torch.tensor(event_binary, dtype=torch.long),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

def make_deephit_dataloader(X, time_bin, event, batch_size=512, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(time_bin, dtype=torch.long),
        torch.tensor(event, dtype=torch.long),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def predict_risk(model, X, device, batch_size=4096):
    model.eval()

    risks = []
    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            risk = model(xb)
            risks.append(risk.cpu().numpy())

    return np.concatenate(risks)

def predict_deephit_risk(model, X, device, batch_size=4096):
    """
    Binary DeepHit risk score:
    risk = - expected event time bin

    Higher risk means earlier predicted event.
    """
    model.eval()

    risks = []
    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)

            logits = model(xb)  # [B, K, T]
            probs = torch.softmax(logits.view(logits.shape[0], -1), dim=1)
            probs = probs.view_as(logits)

            # For binary DeepHit, K = 1
            event_probs = probs[:, 0, :]  # [B, T]

            time_bins = torch.arange(
                event_probs.shape[1],
                device=device,
                dtype=torch.float32,
            )

            expected_time = (event_probs * time_bins).sum(dim=1)
            risk = -expected_time

            risks.append(risk.cpu().numpy())

    return np.concatenate(risks)

def predict_deephit_event_risk(model, X, event_id, device, batch_size=4096):
    """
    Event-specific DeepHit risk score.

    risk = - expected event time for selected event type.
    event_id uses original labels: 1, 2, ..., K
    """

    model.eval()

    risks = []
    event_channel = event_id - 1

    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)

            logits = model(xb)
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

            risks.append(risk.cpu().numpy())

    return np.concatenate(risks)

def predict_deephit_event_risks_all(model, X, device, batch_size=4096):
    """
    Predict event-specific DeepHit risks for all event channels in one model pass.

    Returns:
        risks: [N, K]
        risk[:, k] = - expected event time for event channel k
    """
    model.eval()

    all_risks = []

    loader = DataLoader(
        torch.tensor(X, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)

            logits = model(xb)  # [B, K, T]
            probs = torch.softmax(logits.view(logits.shape[0], -1), dim=1)
            probs = probs.view_as(logits)

            B, K, T = probs.shape

            time_bins = torch.arange(
                T,
                device=device,
                dtype=torch.float32,
            )

            expected_time = (probs * time_bins.view(1, 1, T)).sum(dim=2)  # [B, K]

            risk = -expected_time  # [B, K]

            all_risks.append(risk.cpu().numpy())

    return np.concatenate(all_risks, axis=0)

def train_deepsurv(
    model,
    train_data,
    val_data,
    device="cpu",
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=20,
    early_stopping=None,
):
    model = model.to(device)

    loss_fn = CoxPHLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loader = make_dataloader(
        train_data["X"],
        train_data["time"],
        train_data["event"],
        batch_size=batch_size,
        shuffle=True,
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
        train_losses = []

        for xb, tb, eb in train_loader:
            xb = xb.to(device)
            tb = tb.to(device)
            eb = eb.to(device)

            optimizer.zero_grad()

            log_risk = model(xb)
            loss = loss_fn(log_risk, tb, eb)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        val_risk = predict_risk(
            model,
            val_data["X"],
            device=device,
        )

        val_cindex = c_index(
            val_data["time"],
            val_data["event"],
            val_risk,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
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
            f"train_loss={train_loss:.4f} | "
            f"val_cindex={val_cindex:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model from epoch {best_epoch}.")

    return model, history

def train_deephit(
    model,
    train_data,
    val_data,
    device="cpu",
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=20,
    alpha=1.0,
    beta=0.2,
    sigma=0.1,
    max_rank_pairs=None,
    early_stopping=None,
):
    model = model.to(device)

    loss_fn = DeepHitLoss(alpha=alpha, beta=beta, sigma=sigma, max_rank_pairs=max_rank_pairs,)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loader = make_deephit_dataloader(
        train_data["X"],
        train_data["time_bin"],
        train_data["event"],
        batch_size=batch_size,
        shuffle=True,
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
        train_losses = []

        for xb, tb, eb in train_loader:
            xb = xb.to(device)
            tb = tb.to(device)
            eb = eb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = loss_fn(logits, tb, eb)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        unique_events = sorted(np.unique(val_data["event"]).tolist())
        event_ids = [e for e in unique_events if e != 0]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
        }

        if len(event_ids) <= 1:
            # Binary/single-event DeepHit validation
            val_risk = predict_deephit_risk(
                model,
                val_data["X"],
                device=device,
            )

            val_event_binary = (val_data["event"] != 0).astype("int64")

            val_cindex = c_index(
                val_data["time"],
                val_event_binary,
                val_risk,
            )

            row["val_cindex"] = val_cindex
            print_metric = f"val_cindex={val_cindex:.4f}"

        else:
            # Competing-risk DeepHit validation
            event_cindices = []

            val_risks_all = predict_deephit_event_risks_all(
                model=model,
                X=val_data["X"],
                device=device,
            )

            for event_id in event_ids:
                event_channel = event_id - 1
                val_risk = val_risks_all[:, event_channel]

                score = event_specific_c_index(
                    time=val_data["time"],
                    event=val_data["event"],
                    risk_score=val_risk,
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
            f"{print_metric}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model from epoch {best_epoch}.")

    return model, history