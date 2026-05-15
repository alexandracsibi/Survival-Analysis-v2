import torch


def make_limited_label_masks(
    event: torch.Tensor,
    train_mask: torch.Tensor,
    labeled_event_fraction: float = 0.2,
    min_labeled_events: int = 0,
    keep_censored_in_supervised: bool = True,
    seed: int = 42,
):
    """
    Create masks for limited-label semi-supervised training.

    supervised_mask:
        Nodes used in supervised DeepHit loss.

    unlabeled_train_mask:
        Train nodes not used in supervised event-label training.
    """
    if not 0.0 < labeled_event_fraction <= 1.0:
        raise ValueError("labeled_event_fraction must be in (0, 1].")

    device = event.device
    train_mask = train_mask.to(device=device, dtype=torch.bool)

    event_nodes = train_mask & (event != 0)
    censored_nodes = train_mask & (event == 0)

    event_idx = torch.where(event_nodes)[0]
    n_events = event_idx.numel()

    if n_events == 0:
        raise ValueError("No observed event nodes found in train split.")

    n_labeled = int(round(n_events * labeled_event_fraction))
    n_labeled = max(n_labeled, min_labeled_events)
    n_labeled = min(n_labeled, n_events)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    perm = torch.randperm(n_events, generator=generator, device="cpu").to(device)
    labeled_event_idx = event_idx[perm[:n_labeled]]

    labeled_event_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    labeled_event_mask[labeled_event_idx] = True

    if keep_censored_in_supervised:
        supervised_mask = labeled_event_mask | censored_nodes
    else:
        supervised_mask = labeled_event_mask

    unlabeled_train_mask = train_mask & ~labeled_event_mask

    stats = {
        "n_train": int(train_mask.sum().item()),
        "n_event_train": int(n_events),
        "n_labeled_events": int(n_labeled),
        "labeled_event_fraction_actual": float(n_labeled / n_events),
        "n_supervised": int(supervised_mask.sum().item()),
        "n_unlabeled_train": int(unlabeled_train_mask.sum().item()),
        "keep_censored_in_supervised": bool(keep_censored_in_supervised),
    }

    return supervised_mask, unlabeled_train_mask, stats