import torch


def deephit_logits_to_probs_torch(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert DeepHit logits [N, K, T] to joint probabilities [N, K, T].
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [N, K, T], got {tuple(logits.shape)}")

    n, k, t = logits.shape

    return torch.softmax(
        logits.reshape(n, k * t),
        dim=1,
    ).reshape(n, k, t)


def deephit_expected_time_risk_torch(
    logits: torch.Tensor,
    event_id: int | str | None = "any",
) -> torch.Tensor:
    """
    DeepHit risk score based on negative expected event-time bin.

    event_id:
        "any" or None:
            any-event / binary risk using all event channels

        integer:
            original 1-based event label, e.g. 1 or 2

    Returns:
        risk [N], where higher = higher predicted risk.
    """
    probs = deephit_logits_to_probs_torch(logits)

    _, k, t = probs.shape
    time_idx = torch.arange(
        t,
        device=logits.device,
        dtype=torch.float32,
    )

    if event_id is None or event_id == "any":
        time_probs = probs.sum(dim=1)  # [N, T]
    else:
        event_idx = int(event_id) - 1

        if event_idx < 0 or event_idx >= k:
            raise ValueError(f"event_id={event_id} is invalid for K={k}")

        time_probs = probs[:, event_idx, :]  # [N, T]

    total_prob = time_probs.sum(dim=1).clamp_min(1e-8)
    expected_time = (time_probs * time_idx).sum(dim=1) / total_prob

    return -expected_time


def deephit_expected_time_risks_all_torch(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    DeepHit event-specific expected-time risks for all event channels.

    Returns:
        risks [N, K]
    """
    probs = deephit_logits_to_probs_torch(logits)

    _, _, t = probs.shape
    time_idx = torch.arange(
        t,
        device=logits.device,
        dtype=torch.float32,
    )

    event_probs = probs.sum(dim=2).clamp_min(1e-8)  # [N, K]
    expected_time = (probs * time_idx.view(1, 1, t)).sum(dim=2) / event_probs

    return -expected_time