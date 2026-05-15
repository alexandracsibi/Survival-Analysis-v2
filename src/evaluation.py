import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score


def c_index(time, event, risk_score):
    """
    Concordance index for survival models.

    time:
        observed/censoring time

    event:
        1 = observed event
        0 = censored

    risk_score:
        higher score = higher risk = shorter survival
    """

    time = np.asarray(time)
    event = np.asarray(event)
    risk_score = np.asarray(risk_score)

    # lifelines expects higher predicted value = longer survival,
    # so negate risk scores.
    return concordance_index(
        event_times=time,
        predicted_scores=-risk_score,
        event_observed=event,
    )

def event_specific_c_index(time, event, risk_score, event_id: int):
    """
    Cause-specific C-index for competing risks.

    event:
        0 = censored
        1..K = event type

    For event_id:
        event_id is treated as observed event.
        all other event types are treated as censored.
    """

    event = np.asarray(event)
    event_binary = (event == event_id).astype(int)

    return c_index(
        time=time,
        event=event_binary,
        risk_score=risk_score,
    )

def make_survival_array(time, event):
    """
    Create scikit-survival structured array.

    event:
        0 = censored
        1 = event
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)

    return np.array(
        list(zip(event, time)),
        dtype=[("event", bool), ("time", float)],
    )


def choose_eval_times(train_time, train_event, n_times=5):
    """
    Choose evaluation horizons from observed train event times.
    """
    train_time = np.asarray(train_time, dtype=float)
    train_event = np.asarray(train_event).astype(int)

    event_times = train_time[train_event == 1]

    if len(event_times) == 0:
        return np.array([])

    quantiles = np.linspace(0.2, 0.8, n_times)
    times = np.quantile(event_times, quantiles)
    times = np.unique(times)

    return times


def time_dependent_auc(
    train_time,
    train_event,
    test_time,
    test_event,
    risk_score,
    times=None,
    n_times=5,
):
    """
    Cumulative/dynamic time-dependent AUC.

    Higher risk_score means higher event risk.
    """
    train_time = np.asarray(train_time, dtype=float)
    train_event = np.asarray(train_event).astype(int)

    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event).astype(int)
    risk_score = np.asarray(risk_score, dtype=float)

    if times is None:
        times = choose_eval_times(train_time, train_event, n_times=n_times)

    if len(times) == 0:
        return {}, np.nan

    max_test_time = test_time.max()
    min_test_time = test_time.min()

    times = np.asarray(times, dtype=float)
    times = times[(times > min_test_time) & (times < max_test_time)]

    if len(times) == 0:
        return {}, np.nan

    survival_train = make_survival_array(train_time, train_event)
    survival_test = make_survival_array(test_time, test_event)

    auc_values, mean_auc = cumulative_dynamic_auc(
        survival_train,
        survival_test,
        risk_score,
        times,
    )

    auc_by_time = {
        float(t): float(a)
        for t, a in zip(times, auc_values)
    }

    return auc_by_time, float(mean_auc)

def deephit_survival_at_times(
    probs,
    time_bin_edges,
    eval_times,
):
    """
    Convert DeepHit probabilities to survival probabilities at eval times.

    probs:
        [N, K, T] joint event-time probabilities

    Returns:
        survival_probs [N, len(eval_times)]
    """
    probs = np.asarray(probs, dtype=float)
    time_bin_edges = np.asarray(time_bin_edges, dtype=float)
    eval_times = np.asarray(eval_times, dtype=float)

    event_time_probs = probs.sum(axis=1)          # [N, T]
    cdf = np.cumsum(event_time_probs, axis=1)     # [N, T]

    t = event_time_probs.shape[1]

    survival_list = []

    for eval_time in eval_times:
        bin_idx = np.searchsorted(time_bin_edges[1:], eval_time, side="right")
        bin_idx = min(max(bin_idx, 0), t - 1)

        survival = 1.0 - cdf[:, bin_idx]
        survival_list.append(survival)

    survival_probs = np.stack(survival_list, axis=1)
    survival_probs = np.clip(survival_probs, 0.0, 1.0)

    return survival_probs


def brier_and_ibs(
    train_time,
    train_event,
    test_time,
    test_event,
    survival_probs,
    eval_times,
):
    """
    IPCW Brier score and Integrated Brier Score.

    survival_probs:
        [N_test, len(eval_times)]
    """
    survival_train = make_survival_array(train_time, train_event)
    survival_test = make_survival_array(test_time, test_event)

    _, brier_values = brier_score(
        survival_train,
        survival_test,
        survival_probs,
        eval_times,
    )

    ibs = integrated_brier_score(
        survival_train,
        survival_test,
        survival_probs,
        eval_times,
    )

    brier_by_time = {
        float(t): float(v)
        for t, v in zip(eval_times, brier_values)
    }

    return brier_by_time, float(ibs)