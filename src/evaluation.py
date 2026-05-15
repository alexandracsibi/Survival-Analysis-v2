import numpy as np
from lifelines.utils import concordance_index


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