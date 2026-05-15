import torch
import torch.nn as nn


class CoxPHLoss(nn.Module):
    """
    Negative Cox partial log-likelihood loss.
    DeepSurv-style Cox loss.

    Higher predicted log_risk means higher hazard / shorter survival.
    """

    def __init__(self):
        super().__init__()

    def forward(self, log_risk, time, event):
        log_risk = log_risk.view(-1)
        time = time.view(-1)
        event = event.view(-1).float()

        # Sort by descending time.
        # For each event sample i, the risk set contains samples with time_j >= time_i.
        order = torch.argsort(time, descending=True)
        log_risk = log_risk[order]
        event = event[order]

        # Efficient risk-set denominator using cumulative log-sum-exp.
        log_risk_set_sum = torch.logcumsumexp(log_risk, dim=0)

        # Cox partial log-likelihood contribution only for observed events.
        loss = -(log_risk - log_risk_set_sum) * event

        n_events = event.sum()

        if n_events == 0:
            return log_risk.sum() * 0.0

        return loss.sum() / n_events
    
class DeepHitLoss(nn.Module):
    """
    DeepHit-style loss.

    logits: [B, K, T]
        K = number of event types
        T = number of discrete time bins

    event:
        0 = censored
        1..K = event type

    time_bin:
        integer time-bin index, 0..T-1
    """

    def __init__(self, alpha=1.0, beta=0.2, sigma=0.1, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.eps = eps

    def forward(self, logits, time_bin, event):
        probs = torch.softmax(logits.view(logits.shape[0], -1), dim=1)
        probs = probs.view_as(logits)

        likelihood = self._likelihood_loss(probs, time_bin, event)
        ranking = self._ranking_loss(probs, time_bin, event)

        return self.alpha * likelihood + self.beta * ranking

    def _likelihood_loss(self, probs, time_bin, event):
        B, K, T = probs.shape

        time_bin = time_bin.view(-1).long()
        event = event.view(-1).long()

        observed = event > 0
        censored = event == 0

        losses = []

        # Observed event likelihood:
        # P(event type k at time t)
        if observed.any():
            idx = torch.where(observed)[0]
            k = event[idx] - 1
            t = time_bin[idx].clamp(0, T - 1)

            p_event = torch.clamp(probs[idx, k, t], min=self.eps, max=1.0)
            losses.append(-torch.log(p_event))

        # Censored likelihood:
        # P(T > censoring time)
        if censored.any():
            idx = torch.where(censored)[0]
            t = time_bin[idx].clamp(0, T - 1)

            cumulative_event_prob = probs[idx].cumsum(dim=2)
            p_event_until_censor = cumulative_event_prob[
                torch.arange(len(idx), device=probs.device),
                :,
                t,
            ].sum(dim=1)

            p_survive_after_censor = 1.0 - p_event_until_censor
            p_survive_after_censor = torch.clamp(p_survive_after_censor, min=self.eps, max=1.0)

            losses.append(-torch.log(p_survive_after_censor))

        if not losses:
            return probs.sum() * 0.0

        return torch.cat(losses).mean()

    def _ranking_loss(self, probs, time_bin, event):
        B, K, T = probs.shape

        time_bin = time_bin.view(-1).long()
        event = event.view(-1).long()

        observed_idx = torch.where(event > 0)[0]

        if len(observed_idx) == 0:
            return probs.sum() * 0.0

        total_loss = probs.sum() * 0.0
        count = 0

        cumulative_incidence = probs.cumsum(dim=2)

        for i in observed_idx:
            k = event[i] - 1
            t_i = time_bin[i].clamp(0, T - 1)

            # Compare sample i only with samples that survived longer.
            comparable = time_bin > t_i

            if comparable.sum() == 0:
                continue

            F_i = cumulative_incidence[i, k, t_i]
            F_j = cumulative_incidence[comparable, k, t_i]

            diff = -(F_i - F_j) / self.sigma
            diff = torch.clamp(diff, min=-50.0, max=50.0)

            total_loss = total_loss + torch.exp(diff).mean()
            count += 1

        if count == 0:
            return probs.sum() * 0.0

        return total_loss / count