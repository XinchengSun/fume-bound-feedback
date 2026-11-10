"""Continuous last-layer training with the archived FUME and Laplace objectives."""
from pathlib import Path
import sys

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.spatial.distance import cdist
from torch.nn import functional as F

UPSTREAM = Path(__file__).resolve().parents[1] / "certification/replay/sources/upstream"
sys.path.insert(0, str(UPSTREAM))
from processor import calc_loss  # noqa: E402; unchanged, hash-registered source

ARMS = ("original", "predictive_mean", "stop_sd", "full")
Z95 = 1.6448536269514722


def make_batches(n, batch_size, epochs, seed):
    """Every row exactly once per epoch; merge a singleton tail, never drop it."""
    if min(n, batch_size) < 2 or epochs < 1:
        raise ValueError("Need at least two rows, batch_size >= 2 and epochs >= 1")
    gen = torch.Generator().manual_seed(seed)
    result = []
    for _ in range(epochs):
        batches = list(torch.randperm(n, generator=gen).split(batch_size))
        if len(batches) > 1 and len(batches[-1]) == 1:
            tail = batches.pop()
            batches[-1] = torch.cat((batches[-1], tail))
        joined = torch.cat(batches)
        assert len(joined) == n and len(joined.unique()) == n
        result.append(batches)
    return result


def emb(x, w):
    value = F.linear(x, w)
    if not torch.isfinite(value).all() or (value.detach().norm(dim=1) == 0).any():
        raise FloatingPointError("Nonfinite or zero embedding")
    return F.normalize(value, dim=1)


def design(a, b):
    value = 10 * (a * b).double()
    return torch.cat((torch.ones_like(value[:, :1]), value), dim=1)


def crossed(a, b):
    return design(a[:, None, :].expand(-1, len(b), -1).reshape(-1, a.shape[1]),
                  b.repeat(len(a), 1))


def posterior(x, y):
    """N(0, 4I) MAP plus the archived implicit differentiation Newton step."""
    y = y.double()
    prior = torch.eye(x.shape[1], dtype=x.dtype, device=x.device) / 4

    def derivatives(w):
        p = (x @ w).sigmoid()
        return x.T @ ((p * (1-p))[:, None] * x) + prior, x.T @ (p-y) + prior @ w

    with torch.no_grad():
        w = x.new_zeros(x.shape[1])
        for _ in range(40):
            h, g = derivatives(w)
            if g.abs().max() < 1e-9:
                break
            w = w - torch.linalg.solve(h, g)
        h, g = derivatives(w)
        if not torch.isfinite(w).all() or g.abs().max() >= 1e-7:
            raise FloatingPointError("Laplace MAP solve failed its stationarity check")
    h, g = derivatives(w)
    w = w - torch.linalg.solve(h, g)
    h, _ = derivatives(w)
    return w, torch.linalg.inv(h)


def balanced(values, labels):
    if not labels.any() or labels.all():
        raise ValueError("Balanced feedback needs both positive and negative pairs")
    return (values[labels].mean() + values[~labels].mean()) / 2


def entropy(p):
    return -torch.xlogy(p, p) - torch.xlogy(1-p, 1-p)


def map_scores(order, y):
    relevance = y[order] == y[:, None]
    return (np.sum(relevance * relevance.cumsum(1) / np.arange(1, len(y)+1), axis=1)
            / relevance.sum(1)).mean()


class Experiment:
    def __init__(self, pack, support):
        self.pack = pack
        self.I, self.T, self.Y = pack["train"]
        self.support = support
        roots, mass = hermgauss(24)
        self.nodes = torch.tensor(roots * np.sqrt(2), dtype=torch.float64)
        self.mass = torch.tensor(mass / np.sqrt(np.pi), dtype=torch.float64)
        self.log_mass = self.mass.log()

    def context(self, wi, wt):
        i, t = self.support
        return posterior(design(emb(self.I[i], wi), emb(self.T[t], wt)), self.Y[i] == self.Y[t])

    @staticmethod
    def sufficient(x, context):
        w, cov = context
        variance = ((x @ cov) * x).sum(1)
        if not torch.isfinite(variance).all() or (variance <= 0).any():
            raise FloatingPointError("Nonpositive or nonfinite posterior logit variance")
        return x @ w, variance.sqrt()

    def predictive(self, mu, sd):
        samples = mu[:, None] + sd[:, None] * self.nodes
        log_q = torch.logsumexp(F.logsigmoid(samples) + self.log_mass, dim=1)
        log_one_minus_q = torch.logsumexp(F.logsigmoid(-samples) + self.log_mass, dim=1)
        q = samples.sigmoid() @ self.mass
        au = (F.softplus(-samples.abs()) + samples.abs() * (-samples.abs()).sigmoid()) @ self.mass
        return q, au, log_q, log_one_minus_q

    def feedback_loss(self, mu, sd, labels, arm):
        sign = 2 * labels.double() - 1
        if arm == "predictive_mean":
            _, _, log_q, log_not_q = self.predictive(mu, sd)
            return balanced(torch.where(labels, -log_q, -log_not_q), labels)
        if arm in ("full", "stop_sd"):
            used_sd = sd.detach() if arm == "stop_sd" else sd
            return balanced(F.softplus(-sign * mu + Z95 * used_sd), labels)
        raise ValueError(f"Unknown feedback arm: {arm}")

    def loss(self, wi, wt, ids, arm, coefficient=1.0):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        a, b = emb(self.I[ids], wi), emb(self.T[ids], wt)
        labels = F.one_hot(self.Y[ids], self.pack["W"].shape[1]).float()
        W = F.normalize(self.pack["W"], dim=1)
        base = calc_loss(a, b, (a @ W).relu(), (b @ W).relu(), labels, labels, .5)
        feedback = base.new_zeros(())
        if arm != "original":
            mu, sd = self.sufficient(crossed(a, b), self.context(wi, wt))
            y = (self.Y[ids, None] == self.Y[ids][None, :]).flatten()
            feedback = self.feedback_loss(mu, sd, y, arm)
        total = base + (coefficient * feedback if arm != "original" else 0)
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite objective; no clipping or sample filtering applied")
        return total, {"base_loss": base.item(), "feedback_loss": feedback.item(), "loss": total.item()}

    @torch.no_grad()
    def evaluate(self, wi, wt, split, chunk=32):
        ip, tp, y = self.pack[split]
        a, b = emb(ip, wi), emb(tp, wt)
        distance = cdist(a.numpy(), b.numpy(), "cosine")
        yn = y.numpy()
        ab, ba = [float(map_scores(d.argsort(axis=1), yn)) for d in (distance, distance.T)]
        ctx = self.context(wi, wt)
        sums = dict.fromkeys(("brier", "nll", "AU", "EU", "TU", "lower", "upper", "width",
                              "mean_probability", "logit_sd", "pos_L", "neg_L", "pos_loss", "neg_loss"), 0.0)
        gaps = torch.zeros(10, dtype=torch.float64)
        positives = negatives = 0
        for start in range(0, len(a), chunk):
            mu, sd = self.sufficient(crossed(a[start:start+chunk], b), ctx)
            labels = (y[start:start+chunk, None] == y[None, :]).flatten()
            q, au, log_q, log_not_q = self.predictive(mu, sd)
            lo, hi = (mu-Z95*sd).sigmoid(), (mu+Z95*sd).sigmoid()
            tu = entropy(q)
            for key, value in {"brier": (q-labels.double()).square(),
                               "nll": torch.where(labels, -log_q, -log_not_q),
                               "AU": au, "EU": tu-au, "TU": tu, "lower": lo, "upper": hi,
                               "width": hi-lo, "mean_probability": q, "logit_sd": sd}.items():
                sums[key] += value.sum().item()
            sums["pos_L"] += lo[labels].sum().item()
            sums["neg_L"] += (1-hi[~labels]).sum().item()
            event_loss = F.softplus(-(2*labels.double()-1)*mu + Z95*sd)
            sums["pos_loss"] += event_loss[labels].sum().item()
            sums["neg_loss"] += event_loss[~labels].sum().item()
            positives += int(labels.sum())
            negatives += int((~labels).sum())
            gaps += torch.bincount((10*q).long().clamp(max=9), weights=q-labels.double(), minlength=10)
        n = len(y)**2
        if not positives or not negatives:
            raise ValueError("Evaluation must include both pair classes")
        result = {k: v/n for k, v in sums.items() if k not in ("pos_L", "neg_L", "pos_loss", "neg_loss")}
        result.update(i2t=ab, t2i=ba, map=(ab+ba)/2, ece=gaps.abs().sum().item()/n,
                      pos_L=sums["pos_L"]/positives, neg_L=sums["neg_L"]/negatives,
                      event_loss=.5*(sums["pos_loss"]/positives+sums["neg_loss"]/negatives),
                      queries=len(y), pairs=n)
        if not all(np.isfinite(v) for v in result.values()):
            raise FloatingPointError("Nonfinite evaluation; result not silently repaired")
        return result
