"""Small NumPy cosine student using the archived two-class FUME objective.

This is a controlled linear student, not the original 4096-dimensional network.
For unit two-dimensional embeddings and W=I, the fuzzy term simplifies exactly
to (1 + wrong_membership - correct_membership)/sqrt(2) per modality. The two
consistency terms in archived processor.calc_loss are equal by score symmetry.
No uncertainty penalty enters this candidate generator.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


def design(raw):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
        raise ValueError("This pilot requires finite two-dimensional inputs")
    return np.column_stack((np.ones(len(raw)), raw))


def initialize(seed):
    rng = np.random.default_rng(seed)
    return {"image": rng.normal(0., .5, (2, 3)),
            "text": rng.normal(0., .5, (2, 3))}


def copy_parameters(parameters):
    return {name: value.copy() for name, value in parameters.items()}


def embedding(inputs, weights):
    raw = np.asarray(inputs) @ np.asarray(weights).T
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    if not np.isfinite(raw).all() or np.any(norm == 0):
        raise FloatingPointError("Nonfinite or zero cosine embedding")
    return raw / norm


def loss_and_grad(parameters, image, text, labels, alpha=.5, weight_decay=1e-4):
    labels = np.asarray(labels)
    n = len(labels)
    if n < 2 or image.shape != (n, 3) or text.shape != (n, 3):
        raise ValueError("Need matched non-singleton batches with design width 3")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("This controlled student uses binary class IDs")
    views, raws, norms = [], [], []
    for name, inputs in (("image", image), ("text", text)):
        raw = inputs @ parameters[name].T
        norm = np.linalg.norm(raw, axis=1, keepdims=True)
        if not np.isfinite(raw).all() or np.any(norm == 0):
            raise FloatingPointError("Invalid embedding in training")
        views.append(raw / norm)
        raws.append(raw)
        norms.append(norm)
    onehot = np.eye(2)[labels.astype(int)]
    sign = 1. - 2. * onehot
    fuzzy = sum(np.mean(1. + np.sum(np.maximum(view, 0.) * sign, axis=1))
                / np.sqrt(2.) for view in views)
    fuzzy_grads = [sign * (view > 0.) / (n * np.sqrt(2.)) for view in views]
    features = np.concatenate(views)
    count = 2 * n
    sim = features @ features.T
    np.fill_diagonal(sim, -np.inf)
    log_denominator = logsumexp(sim, axis=1)
    partner = (np.arange(count) + n) % count
    one_consistency = float(np.mean(log_denominator - sim[np.arange(count), partner]))
    consistency = 2. * one_consistency
    probability = np.exp(sim - log_denominator[:, None])
    probability[np.arange(count), partner] -= 1.
    derivative_sim = 2. * probability / count
    derivative_features = (derivative_sim + derivative_sim.T) @ features
    gradients = {}
    for j, (name, inputs) in enumerate((("image", image), ("text", text))):
        derivative = fuzzy_grads[j] + alpha * derivative_features[j*n:(j+1)*n]
        projected = (derivative - views[j] * np.sum(derivative * views[j], axis=1, keepdims=True)) / norms[j]
        gradients[name] = projected.T @ inputs + weight_decay * parameters[name]
    base = float(fuzzy + alpha * consistency)
    regularizer = .5 * weight_decay * sum(float(np.sum(w*w)) for w in parameters.values())
    loss = base + regularizer
    if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients.values()):
        raise FloatingPointError("Nonfinite loss or gradient")
    return loss, gradients, {"loss": loss, "base_loss": base, "fuzzy_loss": float(fuzzy),
                             "consistency_loss": consistency,
                             "gradient_norm": float(np.sqrt(sum(np.sum(g*g) for g in gradients.values())))}


@dataclass
class AdamStudent:
    parameters: dict
    learning_rate: float = .01
    weight_decay: float = 1e-4
    alpha: float = .5

    def __post_init__(self):
        self.parameters = copy_parameters(self.parameters)
        self.m = {k: np.zeros_like(w) for k, w in self.parameters.items()}
        self.v = {k: np.zeros_like(w) for k, w in self.parameters.items()}
        self.step_count = 0

    def train_epoch(self, image, text, labels, batch_size, rng):
        order = rng.permutation(len(labels))
        batches = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2] = np.concatenate((batches[-2], batches[-1]))
            batches.pop()
        if not np.array_equal(np.sort(np.concatenate(batches)), np.arange(len(labels))):
            raise AssertionError("Every training entity must appear exactly once")
        values = []
        for ids in batches:
            _, grads, record = loss_and_grad(
                self.parameters, image[ids], text[ids], labels[ids],
                alpha=self.alpha, weight_decay=self.weight_decay)
            self.step_count += 1
            for name, grad in grads.items():
                self.m[name] = .9*self.m[name] + .1*grad
                self.v[name] = .999*self.v[name] + .001*grad*grad
                mhat = self.m[name] / (1. - .9**self.step_count)
                vhat = self.v[name] / (1. - .999**self.step_count)
                self.parameters[name] -= self.learning_rate * mhat / (np.sqrt(vhat) + 1e-8)
            values.append((len(ids), record))
        return {**{key: sum(size*record[key] for size, record in values) / len(labels)
                   for key in values[0][1]},
                "entities": len(labels), "batches": len(batches),
                "cumulative_adam_steps": self.step_count}
