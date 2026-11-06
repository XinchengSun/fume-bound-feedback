"""FUME towers + multilabel evidence, with the fuzzy branch removed.

Tower layout is adapted from siyuancncd/FUME (5dd7a306), model.py.
No original fuzzy loss, necessity, credibility c, or pair evidence head is used.
The shared evidence budget is an explicit v5 modeling restriction, not VLS's
original parameterization. Every Beta parameter retains a unit prior (+1).
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    image_dim: int = 4096
    text_dim: int = 1000
    classes: int = 100
    hidden: int = 4096
    layers: int = 4
    initial_evidence: float = 2.0
    norm_eps: float = 1e-8

    def validate(self):
        for key in ('image_dim', 'text_dim', 'classes', 'hidden', 'layers'):
            v = getattr(self, key)
            if type(v) is not int or v < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.classes < 2:
            raise ValueError('At least two label positions are required')
        if not math.isfinite(self.initial_evidence) or self.initial_evidence <= 0:
            raise ValueError('initial_evidence must be finite and positive')
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('norm_eps must be finite and positive')


class Tower(nn.Module):
    """Same layer sizes, bias convention and hidden ReLUs as FUME."""
    def __init__(self, input_dim, output_dim, hidden, layers, eps):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden)])
        self.layers.extend(nn.Linear(hidden, hidden) for _ in range(layers - 1))
        self.layers.append(nn.Linear(hidden, output_dim, bias=False))
        self.eps = eps

    def features(self, x):
        if x.ndim != 2 or x.shape[-1] != self.layers[0].in_features:
            raise ValueError('Feature matrix shape does not match the tower')
        x = x.to(dtype=self.layers[0].weight.dtype)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        hidden = x
        h = self.layers[-1](hidden)  # No output ReLU.
        z = F.normalize(h, p=2, dim=-1, eps=self.eps)
        return hidden, h, z

    def forward(self, x):
        return self.features(x)[2]


class MultiLabelV5(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        config.validate()
        self.config = config
        k = config.classes
        self.img_net = Tower(config.image_dim, k, config.hidden, config.layers, config.norm_eps)
        self.text_net = Tower(config.text_dim, k, config.hidden, config.layers, config.norm_eps)
        # Preserve the square d=K orthogonal initialization used by the FUME entrypoint.
        w = nn.init.orthogonal_(torch.empty(k, k))
        w = w / w.norm(p=2, dim=1, keepdim=True)
        self.register_buffer('W', w)
        self.img_evidence = nn.Linear(config.hidden, 1)
        self.text_evidence = nn.Linear(config.hidden, 1)
        initial = config.initial_evidence
        bias = initial + math.log(-math.expm1(-initial))  # inverse softplus
        for layer in (self.img_evidence, self.text_evidence):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, bias)

    def view(self, x, modality):
        if modality not in ('image', 'text'):
            raise ValueError('modality must be image or text')
        tower, evidence = ((self.img_net, self.img_evidence) if modality == 'image'
                           else (self.text_net, self.text_evidence))
        hidden, h, z = tower.features(x)
        logits = h @ self.W  # Not z @ W: avoid the unit-norm classification bottleneck.
        direction = torch.sigmoid(logits)
        budget = F.softplus(evidence(hidden))  # shared within a sample, not across samples
        alpha = 1.0 + budget * direction
        beta = 1.0 + budget * (1.0 - direction)
        predictive_logits = alpha.log() - beta.log()
        return dict(hidden=hidden, h=h, z=z, logits=logits, direction=direction,
                    evidence_total=budget, alpha=alpha, beta=beta,
                    predictive_logits=predictive_logits,
                    probability=alpha / (alpha + beta))

    def forward(self, image, text):
        return self.view(image, 'image'), self.view(text, 'text')

    def configuration(self):
        return asdict(self.config)


def warmstart_fume(model, checkpoint):
    """Explicit fuzzy-pretrained encoder warm start; NOT a clean-from-scratch experiment.

    No pickle-unrestricted load here: caller must use torch.load(weights_only=True).
    Missing fixed W or any encoder tensor is an error, never a silent reinitialization.
    """
    state = checkpoint.get('fume_state', checkpoint.get('state_dict'))
    if not isinstance(state, dict) or 'fixed_W' not in checkpoint:
        raise ValueError('FUME warmstart requires state_dict/fume_state and fixed_W')
    expected = {k: v for k, v in model.state_dict().items()
                if k.startswith(('img_net.', 'text_net.'))}
    supplied = {k: v for k, v in state.items() if k.startswith(('img_net.', 'text_net.'))}
    if supplied.keys() != expected.keys():
        raise ValueError('FUME encoder key set differs; do not guess an architecture')
    for k in expected:
        if supplied[k].shape != expected[k].shape or not torch.isfinite(supplied[k]).all():
            raise ValueError(f'Invalid encoder tensor: {k}')
    w = checkpoint['fixed_W']
    if w.shape != model.W.shape or not torch.isfinite(w).all():
        raise ValueError('fixed_W shape/value mismatch')
    if not torch.allclose(w.float().T @ w.float(), torch.eye(len(w), device=w.device),
                          atol=1e-4, rtol=1e-4):
        raise ValueError('v5 expects the square orthogonal FUME W')
    target = model.state_dict()
    target.update(supplied)
    target['W'] = w
    model.load_state_dict(target, strict=True)
