from torch import nn
import torch
from torch.nn import functional as F
import math

def _reshape_weights(w, target_dim):

    shape = [-1] + [1] * (target_dim - 1)
    return w.view(*shape)

class DartsLayerChoice(nn.Module):

    def __init__(self, layer_choice,
                 tau_init: float = 5.0,
                 tau_min: float = 0.1,
                 anneal_rate: float | None = 2.5e-4,
                 hard: bool = True):
        super().__init__()
        self.name = layer_choice.key
        self.op_choices = nn.ModuleDict(layer_choice.named_children())
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

        self.register_buffer("tau", torch.tensor(tau_init))
        self.tau_min = tau_min
        self.anneal_rate = anneal_rate
        self.hard = hard

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs)
                                  for op in self.op_choices.values()])

        weights = F.gumbel_softmax(self.alpha,
                                   tau=float(self.tau),
                                   hard=self.hard,
                                   dim=0)

        return torch.sum(op_results * _reshape_weights(weights, op_results.dim()),
                         dim=0)

    def step_tau(self):
        if self.anneal_rate is not None and self.tau > self.tau_min:
            self.tau = torch.clamp(self.tau * math.exp(-self.anneal_rate),
                                   min=self.tau_min)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super().named_parameters():
            if name == "alpha":
                continue
            yield name, p

    def export(self):
        return torch.argmax(self.alpha).item()

class DartsInputChoice(nn.Module):

    def __init__(self, input_choice,
                 tau_init: float = 5.0,
                 tau_min: float = 0.1,
                 anneal_rate: float | None = 2.5e-4,
                 hard: bool = True):
        super().__init__()
        self.name = input_choice.key
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1
        self.register_buffer("tau", torch.tensor(tau_init))
        self.tau_min = tau_min
        self.anneal_rate = anneal_rate
        self.hard = hard

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        weights = F.gumbel_softmax(self.alpha,
                                   tau=float(self.tau),
                                   hard=self.hard,
                                   dim=0)
        return torch.sum(inputs * _reshape_weights(weights, inputs.dim()), dim=0)

    def step_tau(self):
        if self.anneal_rate is not None and self.tau > self.tau_min:
            self.tau = torch.clamp(self.tau * math.exp(-self.anneal_rate),
                                   min=self.tau_min)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super().named_parameters():
            if name == "alpha":
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().tolist()[: self.n_chosen]
