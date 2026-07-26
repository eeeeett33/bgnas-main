import logging
import torch
import torch.optim
from .spos import Evolution, UniformSampler, Spos
from ..space import (
    replace_layer_choice,
    replace_input_choice,
    get_module_order,
    sort_replaced_module,
    PathSamplingInputChoice,
    PathSamplingLayerChoice,
)

_logger = logging.getLogger(__name__)

class GRNA(Spos):

    def __init__(
        self,
        n_warmup=1000,
        grad_clip=5.0,
        disable_progress=False,
        optimize_mode="maximize",
        population_size=100,
        sample_size=25,
        cycles=20000,
        mutation_prob=0.05,
        device="cuda",
    ):
        super().__init__(
            n_warmup,
            grad_clip,
            disable_progress,
            optimize_mode,
            population_size,
            sample_size,
            cycles,
            mutation_prob,
            device,
        )

    def _prepare(self):
        self.nas_modules = []
        k2o = get_module_order(self.model)
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)
        self.model = self.model.to(self.device)
        self.model_optim = torch.optim.Adam(
            self.model.parameters(), lr=self.model_lr, weight_decay=self.model_wd
        )
        self.controller = UniformSampler(self.nas_modules)

        self.evolve = Evolution(
            optimize_mode="maximize",
            population_size=self.population_size,
            sample_size=self.sample_size,
            cycles=self.cycles,
            mutation_prob=self.mutation_prob,
            disable_progress=self.disable_progress,
        )

    def _infer(self, mask="train"):
        metric, loss = self.estimator.infer(self.model, self.dataset, mask=mask)
        return metric, loss
