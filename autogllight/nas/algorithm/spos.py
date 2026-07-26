import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseNAS
from ..space import (
    BaseSpace,
    replace_layer_choice,
    replace_input_choice,
    get_module_order,
    sort_replaced_module,
    PathSamplingInputChoice,
    PathSamplingLayerChoice,
    apply_fixed_architecture,
)
from tqdm import tqdm
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

from .ea_utils import Individual, UniformSampler, MutationSampler
import collections
import random

class Evolution:

    def __init__(
        self,
        optimize_mode="maximize",
        population_size=100,
        sample_size=25,
        cycles=20000,
        mutation_prob=0.05,
        disable_progress=False,
    ):
        assert optimize_mode in ["maximize", "minimize"]
        assert sample_size < population_size
        self.optimize_mode = optimize_mode
        self.population_size = population_size
        self.sample_size = sample_size
        self.cycles = cycles
        self.mutation_prob = mutation_prob
        self.disable_progress = disable_progress
        self._worst = (
            float("-inf") if self.optimize_mode == "maximize" else float("inf")
        )

        self._success_count = 0
        self._population = collections.deque()
        self._running_models = []
        self._polling_interval = 2.0
        self._history = []

    def best_parent(self, sample_size=None):

        samples = [p for p in self._population]
        random.shuffle(samples)
        if sample_size is not None:
            samples = list(samples)[:sample_size]
        if self.optimize_mode == "maximize":
            parent = max(samples, key=lambda sample: sample.y)
        else:
            parent = min(samples, key=lambda sample: sample.y)
        return parent.x

    def _prepare(self):
        self.uniform = UniformSampler(self.nas_modules)
        self.mutation = MutationSampler(self.nas_modules, self.mutation_prob)

    def _get_metric(self, config):
        for name, module in self.nas_modules:
            module.sampled = config[name]
        with torch.no_grad():
            metric, loss = self.estimator.infer(self.model, self.dataset, mask="val")
        return metric["acc"]

    def search(self, space: BaseSpace, nas_modules, dset, estimator, device):
        self.model = space
        self.dataset = dset
        self.estimator = estimator
        self.nas_modules = nas_modules
        self.device = device

        self._prepare()
        LOGGER.info("Initializing the first population.")
        with tqdm(range(self.population_size), disable=self.disable_progress) as bar:
            for i in bar:
                config = self.uniform.resample()
                metric = self._get_metric(config)
                individual = Individual(config, metric)
                self._population.append(individual)
                self._history.append(individual)
                bar.set_postfix(
                    metric=metric,
                    max=max(x.y for x in self._population),
                    min=min(x.y for x in self._population),
                )

        LOGGER.info("Running mutations.")
        with tqdm(range(self.cycles), disable=self.disable_progress) as bar:
            for i in bar:
                parent = self.best_parent(self.sample_size)
                config = self.mutation.resample(parent)
                metric = self._get_metric(config)
                individual = Individual(config, metric)
                LOGGER.debug("Individual created: %s", str(individual))
                self._population.append(individual)
                self._history.append(individual)
                if len(self._population) > self.population_size:
                    self._population.popleft()
                bar.set_postfix(
                    metric=metric,
                    max_h=max(x.y for x in self._history),
                    max=max(x.y for x in self._population),
                    min=min(x.y for x in self._population),
                )

        self._history.sort(key=lambda x: x.y)
        if self.optimize_mode == "maximize":
            best = self._history[-1].x
        else:
            best = self._history[0].x
        return best

class Spos(BaseNAS):

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
        super().__init__(device)
        self.model_lr = 5e-3
        self.model_wd = 5e-4
        self.n_warmup = n_warmup
        self.disable_progress = disable_progress
        self.grad_clip = grad_clip
        self.optimize_mode = optimize_mode
        self.population_size = population_size
        self.sample_size = sample_size
        self.cycles = cycles
        self.mutation_prob = mutation_prob

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

    def search(self, space: BaseSpace, dset, estimator):
        self.model = space
        self.dataset = dset
        self.estimator = estimator

        self._prepare()
        self._train()
        self._search()

        selection = self.export()

        print(selection)
        return space.parse_model(selection)

    def _search(self):
        self.best_config = self.evolve.search(
            self.model, self.nas_modules, self.dataset, self.estimator, self.device,
        )

    def _train(self):
        with tqdm(range(self.n_warmup), disable=self.disable_progress) as bar:
            for i in bar:
                acc, l1 = self._train_one_epoch(i)
                with torch.no_grad():
                    val_acc, val_loss = self._infer("val")
                bar.set_postfix(
                    loss=l1.item(),
                    acc=acc["acc"],
                    val_acc=val_acc["acc"],
                    val_loss=val_loss.item(),
                )

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.model_optim.zero_grad()
        self._resample()
        metric, loss = self._infer(mask="train")
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.model_optim.step()
        return metric, loss

    def _resample(self):
        result = self.controller.resample()
        for name, module in self.nas_modules:
            module.sampled = result[name]

    def export(self):
        return self.best_config

    def _infer(self, mask="train"):
        metric, loss = self.estimator.infer(self.model, self.dataset, mask=mask)
        return metric, loss
