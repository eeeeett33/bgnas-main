import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .base import BaseNAS
from ..space import BaseSpace
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
from datetime import datetime
import numpy as np
from .rl_utils import ReinforceController, ReinforceField

LOGGER = logging.getLogger(__name__)

class Enas(BaseNAS):

    def __init__(
        self,
        num_epochs=5,
        n_warmup=100,
        log_frequency=None,
        grad_clip=5.0,
        entropy_weight=0.0001,
        skip_weight=0.8,
        baseline_decay=0.999,
        ctrl_lr=0.00035,
        ctrl_steps_aggregate=20,
        ctrl_kwargs=None,
        model_lr=5e-3,
        model_wd=5e-4,
        disable_progress=False,
        device="auto",
    ):
        super().__init__(device)
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency
        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.baseline = 0.0
        self.ctrl_steps_aggregate = ctrl_steps_aggregate
        self.grad_clip = grad_clip
        self.ctrl_kwargs = ctrl_kwargs
        self.ctrl_lr = ctrl_lr
        self.n_warmup = n_warmup
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.disable_progress = disable_progress

    def search(self, space: BaseSpace, dset, estimator):
        self.model = space
        self.dataset = dset
        self.estimator = estimator
        self.nas_modules = []

        k2o = get_module_order(self.model)
        replace_layer_choice(self.model, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.model, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)

        self.model = self.model.to(self.device)
        self.model_optim = torch.optim.Adam(
            self.model.parameters(), lr=self.model_lr, weight_decay=self.model_wd
        )
        self.nas_fields = [
            ReinforceField(
                name,
                len(module),
                isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1,
            )
            for name, module in self.nas_modules
        ]
        self.controller = ReinforceController(
            self.nas_fields, **(self.ctrl_kwargs or {})
        )
        self.ctrl_optim = torch.optim.Adam(
            self.controller.parameters(), lr=self.ctrl_lr
        )

        with tqdm(range(self.n_warmup), disable=self.disable_progress) as bar:
            for i in bar:
                acc, l1 = self._train_model(i)
                with torch.no_grad():
                    val_acc, val_loss = self._infer("val")
                bar.set_postfix(
                    loss=l1.item(),
                    acc=acc["acc"],
                    val_acc=val_acc["acc"],
                    val_loss=val_loss.item(),
                )

        with tqdm(range(self.num_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                acc, l1 = self._train_model(i)
                l2 = self._train_controller(i)
                bar.set_postfix(loss=l1.item(), acc=acc["acc"], reward_controller=l2)

        selection = self.export()
        return space.parse_model(selection)

    def _train_model(self, epoch):
        self.model.train()
        self.controller.eval()
        self.model_optim.zero_grad()
        self._resample()
        metric, loss = self._infer()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.model_optim.step()

        return metric, loss

    def _train_controller(self, epoch):
        self.model.eval()
        self.controller.train()
        self.ctrl_optim.zero_grad()
        rewards = []
        for ctrl_step in range(self.ctrl_steps_aggregate):
            self._resample()
            with torch.no_grad():
                metric, loss = self._infer(mask="val")
                reward = metric["acc"]
            rewards.append(reward)
            if self.entropy_weight:
                reward += self.entropy_weight * self.controller.sample_entropy.item()
            self.baseline = self.baseline * self.baseline_decay + reward * (
                1 - self.baseline_decay
            )
            loss = self.controller.sample_log_prob * (reward - self.baseline)
            if self.skip_weight:
                loss += self.skip_weight * self.controller.sample_skip_penalty
            loss /= self.ctrl_steps_aggregate
            loss.backward()

            if (ctrl_step + 1) % self.ctrl_steps_aggregate == 0:
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.controller.parameters(), self.grad_clip
                    )
                self.ctrl_optim.step()
                self.ctrl_optim.zero_grad()

            if self.log_frequency is not None and ctrl_step % self.log_frequency == 0:
                LOGGER.info(
                    "RL Epoch [%d/%d] Step [%d/%d]  %s",
                    epoch + 1,
                    self.num_epochs,
                    ctrl_step + 1,
                    self.ctrl_steps_aggregate,
                )
        return sum(rewards) / len(rewards)

    def _resample(self):
        result = self.controller.resample()
        for name, module in self.nas_modules:
            module.sampled = result[name]

    def export(self):
        self.controller.eval()
        with torch.no_grad():
            return self.controller.resample()

    def _infer(self, mask="train"):
        metric, loss = self.estimator.infer(self.model, self.dataset, mask=mask)
        return metric, loss
