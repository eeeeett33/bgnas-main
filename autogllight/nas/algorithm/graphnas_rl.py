import torch
import torch.nn as nn
import logging

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
from datetime import datetime
import numpy as np
from .rl_utils import ReinforceController, ReinforceField

LOGGER = logging.getLogger(__name__)

class GraphNasRL(BaseNAS):

    def __init__(
        self,
        device="auto",
        num_epochs=10,
        log_frequency=None,
        grad_clip=5.0,
        entropy_weight=0.0001,
        skip_weight=0,
        baseline_decay=0.95,
        ctrl_lr=0.00035,
        ctrl_steps_aggregate=100,
        ctrl_kwargs=None,
        n_warmup=100,
        model_lr=5e-3,
        model_wd=5e-4,
        topk=5,
        disable_progress=False,
        hardware_metric_limit=None,
        weight_share=True,
    ):
        super().__init__(device)
        self.num_epochs = num_epochs
        self.log_frequency = log_frequency
        self.entropy_weight = entropy_weight
        self.skip_weight = skip_weight
        self.baseline_decay = baseline_decay
        self.ctrl_steps_aggregate = ctrl_steps_aggregate
        self.grad_clip = grad_clip
        self.ctrl_kwargs = ctrl_kwargs
        self.ctrl_lr = ctrl_lr
        self.n_warmup = n_warmup
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.hist = []
        self.topk = topk
        self.disable_progress = disable_progress
        self.hardware_metric_limit = hardware_metric_limit
        self.weight_share = weight_share

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
        self.nas_fields = [
            ReinforceField(
                name,
                len(module),
                isinstance(module, PathSamplingLayerChoice) or module.n_chosen == 1,
            )
            for name, module in self.nas_modules
        ]
        self.controller = ReinforceController(
            self.nas_fields,
            lstm_size=100,
            temperature=5.0,
            tanh_constant=2.5,
            **(self.ctrl_kwargs or {}),
        )
        self.ctrl_optim = torch.optim.Adam(
            self.controller.parameters(), lr=self.ctrl_lr
        )
        with tqdm(range(self.num_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                l2 = self._train_controller(i)
                bar.set_postfix(reward_controller=l2)

        selections = [x[1] for x in self.hist]
        candidiate_accs = [-x[0] for x in self.hist]
        selection = self._choose_best(selections)
        arch = space.parse_model(selection)
        return arch

    def _choose_best(self, selections):
        results = []
        for selection in selections:
            accs = []
            for i in tqdm(range(20), disable=self.disable_progress):
                self.arch = self.model.parse_model(selection)
                metric, loss = self._infer(mask="val")
                metric = metric["acc"]
                accs.append(metric)
            result = np.mean(accs)
            LOGGER.info(
                "selection {} \n acc {:.4f} +- {:.4f}".format(
                    selection, np.mean(accs), np.std(accs) / np.sqrt(20)
                )
            )
            results.append(result)
        best_selection = selections[np.argmax(results)]
        return best_selection

    def _train_controller(self, epoch):
        self.model.eval()
        self.controller.train()
        self.ctrl_optim.zero_grad()
        rewards = []
        baseline = None
        with tqdm(
            range(self.ctrl_steps_aggregate), disable=self.disable_progress
        ) as bar:
            for ctrl_step in bar:
                self._resample()
                metric, loss = self._infer(mask="val")
                reward = metric["acc"]

                LOGGER.debug(f"{self.selection}\n{metric},{loss}")
                self.hist.append([-reward, self.selection])
                if len(self.hist) > self.topk:
                    self.hist.sort(key=lambda x: x[0])
                    self.hist.pop()
                rewards.append(reward)

                if self.entropy_weight:
                    reward += (
                        self.entropy_weight * self.controller.sample_entropy.item()
                    )

                if not baseline:
                    baseline = reward
                else:
                    baseline = baseline * self.baseline_decay + reward * (
                        1 - self.baseline_decay
                    )

                loss = self.controller.sample_log_prob * (reward - baseline)
                self.ctrl_optim.zero_grad()
                loss.backward()

                self.ctrl_optim.step()

                bar.set_postfix(max_acc=max(rewards), **metric)

        LOGGER.info(
            "epoch:{}, mean rewards:{}".format(epoch, sum(rewards) / len(rewards))
        )
        return sum(rewards) / len(rewards)

    def _resample(self):
        result = self.controller.resample()
        if self.weight_share:
            for name, module in self.nas_modules:
                module.sampled = result[name]
        else:
            self.arch = self.model.parse_model(result)
        self.selection = result

    def export(self):
        self.controller.eval()
        with torch.no_grad():
            return self.controller.resample()

    def _infer(self, mask="train"):
        if self.weight_share:
            metric, loss = self.estimator.infer(self.model, self.dataset, mask=mask)
        else:
            metric, loss = self.estimator.infer(
                self.arch._model, self.dataset, mask=mask
            )
        return metric, loss
