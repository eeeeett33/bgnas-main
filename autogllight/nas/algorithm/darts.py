import copy
import logging

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNAS
from ..estimator.base import BaseEstimator
from ..space import BaseSpace
from ..space.nni import (
    replace_layer_choice,
    replace_input_choice,
    DartsLayerChoice,
    DartsInputChoice,
)
from tqdm import trange

_logger = logging.getLogger(__name__)

class Darts(BaseNAS):

    def __init__(
            self,
            num_epochs=5,
            workers=4,
            gradient_clip=5.0,
            model_lr=1e-3,
            model_wd=5e-4,
            arch_lr=1e-3,
            arch_wd=1e-3,
            device="auto",
            disable_progress=False,
    ):
        super().__init__(device=device)
        self.num_epochs = num_epochs
        self.workers = workers
        self.gradient_clip = gradient_clip
        self.model_optimizer = torch.optim.Adam
        self.arch_optimizer = torch.optim.Adam
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.arch_lr = arch_lr
        self.arch_wd = arch_wd
        self.disable_progress = disable_progress

    def search(self, space: BaseSpace, dataloader, estimator, Dataloader):
        model_optim_params = list(space.parameters())

        nas_modules = []
        replace_layer_choice(space, DartsLayerChoice, nas_modules)
        replace_input_choice(space, DartsInputChoice, nas_modules)
        space = space.to(self.device)

        ctrl_params = {}
        for _, m in nas_modules:
            if m.name in ctrl_params:
                assert (
                        m.alpha.size() == ctrl_params[m.name].size()
                ), "Size of parameters with the same label should be same."
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        arch_optim = self.arch_optimizer(
            list(ctrl_params.values()), self.arch_lr, weight_decay=self.arch_wd
        )

        model_optim = self.model_optimizer(
            model_optim_params, self.model_lr, weight_decay=self.model_wd
        )

        gen_optim = None
        trigger_gen_ref = None
        if hasattr(estimator, 'trainer') and estimator.trainer is not None:
            owner = getattr(estimator.trainer, '__self__', None)
            if owner is not None and hasattr(owner, 'trigger_gen'):
                trigger_gen_ref = owner.trigger_gen
        if trigger_gen_ref is None and hasattr(estimator, 'trainer_obj') and estimator.trainer_obj is not None:
            if hasattr(estimator.trainer_obj, 'trigger_gen'):
                trigger_gen_ref = estimator.trainer_obj.trigger_gen
        if trigger_gen_ref is not None:
            gen_optim = self.model_optimizer(
                trigger_gen_ref.parameters(), self.model_lr, weight_decay=self.model_wd
            )

        with trange(self.num_epochs, disable=self.disable_progress) as bar:
            for epoch in bar:
                metric, loss = self._train_one_epoch(
                    epoch, space, dataloader, estimator, model_optim, arch_optim, gen_optim
                )
                bar.set_postfix(loss=loss.item(), **metric)

        selection = self.export(nas_modules)
        return space.parse_model(selection)

    def _train_one_epoch(
            self,
            epoch,
            model: BaseSpace,
            dataloader,
            estimator,
            model_optim: torch.optim.Optimizer,
            arch_optim: torch.optim.Optimizer,
            gen_optim: torch.optim.Optimizer = None,
    ):
        model.train()

        model_optim.zero_grad()
        metric, loss_model = self._infer(model, dataloader, estimator, "train")
        loss_model.backward()
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        model_optim.step()

        if gen_optim is not None:
            gen_optim.zero_grad()
            metric_gen, loss_gen = self._infer(model, dataloader, estimator, "gen")
            loss_gen.backward()
            gen_optim.step()

        arch_optim.zero_grad()
        metric_arch, loss_arch = self._infer(model, dataloader, estimator, "val")
        loss_arch.backward()
        arch_optim.step()

        return metric_arch, loss_arch

    def _infer(self, model: BaseSpace, dataloader, estimator: BaseEstimator, mask="train"):
        metric, loss = estimator.infer(model, dataset=dataloader, mask=mask)
        return metric, loss

    @torch.no_grad()
    def export(self, nas_modules) -> dict:
        result = dict()
        for name, module in nas_modules:
            if name not in result:
                result[name] = module.export()
        return result
