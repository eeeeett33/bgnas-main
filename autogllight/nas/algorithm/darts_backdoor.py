import logging
import os
from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn as nn

from tqdm import trange
from .base import BaseNAS
from ..space.nni import (
    replace_layer_choice,
    replace_input_choice,
    DartsLayerChoice,
    DartsInputChoice,
)

_logger = logging.getLogger(__name__)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

class DartsBackdoor(BaseNAS):

    def __init__(
        self,
        num_epochs=30,
        workers=4,
        gradient_clip=5.0,
        model_lr=1e-3,
        model_wd=5e-4,
        arch_lr=1e-3,
        arch_wd=1e-3,
        gen_lr=1e-3,
        gen_wd=5e-4,
        device="auto",
        disable_progress=False,
    ):
        super().__init__(device=device)
        self.num_epochs = num_epochs
        self.workers = workers
        self.gradient_clip = gradient_clip
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.arch_lr = arch_lr
        self.arch_wd = arch_wd
        self.gen_lr = gen_lr
        self.gen_wd = gen_wd
        self.model_optimizer = torch.optim.Adam
        self.arch_optimizer = torch.optim.Adam
        self.gen_optimizer = torch.optim.Adam
        self.disable_progress = disable_progress

    def search(self, space, dataset, estimator):
        nas_modules_model = []
        replace_layer_choice(space, DartsLayerChoice, nas_modules_model)
        replace_input_choice(space, DartsInputChoice, nas_modules_model)
        space = space.to(self.device)

        assert hasattr(estimator, 'trainer') or hasattr(estimator, 'trainer_obj'), "estimator 缺少 trainer 接口"
        trigger_gen = None
        if hasattr(estimator, 'trainer') and estimator.trainer is not None:
            trigger_gen = getattr(estimator.trainer, 'trigger_gen', None)
        if trigger_gen is None and hasattr(estimator, 'trainer_obj') and estimator.trainer_obj is not None:
            if hasattr(estimator.trainer_obj, 'trigger_gen'):
                trigger_gen = estimator.trainer_obj.trigger_gen
        assert trigger_gen is not None, "未找到触发器生成器 trigger_gen"

        nas_modules_gen = []
        if hasattr(trigger_gen, 'trojan') and hasattr(trigger_gen.trojan, 'mid_space'):
            replace_layer_choice(trigger_gen.trojan.mid_space, DartsLayerChoice, nas_modules_gen)
            replace_input_choice(trigger_gen.trojan.mid_space, DartsInputChoice, nas_modules_gen)
            trigger_gen.trojan.mid_space = trigger_gen.trojan.mid_space.to(self.device)

        ctrl_params_model = {}
        for _, m in nas_modules_model:
            if m.name in ctrl_params_model:
                assert m.alpha.size() == ctrl_params_model[m.name].size()
                m.alpha = ctrl_params_model[m.name]
            else:
                ctrl_params_model[m.name] = m.alpha

        ctrl_params_gen = {}
        for _, m in nas_modules_gen:
            if m.name in ctrl_params_gen:
                assert m.alpha.size() == ctrl_params_gen[m.name].size()
                m.alpha = ctrl_params_gen[m.name]
            else:
                ctrl_params_gen[m.name] = m.alpha

        arch_alpha_list = list(ctrl_params_model.values()) + list(ctrl_params_gen.values())
        arch_optim = self.arch_optimizer(arch_alpha_list, lr=self.arch_lr, weight_decay=self.arch_wd)

        model_optim = self.model_optimizer(space.parameters(), lr=self.model_lr, weight_decay=self.model_wd)
        gen_optim = self.gen_optimizer(trigger_gen.parameters(), lr=self.gen_lr, weight_decay=self.gen_wd)

        with trange(self.num_epochs, disable=self.disable_progress) as bar:
            for epoch in bar:
                metric, loss = self._train_one_epoch(
                    epoch, space, dataset, estimator, model_optim, gen_optim, arch_optim
                )
                bar.set_postfix(loss=float(loss.item()), **metric)

        selection_model = self.export(nas_modules_model)
        selection_gen = self.export(nas_modules_gen)
        fixed_model = space.parse_model(selection_model)
        return fixed_model, selection_gen

    def _train_one_epoch(self, epoch, model, dataset, estimator, model_optim, gen_optim, arch_optim):
        model.train()

        model_optim.zero_grad()
        metric_train, loss_train = estimator.infer(model, dataset=dataset, mask="train")
        loss_train.backward()
        if self.gradient_clip and self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        model_optim.step()

        gen_optim.zero_grad()
        metric_gen, loss_gen = estimator.infer(model, dataset=dataset, mask="gen")
        loss_gen.backward()
        gen_optim.step()

        arch_optim.zero_grad()
        metric_val, loss_val = estimator.infer(model, dataset=dataset, mask="val")
        loss_val.backward()
        arch_optim.step()

        return metric_val, loss_val

    @torch.no_grad()
    def export(self, nas_modules) -> dict:
        result = dict()
        for name, module in nas_modules:
            if name not in result:
                result[name] = module.export()
        return result

