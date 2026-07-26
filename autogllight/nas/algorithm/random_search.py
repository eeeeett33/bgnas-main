import copy
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNAS
from ..space import BaseSpace
from ..space.nni import (
    replace_layer_choice,
    replace_input_choice,
    get_module_order,
    sort_replaced_module,
    PathSamplingLayerChoice,
    PathSamplingInputChoice,
)
from tqdm import tqdm
import numpy as np
import logging as _logging

def setup_nas_logger():
    logger = _logging.getLogger("NAS")
    logger.setLevel(_logging.INFO)

    if not logger.handlers:
        log_filename = f"../nas_search_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = _logging.FileHandler(log_filename)
        file_handler.setLevel(_logging.INFO)

        console_handler = _logging.StreamHandler()
        console_handler.setLevel(_logging.INFO)

        formatter = _logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

class RandomSearch(BaseNAS):

    def __init__(
        self,
        device="auto",
        num_epochs=400,
        disable_progress=False,
        hardware_metric_limit=None,
        select_metric="acc",
    ):
        super().__init__(device=device)
        self.num_epochs = num_epochs
        self.disable_progress = disable_progress
        self.hardware_metric_limit = hardware_metric_limit
        self.select_metric = select_metric

    def init_search(self):
        self.nas_modules = []
        k2o = get_module_order(self.space)
        replace_layer_choice(self.space, PathSamplingLayerChoice, self.nas_modules)
        replace_input_choice(self.space, PathSamplingInputChoice, self.nas_modules)
        self.nas_modules = sort_replaced_module(k2o, self.nas_modules)
        selection_range = {}
        for k, v in self.nas_modules:
            selection_range[k] = len(v)
        self.selection_dict = selection_range
        space_size = np.prod(list(selection_range.values()))

    def search(self, space: BaseSpace, dset, estimator, Dataloader):
        self.estimator = estimator
        self.dataset = dset
        self.space = space

        self.init_search()

        arch_perfs = []
        cache = {}
        with tqdm(range(self.num_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                selection = self.sample()
                vec = tuple(list(selection.values()))
                if vec not in cache:
                    self.arch = space.parse_model(selection)
                    self.arch = self.arch.to(self.device)
                    metric, loss = self._infer(mask="train")
                    metric = metric[self.select_metric]
                    arch_perfs.append([metric, selection])
                    cache[vec] = metric
                bar.set_postfix(auc=metric, max_auc=max(cache.values()))

        selection = arch_perfs[np.argmax([x[0] for x in arch_perfs])][1]
        print(selection)
        arch = space.parse_model(selection)
        return arch

    def sample(self):
        selection = {}
        for k, v in self.selection_dict.items():
            selection[k] = np.random.choice(range(v))
        return {k: int(v) if isinstance(v, np.int64) else v for k, v in selection.items()}

    def _infer(self, mask="train"):
        input_loader = self.dataset
        metric, loss = self.estimator.infer(self.arch, input_loader)
        return metric, loss
