from abc import abstractmethod
import torch.nn as nn
import json
from copy import deepcopy
import torch
from .nni import (
    apply_fixed_architecture,
    OrderedLayerChoice,
    OrderedInputChoice,
)

class BoxModel(nn.Module):

    def __init__(self, space_model, *args, **kwargs):
        super().__init__()
        self.init = True
        self.space = []
        self.hyperparams = {}
        self._model = space_model
        self.num_features = self._model.input_dim
        self.num_classes = self._model.output_dim
        self.params = {"num_class": self.num_classes, "features_num": self.num_features}
        self.selection = None

    def fix(self, selection):

        self.selection = selection
        self._model.instantiate()
        apply_fixed_architecture(self._model, selection, verbose=False)
        return self

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __repr__(self) -> str:
        return str({"model": self._model, "selection": self.selection})

class BaseSpace(nn.Module):

    def __init__(self):
        super().__init__()
        self._initialized = False
        self._default_key = 0

    @abstractmethod
    def forward(self, *args, **kwargs):

        raise NotImplementedError()

    def instantiate(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        self.build_graph()
        self._initialized = True

    @abstractmethod
    def build_graph(self):

        raise NotImplementedError()

    def getOriKey(self, key):
        orikey = key
        if orikey == None:
            key = f"default_key_{self._default_key}"
            self._default_key += 1
            orikey = key
        return orikey

    def setLayerChoice(
            self, order, op_candidates, reduction="sum", return_mask=False, key=None
    ):

        key = self.getOriKey(key)
        layer = OrderedLayerChoice(order, op_candidates, reduction, return_mask, key)
        setattr(self, key, layer)
        return layer

    def setInputChoice(
            self,
            order,
            n_candidates=None,
            choose_from=None,
            n_chosen=None,
            reduction="sum",
            return_mask=False,
            key=None,
    ):

        key = self.getOriKey(key)
        layer = OrderedInputChoice(
            order, n_candidates, choose_from, n_chosen, reduction, return_mask, key
        )
        setattr(self, key, layer)
        return layer

    def wrap(self):
        return BoxModel(self)

    def parse_model(self, selection):

        boxmodel = self.wrap().fix(selection)
        return boxmodel

