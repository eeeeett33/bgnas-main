
import logging
import warnings
from collections import OrderedDict

import torch.nn as nn

from .utils import global_mutable_counting

logger = logging.getLogger(__name__)

class Mutable(nn.Module):

    def __init__(self, key=None):
        super().__init__()
        if key is not None:
            if not isinstance(key, str):
                key = str(key)
                logger.warning(
                    'Warning: key "%s" is not string, converted to string.', key
                )
            self._key = key
        else:
            self._key = self.__class__.__name__ + str(global_mutable_counting())
        self.init_hook = self.forward_hook = None

    def __deepcopy__(self, memodict=None):
        raise NotImplementedError("Deep copy doesn't work for mutables.")

    def __call__(self, *args, **kwargs):
        self._check_built()
        return super().__call__(*args, **kwargs)

    def set_mutator(self, mutator):
        if "mutator" in self.__dict__:
            raise RuntimeError(
                "`set_mutator` is called more than once. Did you parse the search space multiple times? "
                "Or did you apply multiple fixed architectures?"
            )
        self.__dict__["mutator"] = mutator

    @property
    def key(self):

        return self._key

    @property
    def name(self):

        return self._name if hasattr(self, "_name") else self._key

    @name.setter
    def name(self, name):
        self._name = name

    def _check_built(self):
        if not hasattr(self, "mutator"):
            raise ValueError(
                "Mutator not set for {}. You might have forgotten to initialize and apply your mutator. "
                "Or did you initialize a mutable on the fly in forward pass? Move to `__init__` "
                "so that trainer can locate all your mutables. See NNI docs for more details.".format(
                    self
                )
            )

class MutableScope(Mutable):

    def __init__(self, key):
        super().__init__(key=key)

    def _check_built(self):
        return True

    def __call__(self, *args, **kwargs):
        if not hasattr(self, "mutator"):
            return super().__call__(*args, **kwargs)
        warnings.warn("`MutableScope` is deprecated in Retiarii.", DeprecationWarning)
        try:
            self._check_built()
            self.mutator.enter_mutable_scope(self)
            return super().__call__(*args, **kwargs)
        finally:
            self.mutator.exit_mutable_scope(self)

class LayerChoice(Mutable):

    def __init__(self, op_candidates, reduction="sum", return_mask=False, key=None):
        super().__init__(key=key)
        self.names = []
        if isinstance(op_candidates, OrderedDict):
            for name, module in op_candidates.items():
                assert name not in [
                    "length",
                    "reduction",
                    "return_mask",
                    "_key",
                    "key",
                    "names",
                ], "Please don't use a reserved name '{}' for your module.".format(name)
                self.add_module(name, module)
                self.names.append(name)
        elif isinstance(op_candidates, list):
            for i, module in enumerate(op_candidates):
                self.add_module(str(i), module)
                self.names.append(str(i))
        else:
            raise TypeError(
                "Unsupported op_candidates type: {}".format(type(op_candidates))
            )
        self.reduction = reduction
        self.return_mask = return_mask

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._modules[idx]
        return list(self)[idx]

    def __setitem__(self, idx, module):
        key = idx if isinstance(idx, str) else self.names[idx]
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in self.names[idx]:
                delattr(self, key)
        else:
            if isinstance(idx, str):
                key, idx = idx, self.names.index(idx)
            else:
                key = self.names[idx]
            delattr(self, key)
        del self.names[idx]

    @property
    def length(self):
        warnings.warn(
            "layer_choice.length is deprecated. Use `len(layer_choice)` instead.",
            DeprecationWarning,
        )
        return len(self)

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return map(lambda name: self._modules[name], self.names)

    @property
    def choices(self):
        warnings.warn(
            "layer_choice.choices is deprecated. Use `list(layer_choice)` instead.",
            DeprecationWarning,
        )
        return list(self)

    def forward(self, *args, **kwargs):

        out, mask = self.mutator.on_forward_layer_choice(self, *args, **kwargs)
        if self.return_mask:
            return out, mask
        return out

class InputChoice(Mutable):

    NO_KEY = ""

    def __init__(
        self,
        n_candidates=None,
        choose_from=None,
        n_chosen=None,
        reduction="sum",
        return_mask=False,
        key=None,
    ):
        super().__init__(key=key)
        assert n_candidates is not None or choose_from is not None, (
            "At least one of `n_candidates` and `choose_from`" "must be not None."
        )
        if choose_from is not None and n_candidates is None:
            n_candidates = len(choose_from)
        elif choose_from is None and n_candidates is not None:
            choose_from = [self.NO_KEY] * n_candidates
        assert n_candidates == len(
            choose_from
        ), "Number of candidates must be equal to the length of `choose_from`."
        assert n_candidates > 0, "Number of candidates must be greater than 0."
        assert n_chosen is None or 0 <= n_chosen <= n_candidates, (
            "Expected selected number must be None or no more "
            "than number of candidates."
        )

        self.n_candidates = n_candidates
        self.choose_from = choose_from.copy()
        self.n_chosen = n_chosen
        self.reduction = reduction
        self.return_mask = return_mask

    def forward(self, optional_inputs):

        optional_input_list = optional_inputs
        if isinstance(optional_inputs, dict):
            optional_input_list = [optional_inputs[tag] for tag in self.choose_from]
        assert isinstance(
            optional_input_list, list
        ), "Optional input list must be a list, not a {}.".format(
            type(optional_input_list)
        )
        assert (
            len(optional_inputs) == self.n_candidates
        ), "Length of the input list must be equal to number of candidates."
        out, mask = self.mutator.on_forward_input_choice(self, optional_input_list)
        if self.return_mask:
            return out, mask
        return out
