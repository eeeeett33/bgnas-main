

from . import mutables
from .fixed import FixedArchitecture
import json
import logging
from torch import nn
from .mutables import Mutable, InputChoice, LayerChoice

_logger = logging.getLogger(__name__)

class OrderedMutable:

    def __init__(self, order):
        self.order = order

class OrderedLayerChoice(OrderedMutable, mutables.LayerChoice):
    def __init__(
        self, order, op_candidates, reduction="sum", return_mask=False, key=None
    ):
        OrderedMutable.__init__(self, order)
        mutables.LayerChoice.__init__(self, op_candidates, reduction, return_mask, key)

class OrderedInputChoice(OrderedMutable, mutables.InputChoice):
    def __init__(
        self,
        order,
        n_candidates=None,
        choose_from=None,
        n_chosen=None,
        reduction="sum",
        return_mask=False,
        key=None,
    ):
        OrderedMutable.__init__(self, order)
        mutables.InputChoice.__init__(
            self, n_candidates, choose_from, n_chosen, reduction, return_mask, key
        )

class StrModule(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.str = name

    def forward(self, *args, **kwargs):
        return self.str

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.str)

def map_nn(names):

    return [StrModule(x) for x in names]

class FixedInputChoice(nn.Module):

    def __init__(self, mask):
        self.mask_len = len(mask)
        for i in range(self.mask_len):
            if mask[i]:
                self.selected = i
                break
        super().__init__()

    def forward(self, optional_inputs):
        if len(optional_inputs) == self.mask_len:
            return optional_inputs[self.selected]

class CleanFixedArchitecture(FixedArchitecture):

    def __init__(self, model, fixed_arc, strict=True, verbose=True):
        super().__init__(model, fixed_arc, strict, verbose)

    def replace_all_choice(self, module=None, prefix=""):

        if module is None:
            module = self.model
        for name, mutable in module.named_children():
            global_name = (prefix + "." if prefix else "") + name
            if isinstance(mutable, OrderedLayerChoice):
                chosen = self._fixed_arc[mutable.key]
                if sum(chosen) == 1 and max(chosen) == 1 and not mutable.return_mask:
                    setattr(module, name, mutable[chosen.index(1)])
                else:
                    for ch, n in zip(chosen, mutable.names):
                        if ch == 0 and not isinstance(ch, float):
                            setattr(mutable, n, None)
            elif isinstance(mutable, OrderedInputChoice):
                chosen = self._fixed_arc[mutable.key]
                setattr(module, name, FixedInputChoice(chosen))
            else:
                self.replace_all_choice(mutable, global_name)

def apply_fixed_architecture(model, fixed_arc, verbose=True):

    if isinstance(fixed_arc, str):
        with open(fixed_arc) as f:
            fixed_arc = json.load(f)
    architecture = CleanFixedArchitecture(model, fixed_arc, verbose)
    architecture.reset()

    architecture.replace_all_choice()
    return architecture

def get_module_order(root_module):
    key2order = {}

    def apply(m):
        for name, child in m.named_children():
            if isinstance(child, Mutable):
                key2order[child.key] = child.order
            else:
                apply(child)

    apply(root_module)
    return key2order

def sort_replaced_module(k2o, modules):
    modules = sorted(modules, key=lambda x: k2o[x[0]])
    return modules

def _replace_module_with_type(root_module, init_fn, type_name, modules):
    if modules is None:
        modules = []

    def apply(m):
        for name, child in m.named_children():
            if isinstance(child, type_name):
                setattr(m, name, init_fn(child))
                modules.append((child.key, getattr(m, name)))
            else:
                apply(child)

    apply(root_module)
    return modules

def replace_layer_choice(root_module, init_fn, modules=None):

    return _replace_module_with_type(root_module, init_fn, (LayerChoice,), modules)

def replace_input_choice(root_module, init_fn, modules=None):

    return _replace_module_with_type(root_module, init_fn, (InputChoice,), modules)
