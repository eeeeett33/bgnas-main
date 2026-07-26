
import logging
from collections import OrderedDict

import numpy as np
import torch

_counter = 0

_logger = logging.getLogger(__name__)

def global_mutable_counting():

    global _counter
    _counter += 1
    return _counter

def _reset_global_mutable_counting():

    global _counter
    _counter = 0

def to_device(obj, device):

    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device(t, device) for t in obj)
    if isinstance(obj, list):
        return [to_device(t, device) for t in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))

def to_list(arr):
    if torch.is_tensor(arr):
        return arr.cpu().numpy().tolist()
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (list, tuple)):
        return list(arr)
    return arr

class AverageMeterGroup:

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data):

        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):

        return "  ".join(v.summary() for v in self.meters.values())

class AverageMeter:

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)

class StructuredMutableTreeNode:

    def __init__(self, mutable):
        self.mutable = mutable
        self.children = []

    def add_child(self, mutable):

        self.children.append(StructuredMutableTreeNode(mutable))
        return self.children[-1]

    def type(self):

        return type(self.mutable)

    def __iter__(self):
        return self.traverse()

    def traverse(self, order="pre", deduplicate=True, memo=None):

        if memo is None:
            memo = set()
        assert order in ["pre", "post"]
        if order == "pre":
            if self.mutable is not None:
                if not deduplicate or self.mutable.key not in memo:
                    memo.add(self.mutable.key)
                    yield self.mutable
        for child in self.children:
            for m in child.traverse(order=order, deduplicate=deduplicate, memo=memo):
                yield m
        if order == "post":
            if self.mutable is not None:
                if not deduplicate or self.mutable.key not in memo:
                    memo.add(self.mutable.key)
                    yield self.mutable
