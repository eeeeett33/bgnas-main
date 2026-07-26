from torch import nn
import torch

class PathSamplingLayerChoice(nn.Module):

    def __init__(self, layer_choice):
        super(PathSamplingLayerChoice, self).__init__()
        self.op_names = []
        for name, module in layer_choice.named_children():
            self.add_module(name, module)
            self.op_names.append(name)
        assert self.op_names, "There has to be at least one op to choose from."
        self.sampled = None

    def forward(self, *args, **kwargs):
        assert (
            self.sampled is not None
        ), "At least one path needs to be sampled before fprop."
        if isinstance(self.sampled, list):
            return sum(
                [getattr(self, self.op_names[i])(*args, **kwargs) for i in self.sampled]
            )
        else:
            return getattr(self, self.op_names[self.sampled])(
                *args, **kwargs
            )

    def sampled_choices(self):
        if self.sampled is None:
            return []
        elif isinstance(self.sampled, list):
            return [
                getattr(self, self.op_names[i]) for i in self.sampled
            ]
        else:
            return [
                getattr(self, self.op_names[self.sampled])
            ]

    def __len__(self):
        return len(self.op_names)

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))

    def __repr__(self):
        return (
            f"PathSamplingLayerChoice(op_names={self.op_names}, chosen={self.sampled})"
        )

class PathSamplingInputChoice(nn.Module):

    def __init__(self, input_choice):
        super(PathSamplingInputChoice, self).__init__()
        self.n_candidates = input_choice.n_candidates
        self.n_chosen = input_choice.n_chosen
        self.sampled = None

    def forward(self, input_tensors):
        if isinstance(self.sampled, list):
            return sum(
                [input_tensors[t] for t in self.sampled]
            )
        else:
            return input_tensors[self.sampled]

    def __len__(self):
        return self.n_candidates

    @property
    def mask(self):
        return _get_mask(self.sampled, len(self))

    def __repr__(self):
        return f"PathSamplingInputChoice(n_candidates={self.n_candidates}, chosen={self.sampled})"
