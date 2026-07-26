import dataclasses
import random
import numpy as np

@dataclasses.dataclass
class Individual:

    x: dict
    y: float

class MutationSampler:

    def __init__(self, nas_modules, mutation_prob):
        selection_range = {}
        for k, v in nas_modules:
            selection_range[k] = len(v)
        self.selection_dict = selection_range
        self.mutation_prob = mutation_prob

    def resample(self, parent):
        search_space = self.selection_dict
        child = {}
        for k, v in parent.items():
            if random.uniform(0, 1) < self.mutation_prob:
                child[k] = np.random.choice(
                    range(search_space[k])
                )
            else:
                child[k] = v
        return child

class UniformSampler:

    def __init__(self, nas_modules):
        selection_range = {}
        for k, v in nas_modules:
            selection_range[k] = len(v)
        self.selection_dict = selection_range

    def resample(self):
        selection = {}
        for k, v in self.selection_dict.items():
            selection[k] = np.random.choice(range(v))
        return selection
