
import torch
from abc import abstractmethod
from autogllight.utils import get_device

class BaseNAS:

    def __init__(self, device="auto") -> None:
        self.device = get_device(device)
        self.selection = None

    def to(self, device):

        self.device = get_device(device)

    @abstractmethod
    def search(self, space, dataset, estimator):

        raise NotImplementedError()

    def get_selection(self):
        return self.selection
