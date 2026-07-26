

from abc import abstractmethod
from ..space import BaseSpace
from typing import Tuple
import torch.nn.functional as F
import torch

class BaseEstimator:

    def __init__(self, loss_f, evaluation):

        self.loss_f = loss_f
        self.evaluation = evaluation

    def setLossFunction(self, loss_f: str):
        self.loss_f = loss_f

    def setEvaluation(self, evaluation):
        self.evaluation = evaluation

    @abstractmethod
    def infer(
        self, model: BaseSpace, loader, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError()
