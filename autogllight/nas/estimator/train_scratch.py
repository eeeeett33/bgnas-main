import torch.nn.functional as F
from ..space import BaseSpace
from .base import BaseEstimator
from autogllight.utils.evaluation import Auc
from autogllight.utils.backend.op import *

class TrainScratchEstimator(BaseEstimator):

    def __init__(self, trainer, evaluation=[Auc()], trainer_obj=None):
        super().__init__(None, evaluation)
        self.trainer = trainer
        self.trainer_obj = trainer_obj
        self.evaluation = evaluation

    def infer(self, model: BaseSpace, dataset, mask="train", *args, **kwargs):
        metrics, loss = self.trainer(model, dataset, mask, self.evaluation, *args, **kwargs)
        return metrics, loss
