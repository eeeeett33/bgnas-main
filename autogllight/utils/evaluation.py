import re
import numpy as np
import typing as _typing
import torch
from sklearn.metrics import (
    f1_score,
    log_loss,
    accuracy_score,
    roc_auc_score,
    label_ranking_average_precision_score,
    precision_recall_curve,
    auc, precision_score, recall_score,
)

class Evaluation:
    def __int__(self, *args, **kwargs):
        pass

    @staticmethod
    def get_eval_name() -> str:

        raise NotImplementedError

    @staticmethod
    def is_higher_better() -> bool:

        raise NotImplementedError

    @staticmethod
    def evaluate(predict, label) -> float:

        raise NotImplementedError

class EvaluatorUtility:

    class PredictionBatchCumulativeBuilder:

        def __init__(self):
            self.__indexes_in_integral_data: _typing.Optional[np.ndarray] = None
            self.__prediction: _typing.Optional[np.ndarray] = None

        def clear_batches(
                self, *__args, **__kwargs
        ) -> "EvaluatorUtility.PredictionBatchCumulativeBuilder":
            self.__indexes_in_integral_data = None
            self.__prediction = None
            return self

        def add_batch(
                self, indexes_in_integral_data: np.ndarray, batch_prediction: np.ndarray
        ) -> "EvaluatorUtility.PredictionBatchCumulativeBuilder":
            if not (
                    isinstance(indexes_in_integral_data, np.ndarray)
                    and isinstance(batch_prediction, np.ndarray)
                    and len(indexes_in_integral_data.shape) == 1
            ):
                raise TypeError
            elif indexes_in_integral_data.shape[0] != batch_prediction.shape[0]:
                raise ValueError

            if self.__indexes_in_integral_data is None:
                if (
                        indexes_in_integral_data.shape
                        != np.unique(indexes_in_integral_data).shape
                ):
                    raise ValueError(
                        f"There exists duplicate index "
                        f"in the argument indexes_in_integral_data {indexes_in_integral_data}"
                    )
                else:
                    self.__indexes_in_integral_data: np.ndarray = np.unique(
                        indexes_in_integral_data
                    )
            else:
                __indexes_in_integral_data = np.concatenate(
                    (self.__indexes_in_integral_data, indexes_in_integral_data)
                )
                if (
                        __indexes_in_integral_data.shape
                        != np.unique(__indexes_in_integral_data).shape
                ):
                    raise ValueError
                else:
                    self.__indexes_in_integral_data: np.ndarray = (
                        __indexes_in_integral_data
                    )

            if self.__prediction is None:
                self.__prediction: np.ndarray = batch_prediction
            else:
                self.__prediction: np.ndarray = np.concatenate(
                    (self.__prediction, batch_prediction)
                )

            return self

        def compose(
                self, __sorted: bool = True, **__kwargs
        ) -> _typing.Tuple[np.ndarray, np.ndarray]:
            if __sorted:
                sorted_index = np.argsort(self.__indexes_in_integral_data)
                return (
                    self.__indexes_in_integral_data[sorted_index],
                    self.__prediction[sorted_index],
                )
            else:
                return self.__indexes_in_integral_data, self.__prediction

EVALUATE_DICT: _typing.Dict[str, _typing.Type[Evaluation]] = {}

def register_evaluate(*name):
    def register_evaluate_cls(cls):
        for n in name:
            if n in EVALUATE_DICT:
                raise ValueError("Cannot register duplicate evaluator ({})".format(n))
            if not issubclass(cls, Evaluation):
                raise ValueError(
                    "Evaluator ({}: {}) must extend Evaluation".format(n, cls.__name__)
                )
            EVALUATE_DICT[n] = cls
        return cls

    return register_evaluate_cls

def get_feval(feval):
    if isinstance(feval, str):
        return EVALUATE_DICT[feval]
    if isinstance(feval, type) and issubclass(feval, Evaluation):
        return feval
    if isinstance(feval, _typing.Sequence):
        return [get_feval(f) for f in feval]
    raise ValueError("feval argument of type", type(feval), "is not supported!")

def map_eva(evaluate):
    eva_list = []
    for eva in evaluate:
        match = re.search(r'@(\d+)', eva)
        if match:
            k = int(match.group(1))
            key = eva.split('@')[0] + f"@k"
        else:
            key = eva

        if key not in EVALUATE_DICT:
            raise KeyError(f"Model evaluation metric error: {key}")

        eva_result = EVALUATE_DICT[key](k=k) if match else EVALUATE_DICT[key]()
        eva_list.append(eva_result)
    return eva_list

class EvaluationUniversalRegistry:
    @classmethod
    def register_evaluation(
            cls, *names
    ) -> _typing.Callable[[_typing.Type[Evaluation]], _typing.Type[Evaluation]]:
        def _register_evaluation(
                _class: _typing.Type[Evaluation],
        ) -> _typing.Type[Evaluation]:
            for n in names:
                if n in EVALUATE_DICT:
                    raise ValueError(
                        "Cannot register duplicate evaluator ({})".format(n)
                    )
                if not issubclass(_class, Evaluation):
                    raise ValueError(
                        "Evaluator ({}: {}) must extend Evaluation".format(
                            n, cls.__name__
                        )
                    )
                EVALUATE_DICT[n] = _class
            return _class

        return _register_evaluation

@register_evaluate("logloss")
class Logloss(Evaluation):
    @staticmethod
    def get_eval_name():
        return "logloss"

    @staticmethod
    def is_higher_better():

        return False

    @staticmethod
    def evaluate(predict, label):

        return log_loss(label, predict)

@register_evaluate("auc_pr", "AUC-PR")
class AucPR(Evaluation):
    @staticmethod
    def get_eval_name():

        return "auc_pr"

    @staticmethod
    def is_higher_better():

        return True

    @staticmethod
    def evaluate(predict, label):

        if len(predict.shape) == 1:
            pos_predict = predict
        else:
            assert (
                    predict.shape[1] == 2
            ), "Cannot use auc_pr on given data with %d classes!" % (predict.shape[1])
            pos_predict = predict[:, 1]

        precision, recall, _ = precision_recall_curve(label, pos_predict)

        auc_pr = auc(recall, precision)

        return auc_pr

@register_evaluate("auc", "ROC-AUC")
class Auc(Evaluation):
    @staticmethod
    def get_eval_name():
        return "auc"

    @staticmethod
    def is_higher_better():

        return True

    @staticmethod
    def evaluate(predict, label):

        if len(predict.shape) == 1:
            pos_predict = predict
        else:
            assert (
                    predict.shape[1] == 2
            ), "Cannot use auc on given data with %d classes!" % (predict.shape[1])
            pos_predict = predict[:, 1]
        return roc_auc_score(label, pos_predict)

@register_evaluate("Accuracy", "acc")
class Accuracy(Evaluation):
    @staticmethod
    def get_eval_name() -> str:
        return "acc"

    @staticmethod
    def is_higher_better() -> bool:
        return True

    @staticmethod
    def evaluate(predict, label) -> float:

        if len(predict.shape) == 2:
            predict = np.argmax(predict, axis=1)

        else:
            predict = [1 if p > 0.5 else 0 for p in predict]
        return accuracy_score(label, predict)

@register_evaluate("mrr")
class Mrr(Evaluation):
    @staticmethod
    def get_eval_name():
        return "mrr"

    @staticmethod
    def is_higher_better():

        return True

    @staticmethod
    def evaluate(predict, label):

        if len(predict.shape) == 2:
            assert (
                    predict.shape[1] == 2
            ), "Cannot use mrr on given data with %d classes!" % (predict.shape[1])
            pos_predict = predict[:, 1]
        else:
            pos_predict = predict
        return label_ranking_average_precision_score(label, pos_predict)

@register_evaluate("F1", "F1-Score", "f1_score", "f1")
class F1(Evaluation):
    @staticmethod
    def get_eval_name() -> str:
        return "f1"

    @staticmethod
    def is_higher_better() -> bool:
        return True

    @staticmethod
    def evaluate(predict, label) -> float:
        pred_labels = (predict >= 0.75).astype(int)
        f1 = f1_score(label, pred_labels)
        return f1

@register_evaluate("Precision@k", "pre@k")
class TopKPrecision(Evaluation):
    def __init__(self, k=10):

        self.k = k

    def get_eval_name(self):

        return f"pre@{self.k}"

    @staticmethod
    def is_higher_better():

        return True

    def evaluate(self, predict, label):

        k = min(self.k, len(predict))

        top_k_indices = np.argsort(-predict)[:k]

        num_correct = np.sum(label[top_k_indices])

        precision_at_k = num_correct / k

        return precision_at_k
