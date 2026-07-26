
import logging
from collections import defaultdict

import numpy as np
import torch

from .base_mutator import BaseMutator
from .mutables import LayerChoice, InputChoice
from .utils import to_list

logger = logging.getLogger(__name__)

class Mutator(BaseMutator):
    def __init__(self, model):
        super().__init__(model)
        self._cache = dict()
        self._connect_all = False

    def sample_search(self):

        raise NotImplementedError

    def sample_final(self):

        raise NotImplementedError

    def reset(self):

        self._cache = self.sample_search()

    def export(self):

        sampled = self.sample_final()
        result = dict()
        for mutable in self.mutables:
            if not isinstance(mutable, (LayerChoice, InputChoice)):
                continue
            result[mutable.key] = self._convert_mutable_decision_to_human_readable(
                mutable, sampled.pop(mutable.key)
            )
        if sampled:
            raise ValueError(
                "Unexpected keys returned from 'sample_final()': %s",
                list(sampled.keys()),
            )
        return result

    def status(self):

        data = dict()
        for k, v in self._cache.items():
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy().tolist()
            if isinstance(v, np.ndarray):
                v = v.astype(np.float32).tolist()
            data[k] = v
        return data

    def graph(self, inputs):

        if not torch.__version__.startswith("1.4"):
            logger.warning(
                "Graph is only tested with PyTorch 1.4. Other versions might not work."
            )
        from .graph_utils import build_graph
        from google.protobuf import json_format

        try:
            self._connect_all = True
            graph_def, _ = build_graph(self.model, inputs, verbose=False)
            result = json_format.MessageToDict(graph_def)
        finally:
            self._connect_all = False

        result["mutable"] = defaultdict(list)
        for mutable in self.mutables.traverse(deduplicate=False):
            modules = mutable.name.split(".")
            path = [{"type": self.model.__class__.__name__, "name": ""}]
            m = self.model
            for module in modules:
                m = getattr(m, module)
                path.append({"type": m.__class__.__name__, "name": module})
            result["mutable"][mutable.key].append(path)
        return result

    def on_forward_layer_choice(self, mutable, *args, **kwargs):

        if self._connect_all:
            return (
                self._all_connect_tensor_reduction(
                    mutable.reduction, [op(*args, **kwargs) for op in mutable]
                ),
                torch.ones(len(mutable)).bool(),
            )

        def _map_fn(op, args, kwargs):
            return op(*args, **kwargs)

        mask = self._get_decision(mutable)
        assert len(mask) == len(
            mutable
        ), "Invalid mask, expected {} to be of length {}.".format(mask, len(mutable))
        out, mask = self._select_with_mask(
            _map_fn, [(choice, args, kwargs) for choice in mutable], mask
        )
        return self._tensor_reduction(mutable.reduction, out), mask

    def on_forward_input_choice(self, mutable, tensor_list):

        if self._connect_all:
            return (
                self._all_connect_tensor_reduction(mutable.reduction, tensor_list),
                torch.ones(mutable.n_candidates).bool(),
            )
        mask = self._get_decision(mutable)
        assert (
            len(mask) == mutable.n_candidates
        ), "Invalid mask, expected {} to be of length {}.".format(
            mask, mutable.n_candidates
        )
        out, mask = self._select_with_mask(
            lambda x: x, [(t,) for t in tensor_list], mask
        )
        return self._tensor_reduction(mutable.reduction, out), mask

    def _select_with_mask(self, map_fn, candidates, mask):

        if (
            (isinstance(mask, list) and len(mask) >= 1 and isinstance(mask[0], bool))
            or (isinstance(mask, np.ndarray) and mask.dtype == np.bool)
            or "BoolTensor" in mask.type()
        ):
            out = [map_fn(*cand) for cand, m in zip(candidates, mask) if m]
        elif (
            (
                isinstance(mask, list)
                and len(mask) >= 1
                and isinstance(mask[0], (float, int))
            )
            or (
                isinstance(mask, np.ndarray)
                and mask.dtype in (np.float32, np.float64, np.int32, np.int64)
            )
            or "FloatTensor" in mask.type()
        ):
            out = [map_fn(*cand) * m for cand, m in zip(candidates, mask) if m]
        else:
            raise ValueError("Unrecognized mask '%s'" % mask)
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask)
        return out, mask

    def _tensor_reduction(self, reduction_type, tensor_list):
        if reduction_type == "none":
            return tensor_list
        if not tensor_list:
            return None
        if len(tensor_list) == 1:
            return tensor_list[0]
        if reduction_type == "sum":
            return sum(tensor_list)
        if reduction_type == "mean":
            return sum(tensor_list) / len(tensor_list)
        if reduction_type == "concat":
            return torch.cat(tensor_list, dim=1)
        raise ValueError('Unrecognized reduction policy: "{}"'.format(reduction_type))

    def _all_connect_tensor_reduction(self, reduction_type, tensor_list):
        if reduction_type == "none":
            return tensor_list
        if reduction_type == "concat":
            return torch.cat(tensor_list, dim=1)
        return torch.stack(tensor_list).sum(0)

    def _get_decision(self, mutable):

        if mutable.key not in self._cache:
            raise ValueError('"{}" not found in decision cache.'.format(mutable.key))
        result = self._cache[mutable.key]
        logger.debug("Decision %s: %s", mutable.key, result)
        return result

    def _convert_mutable_decision_to_human_readable(self, mutable, sampled):
        multihot_list = to_list(sampled)
        converted = None
        if all([t == 0 or t == 1 for t in multihot_list]):
            if isinstance(mutable, LayerChoice):
                assert len(multihot_list) == len(mutable), (
                    "Results returned from 'sample_final()' (%s: %s) either too short or too long."
                    % (mutable.key, multihot_list)
                )
                if len(set(mutable.names)) == len(mutable) and not all(
                    d.isdigit() for d in mutable.names
                ):
                    converted = [
                        name for i, name in enumerate(mutable.names) if multihot_list[i]
                    ]
                else:
                    converted = [
                        i for i in range(len(multihot_list)) if multihot_list[i]
                    ]
            if isinstance(mutable, InputChoice):
                assert len(multihot_list) == mutable.n_candidates, (
                    "Results returned from 'sample_final()' (%s: %s) either too short or too long."
                    % (mutable.key, multihot_list)
                )
                if len(set(mutable.choose_from)) == mutable.n_candidates:
                    converted = [
                        name
                        for i, name in enumerate(mutable.choose_from)
                        if multihot_list[i]
                    ]
                else:
                    converted = [
                        i for i in range(len(multihot_list)) if multihot_list[i]
                    ]
        if converted is not None:
            if len(converted) == 1:
                converted = converted[0]
        else:
            converted = multihot_list
        return converted
