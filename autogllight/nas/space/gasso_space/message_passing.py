import os
import re
import inspect
import os.path as osp
from uuid import uuid1
from itertools import chain
from inspect import Parameter
from typing import List, Optional, Set
from torch_geometric.typing import Adj, Size

import torch
from torch import Tensor
from jinja2 import Template
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr

from .utils.helpers import expand_left
from .utils.jit import class_from_module_repr
from .utils.typing import sanitize, split_types_repr, parse_types, resolve_types
from .utils.inspector import Inspector, func_header_repr, func_body_repr

class MessagePassing(torch.nn.Module):

    special_args: Set[str] = {
        "edge_index",
        "adj_t",
        "edge_index_i",
        "edge_index_j",
        "size",
        "size_i",
        "size_j",
        "ptr",
        "index",
        "dim_size",
    }

    def __init__(
        self,
        aggr: Optional[str] = "add",
        flow: str = "source_to_target",
        node_dim: int = -2,
    ):

        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ["add", "mean", "max", None]

        self.flow = flow
        assert self.flow in ["source_to_target", "target_to_source"]

        self.node_dim = node_dim

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.message_and_aggregate, pop_first=True)
        self.inspector.inspect(self.update, pop_first=True)

        self.__user_args__ = self.inspector.keys(
            ["message", "aggregate", "update"]
        ).difference(self.special_args)
        self.__fused_user_args__ = self.inspector.keys(
            ["message_and_aggregate", "update"]
        ).difference(self.special_args)

        self.fuse = self.inspector.implements("message_and_aggregate")

        self.__explain__ = False
        self.__edge_mask__ = None

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype == torch.long
            assert edge_index.dim() == 2
            assert edge_index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        elif isinstance(edge_index, SparseTensor):
            if self.flow == "target_to_source":
                raise ValueError(
                    (
                        'Flow direction "target_to_source" is invalid for '
                        "message propagation via `torch_sparse.SparseTensor`. If "
                        "you really want to make use of a reverse message "
                        "passing flow, pass in the transposed sparse tensor to "
                        "the message passing module, e.g., `adj_t.t()`."
                    )
                )
            the_size[0] = edge_index.sparse_size(1)
            the_size[1] = edge_index.sparse_size(0)
            return the_size

        raise ValueError(
            (
                "`MessagePassing.propagate` only supports `torch.LongTensor` of "
                "shape `[2, num_messages]` or `torch_sparse.SparseTensor` for "
                "argument `edge_index`."
            )
        )

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (
                    f"Encountered tensor with size {src.size(self.node_dim)} in "
                    f"dimension {self.node_dim}, but expected size {the_size}."
                )
            )

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return src.index_select(self.node_dim, index)
        elif isinstance(edge_index, SparseTensor):
            if dim == 1:
                rowptr = edge_index.storage.rowptr()
                rowptr = expand_left(rowptr, dim=self.node_dim, dims=src.dim())
                return gather_csr(src, rowptr)
            elif dim == 0:
                col = edge_index.storage.col()
                return src.index_select(self.node_dim, col)
        raise ValueError

    def __collect__(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == "source_to_target" else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ["_i", "_j"]:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = 0 if arg[-2:] == "_j" else 1
                data = kwargs.get(arg[:-2], Parameter.empty)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self.__set_size__(size, dim, data)
                    data = self.__lift__(data, edge_index, j if arg[-2:] == "_j" else i)

                out[arg] = data

        if isinstance(edge_index, Tensor):
            out["adj_t"] = None
            out["edge_index"] = edge_index
            out["edge_index_i"] = edge_index[i]
            out["edge_index_j"] = edge_index[j]
            out["ptr"] = None
        elif isinstance(edge_index, SparseTensor):
            out["adj_t"] = edge_index
            out["edge_index"] = None
            out["edge_index_i"] = edge_index.storage.row()
            out["edge_index_j"] = edge_index.storage.col()
            out["ptr"] = edge_index.storage.rowptr()
            out["edge_weight"] = edge_index.storage.value()
            out["edge_attr"] = edge_index.storage.value()
            out["edge_type"] = edge_index.storage.value()

        out["index"] = out["edge_index_i"]
        out["size"] = size
        out["size_i"] = size[1] or size[0]
        out["size_j"] = size[0] or size[1]
        out["dim_size"] = out["size_i"]

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):

        size = self.__check_input__(edge_index, size)

        if isinstance(edge_index, SparseTensor) and self.fuse and not self.__explain__:
            coll_dict = self.__collect__(
                self.__fused_user_args__, edge_index, size, kwargs
            )

            msg_aggr_kwargs = self.inspector.distribute(
                "message_and_aggregate", coll_dict
            )
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute("update", coll_dict)
            return self.update(out, **update_kwargs)

        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)

            msg_kwargs = self.inspector.distribute("message", coll_dict)
            out = self.message(**msg_kwargs)

            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute("update", coll_dict)
            return self.update(out, **update_kwargs)

    def message(self, x_j: Tensor) -> Tensor:

        return x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:

        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(
                inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr
            )

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:

        raise NotImplementedError

    def update(self, inputs: Tensor) -> Tensor:

        return inputs

    @torch.jit.unused
    def jittable(self, typing: Optional[str] = None):

        if hasattr(self, "propagate_type"):
            prop_types = {k: sanitize(str(v)) for k, v in self.propagate_type.items()}
        else:
            source = inspect.getsource(self.__class__)
            match = re.search(r"#\s*propagate_type:\s*\((.*)\)", source)
            if match is None:
                raise TypeError(
                    "TorchScript support requires the definition of the types "
                    "passed to `propagate()`. Please specificy them via\n\n"
                    'propagate_type = {"arg1": type1, "arg2": type2, ... }\n\n'
                    "or via\n\n"
                    "# propagate_type: (arg1: type1, arg2: type2, ...)\n\n"
                    "inside the `MessagePassing` module."
                )
            prop_types = split_types_repr(match.group(1))
            prop_types = dict([re.split(r"\s*:\s*", t) for t in prop_types])

        collect_types = self.inspector.types(["message", "aggregate", "update"])

        forward_types = parse_types(self.forward)
        forward_types = [resolve_types(*types) for types in forward_types]
        forward_types = list(chain.from_iterable(forward_types))

        keep_annotation = len(forward_types) < 2
        forward_header = func_header_repr(self.forward, keep_annotation)
        forward_body = func_body_repr(self.forward, keep_annotation)

        if keep_annotation:
            forward_types = []
        elif typing is not None:
            forward_types = []
            forward_body = 8 * " " + f"# type: {typing}\n{forward_body}"

        root = os.path.dirname(osp.realpath(__file__))
        with open(osp.join(root, "message_passing.jinja"), "r") as f:
            template = Template(f.read())

        uid = uuid1().hex[:6]
        cls_name = f"{self.__class__.__name__}Jittable_{uid}"
        jit_module_repr = template.render(
            uid=uid,
            module=str(self.__class__.__module__),
            cls_name=cls_name,
            parent_cls_name=self.__class__.__name__,
            prop_types=prop_types,
            collect_types=collect_types,
            user_args=self.__user_args__,
            forward_header=forward_header,
            forward_types=forward_types,
            forward_body=forward_body,
            msg_args=self.inspector.keys(["message"]),
            aggr_args=self.inspector.keys(["aggregate"]),
            msg_and_aggr_args=self.inspector.keys(["message_and_aggregate"]),
            update_args=self.inspector.keys(["update"]),
            check_input=inspect.getsource(self.__check_input__)[:-1],
            lift=inspect.getsource(self.__lift__)[:-1],
        )

        cls = class_from_module_repr(cls_name, jit_module_repr)
        module = cls.__new__(cls)
        module.__dict__ = self.__dict__.copy()
        module.jittable = None

        return module
