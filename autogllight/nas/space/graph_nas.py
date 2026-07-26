import typing as _typ
import torch
import torch.nn.functional as F
from torch import nn

from .nni import mutables
from .base import BaseSpace
from .operation import gnn_map, act_map_nn, map_nn

GRAPHNAS_DEFAULT_GNN_OPS = [
    "gat_8",
    "gat_6",
    "gat_4",
    "gat_2",
    "gat_1",
    "gcn",
    "sage",
    "arma",
    "linear",
]

GRAPHNAS_DEFAULT_ACT_OPS = [
    "sigmoid",
    "tanh",
    "relu",
    "linear",
    "elu",
]

GRAPHNAS_DEFAULT_CON_OPS = [
    "concat"
]

class GraphBenchmarkingSpace(BaseSpace):
    def __init__(
        self,
        hidden_dim: _typ.Optional[int] = 64,
        layer_number: _typ.Optional[int] = 4,
        dropout: _typ.Optional[float] = 0.9,
        input_dim: _typ.Optional[int] = None,
        output_dim: _typ.Optional[int] = None,
        gnn_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_GNN_OPS,
        act_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_ACT_OPS,
        con_ops: _typ.Sequence[_typ.Union[str, _typ.Any]] = GRAPHNAS_DEFAULT_CON_OPS,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gnn_ops = gnn_ops
        self.act_ops = act_ops
        self.con_ops = con_ops
        self.dropout = dropout

    def build_graph(self):
        self.preproc0 = nn.Linear(self.input_dim, self.hidden_dim)

        node_labels = [mutables.InputChoice.NO_KEY]

        for layer in range(0, self.layer_number):
            in_key = f"in_{layer}"
            self.setInputChoice(
                layer, choose_from=node_labels, n_chosen=1, key=in_key,
            )

            op_key = f"op_{layer}"
            op_candidates = [
                gnn_map(op, self.hidden_dim, self.hidden_dim) for op in self.gnn_ops
            ]
            self.setLayerChoice(layer, op_candidates, key=op_key)

            node_labels.append(op_key)

        self.setLayerChoice(layer + 1, [act_map_nn(a) for a in self.act_ops], key="act")

        if len(self.con_ops) > 1:
            self.setLayerChoice(layer + 2, map_nn(self.con_ops), key="concat")

        self.classifier1 = nn.Linear(
            self.hidden_dim * self.layer_number, self.output_dim
        )

    def forward(self, data, branch_gates=None, return_branch_activations=False):

        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        prev_nodes_out = [self.preproc0(x)]

        branch_activations = {} if return_branch_activations else None

        for layer in range(0, self.layer_number):
            node_in = getattr(self, f"in_{layer}")(prev_nodes_out)
            op = getattr(self, f"op_{layer}")
            node_out = op(node_in, data.edge_index)
            if branch_gates is not None:
                gate = branch_gates.get(f"op_{layer}")
                if gate is not None:
                    node_out = gate * node_out
            if branch_activations is not None:
                branch_activations[f"op_{layer}"] = node_out
            prev_nodes_out.append(node_out)
        act = getattr(self, "act")
        if len(self.con_ops) > 1:
            con = getattr(self, "concat")
        elif len(self.con_ops) == 1:
            con = self.con_ops[0]

        states = prev_nodes_out
        if con == "concat":
            x = torch.cat(states[1:], dim=1)
        else:
            tmp = states[1]
            for i in range(2, len(states)):
                if con == "add":
                    tmp = torch.add(tmp, states[i])
                elif con == "product":
                    tmp = torch.mul(tmp, states[i])
            x = tmp
        x = act(x)
        if con == "concat":
            x = self.classifier1(x)

        out = F.log_softmax(x, dim=1)
        if return_branch_activations:
            return out, branch_activations
        return out

