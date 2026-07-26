#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import ARMAConv, ChebConv, GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.utils import to_undirected

ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "autogllight" / "examples"
NBGRAPH_DIR = ROOT / "NAS-Bench-Graph-main" / "nas-bench-graph"
for p in (ROOT, EXAMPLES_DIR, NBGRAPH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from help_funcs import prune_unrelated_edge, prune_unrelated_edge_isolated
from utils import GraphDataLoader, subgraph

DEFAULT_SEEDS = (666, 777, 888, 999)

@dataclass
class TargetTrainConfig:
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4

@dataclass
class TriggerTrainConfig:
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 5e-4
    trigger_size: int = 3
    target_class: int = 0
    target_loss_weight: float = 1.0
    homo_loss_weight: float = 100.0
    homo_boost_thrd: float = 0.5
    thrd: float = 0.5
    debug: bool = False

@dataclass
class EvalConfig:
    defense_mode: str = "prune"
    prune_thr: float = 0.1

@dataclass
class NTKConfig:
    compute_at: str = "init"
    proxy_split: str = "train"
    proxy_batch_size: int = 16
    similarity_method: str = "fro_cos"

@dataclass
class SplitContext:
    train_edge_index: Tensor
    idx_train: Tensor
    idx_val: Tensor
    idx_clean_test: Tensor
    idx_atk: Tensor
    unlabeled_idx: Tensor
    induct_edge_index: Tensor
    induct_edge_weights: Tensor

class GradWhere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: Tensor, threshold: float, device: torch.device) -> Tensor:
        ctx.save_for_backward(input_tensor)
        return torch.where(
            input_tensor > threshold,
            torch.tensor(1.0, device=device, dtype=input_tensor.dtype),
            torch.tensor(0.0, device=device, dtype=input_tensor.dtype),
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output.clone(), None, None

class LinearConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.linear(x)

class SkipConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.proj(x)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def op_alias(op_name: str) -> str:
    if op_name == "fc":
        return "linear"
    if op_name == "graph":
        return "gcn"
    return op_name

def make_gnn_op(op_name: str, in_dim: int, out_dim: int) -> nn.Module:
    name = op_alias(op_name)
    if name == "gat":
        return GATConv(in_dim, out_dim, heads=2, concat=False)
    if name == "gcn":
        return GCNConv(in_dim, out_dim)
    if name == "gin":
        return GINConv(nn.Linear(in_dim, out_dim))
    if name == "cheb":
        return ChebConv(in_dim, out_dim, K=2)
    if name == "sage":
        return SAGEConv(in_dim, out_dim)
    if name == "arma":
        return ARMAConv(in_dim, out_dim)
    if name in {"linear", "fc"}:
        return LinearConv(in_dim, out_dim)
    if name == "skip":
        return SkipConv(in_dim, out_dim)
    raise ValueError(f"Unsupported op: {op_name}")

class NASBenchGraphModel(nn.Module):
    def __init__(self, link: list[int], ops: list[str], input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.link = list(link)
        self.ops = list(ops)
        self.dropout = float(dropout)
        self.node_ops = nn.ModuleList()
        for parent, op_name in zip(self.link, self.ops):
            in_dim = input_dim if parent == 0 else hidden_dim
            self.node_ops.append(make_gnn_op(op_name, in_dim, hidden_dim))
        parent_set = {parent for parent in self.link if parent >= 1}
        self.leaf_nodes = [idx + 1 for idx in range(len(self.link)) if idx + 1 not in parent_set]
        self.classifier = nn.Linear(hidden_dim * len(self.leaf_nodes), num_classes)

    def forward(self, data: Data, return_logits: bool = False) -> Tensor:
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        states = [x]
        for parent, op in zip(self.link, self.node_ops):
            states.append(op(states[parent], data.edge_index))
        logits = self.classifier(torch.cat([states[idx] for idx in self.leaf_nodes], dim=-1))
        return logits if return_logits else F.log_softmax(logits, dim=-1)

class TriggerGenerator(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, link: list[int], ops: list[str], trigger_size: int, device: torch.device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.link = list(link)
        self.ops = list(ops)
        self.trigger_size = int(trigger_size)
        self.device = device
        self.node_ops = nn.ModuleList()
        for parent, op_name in zip(self.link, self.ops):
            in_dim = input_dim if parent == 0 else hidden_dim
            self.node_ops.append(make_gnn_op(op_name, in_dim, hidden_dim))
        parent_set = {parent for parent in self.link if parent >= 1}
        self.leaf_nodes = [idx + 1 for idx in range(len(self.link)) if idx + 1 not in parent_set]
        head_in_dim = hidden_dim * len(self.leaf_nodes)
        self.feat = nn.Linear(head_in_dim, trigger_size * input_dim)
        self.edge = nn.Linear(head_in_dim, trigger_size * (trigger_size - 1) // 2)

    def forward(self, input_tensor: Tensor, thrd: float) -> tuple[Tensor, Tensor]:
        x = input_tensor.to(self.device)
        edge_index = torch.arange(x.size(0), device=x.device).repeat(2, 1)
        states = [x]
        for parent, op in zip(self.link, self.node_ops):
            states.append(op(states[parent], edge_index))
        h = torch.cat([states[idx] for idx in self.leaf_nodes], dim=-1)
        return self.feat(h), GradWhere.apply(self.edge(h), thrd, x.device)

class HomoLoss(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def forward(self, trigger_edge_index: Tensor, trigger_edge_weights: Tensor, x: Tensor) -> Tensor:
        trigger_edge_index = trigger_edge_index[:, trigger_edge_weights > 0.0]
        if trigger_edge_index.numel() == 0:
            return x.new_tensor(0.0)
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]], x[trigger_edge_index[1]])
        return torch.relu(self.threshold - edge_sims).mean()

class Backdoor:

    def __init__(self, args: SimpleNamespace, device: torch.device):
        self.args = args
        self.device = device
        self.trigger_index = self.get_trigger_index(args.trigger_size)
        self.trojan: Optional[nn.Module] = None

    def get_trigger_index(self, trigger_size: int) -> Tensor:
        edge_list = [[0, 0]]
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j, k])
        return torch.tensor(edge_list, device=self.device).long().t()

    def get_trojan_edge(self, start: int, idx_attach: Tensor, trigger_size: int) -> Tensor:
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start
            edge_list.append(edges)
            start += trigger_size
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        edge_index = torch.cat(edge_list, dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        return torch.stack([row, col])

    def inject_trigger(self, idx_attach: Tensor, features: Tensor, edge_index: Tensor, edge_weight: Tensor, device: torch.device):
        assert self.trojan is not None
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()
        trojan_feat, trojan_weights = self.trojan(features[idx_attach], self.args.thrd)
        trojan_weights = torch.cat([torch.ones([len(idx_attach), 1], dtype=torch.float, device=device), trojan_weights], dim=1).flatten()
        trojan_feat = trojan_feat.view([-1, features.shape[1]])
        trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.args.trigger_size).to(device)
        update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
        update_feat = torch.cat([features, trojan_feat])
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)
        return update_feat, update_edge_index, update_edge_weights

    def fit_with_shadow(
        self,
        features: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        labels: Tensor,
        idx_train: Tensor,
        idx_val: Tensor,
        idx_attach: list[int],
        idx_unlabeled: Tensor,
        target_class: int,
        shadow_model: nn.Module,
        train_iters: int = 200,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        debug: bool = False,
        fixed_trigger: Optional[nn.Module] = None,
    ) -> None:
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)
        self.trojan = fixed_trigger.to(self.device) if fixed_trigger is not None else None
        if self.trojan is None:
            raise ValueError("This benchmark runner requires a fixed TriggerGenerator instance")

        shadow_model = shadow_model.to(self.device)
        optimizer_shadow = torch.optim.Adam(shadow_model.parameters(), lr=lr, weight_decay=weight_decay)
        train_data = Data(x=features, edge_index=edge_index)
        for ep in range(train_iters):
            shadow_model.train()
            optimizer_shadow.zero_grad()
            loss = F.nll_loss(shadow_model(train_data)[idx_train], labels[idx_train])
            loss.backward()
            optimizer_shadow.step()
            if debug and ep % 20 == 0:
                print(f"[Shadow] epoch {ep}, loss {float(loss):.4f}")

        for p in shadow_model.parameters():
            p.requires_grad = False
        shadow_model.eval()
        optimizer_trigger = torch.optim.Adam(self.trojan.parameters(), lr=self.args.retrain_lr, weight_decay=self.args.retrain_wd)
        homo_loss = HomoLoss(self.args.homo_boost_thrd)
        best_state = deepcopy(self.trojan.state_dict())
        best_loss = float("inf")
        for ep in range(self.args.trojan_epochs):
            self.trojan.train()
            optimizer_trigger.zero_grad()
            idx_outter = idx_val
            trojan_feat, trojan_weights = self.trojan(features[idx_outter], self.args.thrd)
            trojan_weights = torch.cat([torch.ones([len(idx_outter), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1).flatten()
            trojan_feat = trojan_feat.view([-1, features.shape[1]])
            trojan_edge = self.get_trojan_edge(len(features), idx_outter, self.args.trigger_size).to(self.device)
            update_feat = torch.cat([features, trojan_feat])
            update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)
            output = shadow_model(Data(x=update_feat, edge_index=update_edge_index))
            labels_outter = labels.clone()
            labels_outter[torch.cat([idx_train, idx_outter])] = target_class
            loss_target = self.args.target_loss_weight * F.nll_loss(
                output[torch.cat([idx_train, idx_outter])],
                labels_outter[torch.cat([idx_train, idx_outter])],
            )
            loss = loss_target

            print("loss_target: ", loss_target.item())
            tt = (output.argmax(dim=1)[torch.cat([idx_train, idx_outter])]==0).float().mean().item()
            print("asr: ", tt)

            loss.backward()
            nn.utils.clip_grad_norm_(self.trojan.parameters(), max_norm=1.0)
            optimizer_trigger.step()
            if float(loss) < best_loss:
                best_loss = float(loss)
                best_state = deepcopy(self.trojan.state_dict())
            if debug and ep % 10 == 0:
                print(f"[Trigger] epoch {ep}, loss {float(loss):.5f}")
        self.trojan.load_state_dict(best_state)
        self.trojan.eval()

def load_nbgraph_arch_module():
    import architecture as nb_arch

    return nb_arch

def enumerate_target_architectures(nb_arch: Any, use_proteins: bool = False) -> list[dict[str, Any]]:
    op_list = nb_arch.gnn_list_proteins if use_proteins else nb_arch.gnn_list
    records, seen = [], set()
    for ops in itertools.product(op_list, repeat=4):
        if all(op == "skip" for op in ops):
            continue
        for link in nb_arch.link_list:
            arch = nb_arch.Arch(list(link), list(ops))
            if arch.check_isomorph():
                arch_id = int(nb_arch.Arch(list(link), list(ops)).valid_hash())
                if arch_id not in seen:
                    seen.add(arch_id)
                    records.append({"architecture_id": arch_id, "link": list(link), "ops": list(ops)})
    return sorted(records, key=lambda item: int(item["architecture_id"]))

def enumerate_trigger_generator_architectures(op_candidates: list[str]) -> list[dict[str, Any]]:
    records, generator_id = [], 0
    for link in ([0, 0], [0, 1]):
        for ops in itertools.product(op_candidates, repeat=2):
            if all(op == "skip" for op in ops):
                continue
            records.append({"generator_architecture_id": generator_id, "link": list(link), "ops": list(ops)})
            generator_id += 1
    return records

def architecture_description(record: dict[str, Any]) -> str:
    return json.dumps({"link": record["link"], "ops": record["ops"]}, sort_keys=True)

def build_target_model(record: dict[str, Any], data: Data, hidden_dim: int, dropout: float) -> NASBenchGraphModel:
    return NASBenchGraphModel(record["link"], record["ops"], data.num_node_features, hidden_dim, int(data.y.max().item() + 1), dropout)

def build_trigger_generator(record: dict[str, Any], data: Data, hidden_dim: int, cfg: TriggerTrainConfig, device: torch.device) -> TriggerGenerator:
    return TriggerGenerator(data.num_node_features, hidden_dim, record["link"], record["ops"], cfg.trigger_size, device)

def load_backdoor_dataset(dataset: str, data_root: Path, device: torch.device, cfg: TriggerTrainConfig) -> Data:
    old_cwd = os.getcwd()
    try:
        os.chdir(EXAMPLES_DIR)
        data = GraphDataLoader(
            device=device,
            dataset_name=dataset,
            root=str(data_root),
            trigger_size=cfg.trigger_size,
            vs_size=40,
            target_class=cfg.target_class,
            split=True,
        ).load_data()
    finally:
        os.chdir(old_cwd)
    data.edge_index = to_undirected(data.edge_index)
    return data.to(device)

def build_split_context(data: Data, device: torch.device) -> SplitContext:
    data.edge_index = to_undirected(data.edge_index)
    train_edge_index, _, edge_mask = subgraph(torch.bitwise_not(data.test_mask), data.edge_index, relabel_nodes=False)
    mask_edge_index = data.edge_index[:, torch.bitwise_not(edge_mask)]
    idx_train = data.train_mask.nonzero(as_tuple=False).flatten()
    idx_val = data.val_mask.nonzero(as_tuple=False).flatten()
    idx_test = data.test_mask.nonzero(as_tuple=False).flatten()
    half = int(len(idx_test) / 2)
    induct_edge_index = torch.cat([train_edge_index, mask_edge_index], dim=1).to(device)
    return SplitContext(
        train_edge_index=train_edge_index.to(device),
        idx_train=idx_train.to(device),
        idx_val=idx_val.to(device),
        idx_clean_test=idx_test[:half].to(device),
        idx_atk=idx_test[half:].to(device),
        unlabeled_idx=(torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)).nonzero().flatten().to(device),
        induct_edge_index=induct_edge_index,
        induct_edge_weights=torch.ones([induct_edge_index.shape[1]], dtype=torch.float, device=device),
    )

def train_target_model(model: nn.Module, data: Data, split_ctx: SplitContext, cfg: TargetTrainConfig, device: torch.device) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_data = Data(x=data.x, edge_index=split_ctx.train_edge_index, y=data.y)
    for _ in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(train_data)[split_ctx.idx_train], data.y[split_ctx.idx_train])
        loss.backward()
        optimizer.step()
    model.eval()
    return model

def make_backdoor_args(target_cfg: TargetTrainConfig, trigger_cfg: TriggerTrainConfig, eval_cfg: EvalConfig, trojan_epochs: Optional[int] = None) -> SimpleNamespace:
    return SimpleNamespace(
        trigger_size=trigger_cfg.trigger_size,
        target_class=trigger_cfg.target_class,
        thrd=trigger_cfg.thrd,
        retrain_epochs=target_cfg.epochs,
        train_lr=target_cfg.lr,
        weight_decay=target_cfg.weight_decay,
        retrain_lr=trigger_cfg.lr,
        retrain_wd=trigger_cfg.weight_decay,
        trojan_epochs=trigger_cfg.epochs if trojan_epochs is None else int(trojan_epochs),
        target_loss_weight=trigger_cfg.target_loss_weight,
        homo_loss_weight=trigger_cfg.homo_loss_weight,
        homo_boost_thrd=trigger_cfg.homo_boost_thrd,
        debug=trigger_cfg.debug,
        defense_mode=eval_cfg.defense_mode,
        prune_thr=eval_cfg.prune_thr,
    )

def train_trigger_generator(
    target_model: nn.Module,
    generator: TriggerGenerator,
    data: Data,
    split_ctx: SplitContext,
    target_cfg: TargetTrainConfig,
    trigger_cfg: TriggerTrainConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
) -> Backdoor:
    args = make_backdoor_args(target_cfg, trigger_cfg, eval_cfg)
    backdoor = Backdoor(args, device)
    backdoor.fit_with_shadow(
        data.x,
        split_ctx.train_edge_index,
        None,
        data.y,
        split_ctx.idx_train,
        split_ctx.idx_val,
        [],
        split_ctx.unlabeled_idx,
        trigger_cfg.target_class,
        shadow_model=target_model,
        train_iters=0,
        lr=target_cfg.lr,
        weight_decay=target_cfg.weight_decay,
        debug=trigger_cfg.debug,
        fixed_trigger=generator,
    )
    return backdoor

@torch.no_grad()
def evaluate_target(model: nn.Module, data: Data, split_ctx: SplitContext, backdoor: Backdoor, args: SimpleNamespace, device: torch.device) -> dict[str, float]:
    model.eval()
    out_clean = model(Data(x=data.x, edge_index=split_ctx.induct_edge_index, y=data.y))
    acc = (out_clean[split_ctx.idx_clean_test].argmax(1) == data.y[split_ctx.idx_clean_test]).float().mean().item()
    induct_x, edge_index2, edge_weights2 = backdoor.inject_trigger(split_ctx.idx_atk, data.x, split_ctx.induct_edge_index, split_ctx.induct_edge_weights, device)
    if args.defense_mode == "prune":
        edge_index2, edge_weights2 = prune_unrelated_edge(args, edge_index2, edge_weights2, induct_x, device)
    elif args.defense_mode == "isolate":
        edge_index2, edge_weights2, _ = prune_unrelated_edge_isolated(args, edge_index2, edge_weights2, induct_x, device)
    out_poison = model(Data(x=induct_x, edge_index=edge_index2))
    asr = (out_poison.argmax(dim=1)[split_ctx.idx_atk] == args.target_class).float().mean().item()
    return {"ACC": float(acc), "ASR": float(asr)}

def make_eval_backdoor(generator: TriggerGenerator, trigger_cfg: TriggerTrainConfig, eval_cfg: EvalConfig, device: torch.device) -> tuple[Backdoor, SimpleNamespace]:
    args = make_backdoor_args(TargetTrainConfig(), trigger_cfg, eval_cfg, trojan_epochs=0)
    backdoor = Backdoor(args, device)
    backdoor.trojan = generator.to(device)
    return backdoor, args

def sample_proxy_nodes(data: Data, split: str, size: int) -> Tensor:
    idx = getattr(data, f"{split}_mask").nonzero(as_tuple=False).flatten()
    if idx.numel() <= size:
        return idx
    return idx[torch.randperm(idx.numel(), device=idx.device)[:size]]

def forward_logits(model: nn.Module, data: Data) -> Tensor:
    try:
        return model(data, return_logits=True)
    except TypeError:
        return model(data)

def compute_jacobian_logits_wrt_params(model: nn.Module, data: Data, node_indices: Tensor) -> Tensor:
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    logits = forward_logits(model, data)[node_indices]
    n, c = logits.shape
    sizes = [p.numel() for p in params]
    rows = []
    for row in range(n):
        for cls in range(c):
            grads = torch.autograd.grad(logits[row, cls], params, retain_graph=True, create_graph=False, allow_unused=True)
            parts = [torch.zeros(size, device=logits.device, dtype=p.dtype) if g is None else g.reshape(-1) for g, p, size in zip(grads, params, sizes)]
            rows.append(torch.cat(parts).detach().cpu())
    return torch.stack(rows, dim=0)

def compute_full_ntk(model: nn.Module, data: Data, node_indices: Tensor) -> Tensor:
    j_full = compute_jacobian_logits_wrt_params(model, data, node_indices)
    return j_full @ j_full.t()

def compute_sample_ntk(model: nn.Module, data: Data, node_indices: Tensor) -> Tensor:
    j_full = compute_jacobian_logits_wrt_params(model, data, node_indices)
    num_nodes = int(node_indices.numel())
    num_classes = j_full.size(0) // num_nodes
    return torch.einsum("icp,jcp->ij", j_full.view(num_nodes, num_classes, -1), j_full.view(num_nodes, num_classes, -1))

def frobenius_cosine(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return float(torch.dot(a, b).div(a.norm() * b.norm() + eps).item())

def linear_cka(a: Tensor, b: Tensor, eps: float = 1e-12) -> float:
    a, b = a.float() - a.float().mean(dim=0, keepdim=True), b.float() - b.float().mean(dim=0, keepdim=True)
    return float(((a.t() @ b).pow(2).sum() / (((a.t() @ a).pow(2).sum().sqrt() * (b.t() @ b).pow(2).sum().sqrt()) + eps)).item())

def matrix_similarity(a: Tensor, b: Tensor, method: str = "fro_cos") -> float:
    return frobenius_cosine(a, b) if method == "fro_cos" else linear_cka(a, b)

def pairwise_similarity(mats: list[Tensor], method: str = "fro_cos") -> float:
    if len(mats) < 2:
        return float("nan")
    return float(np.mean([matrix_similarity(mats[i], mats[j], method) for i in range(len(mats)) for j in range(i + 1, len(mats))]))

def compute_ntk_metrics(models: dict[int, nn.Module], data: Data, node_indices: Tensor, num_classes: int, cfg: NTKConfig, seeds: tuple[int, ...]) -> dict[str, float]:
    ntk_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
    sample_ntks, full_ntks = [], []
    for seed in seeds:
        for p in models[seed].parameters():
            p.requires_grad_(True)
        sample_ntks.append(compute_sample_ntk(models[seed], ntk_data, node_indices))
        full_ntks.append(compute_full_ntk(models[seed], ntk_data, node_indices))
    y_onehot = F.one_hot(data.y[node_indices].detach().cpu(), num_classes=num_classes).float()
    y_vec = y_onehot.reshape(-1).float()
    return {
        "sample_NTK_similarity": pairwise_similarity(sample_ntks, cfg.similarity_method),
        "full_NTK_similarity": pairwise_similarity(full_ntks, cfg.similarity_method),
        "sample_NTK_label_projection_similarity": pairwise_similarity([k.float() @ y_onehot for k in sample_ntks]),
        "full_NTK_label_projection_similarity": pairwise_similarity([k.float() @ y_vec for k in full_ntks]),
    }

def prepare_target_architecture(record: dict[str, Any], data: Data, split_ctx: SplitContext, target_cfg: TargetTrainConfig, ntk_cfg: NTKConfig, hidden_dim: int, dropout: float, device: torch.device, seeds: tuple[int, ...]):
    trained, init_models, clean_accs = {}, {}, {}
    for seed in seeds:
        set_seed(seed)
        model = build_target_model(record, data, hidden_dim, dropout).to(device)
        if ntk_cfg.compute_at == "init":
            init_models[seed] = deepcopy(model).to(device).eval()
        trained[seed] = train_target_model(model, data, split_ctx, target_cfg, device)
        clean_accs[seed] = evaluate_target_clean(trained[seed], data, split_ctx)
    ntk_models = init_models if ntk_cfg.compute_at == "init" else trained
    set_seed(12345)
    proxy_nodes = sample_proxy_nodes(data, ntk_cfg.proxy_split, ntk_cfg.proxy_batch_size)
    ntk_metrics = compute_ntk_metrics(ntk_models, data, proxy_nodes, int(data.y.max().item() + 1), ntk_cfg, seeds)
    return trained, clean_accs, ntk_metrics

@torch.no_grad()
def evaluate_target_clean(model: nn.Module, data: Data, split_ctx: SplitContext) -> float:
    model.eval()
    out = model(Data(x=data.x, edge_index=split_ctx.induct_edge_index, y=data.y))
    return float((out[split_ctx.idx_clean_test].argmax(1) == data.y[split_ctx.idx_clean_test]).float().mean().item())

def run_one_generator_combo(target_record: dict[str, Any], generator_record: dict[str, Any], trained_models: dict[int, nn.Module], clean_accs: dict[int, float], ntk_metrics: dict[str, float], data: Data, split_ctx: SplitContext, target_cfg: TargetTrainConfig, trigger_cfg: TriggerTrainConfig, eval_cfg: EvalConfig, hidden_dim: int, device: torch.device, seeds: tuple[int, ...]) -> dict[str, Any]:
    primary_seed = seeds[0]
    generator = build_trigger_generator(generator_record, data, hidden_dim, trigger_cfg, device).to(device)
    backdoor = train_trigger_generator(trained_models[primary_seed], generator, data, split_ctx, target_cfg, trigger_cfg, eval_cfg, device)
    fixed_generator = deepcopy(backdoor.trojan).to(device).eval()
    eval_backdoor, eval_args = make_eval_backdoor(fixed_generator, trigger_cfg, eval_cfg, device)
    per_seed = {seed: evaluate_target(trained_models[seed], data, split_ctx, eval_backdoor, eval_args, device) for seed in seeds}
    acc_values = np.array([per_seed[s]["ACC"] for s in seeds], dtype=float)
    asr_values = np.array([per_seed[s]["ASR"] for s in seeds], dtype=float)
    row: dict[str, Any] = {
        "architecture_id": f"{target_record['architecture_id']}__gen_{generator_record['generator_architecture_id']}",
        "target_architecture_id": int(target_record["architecture_id"]),
        "target_architecture_description": architecture_description(target_record),
        "generator_architecture_id": int(generator_record["generator_architecture_id"]),
        "generator_architecture_description": architecture_description(generator_record),
        "mean_ACC": float(np.nanmean(acc_values)),
        "std_ACC": float(np.nanstd(acc_values)),
        "mean_ASR": float(np.nanmean(asr_values)),
        "std_ASR": float(np.nanstd(asr_values)),
        **ntk_metrics,
    }
    for seed in seeds:
        row[f"ACC_seed_{seed}"] = float(per_seed[seed]["ACC"])
        row[f"ASR_seed_{seed}"] = float(per_seed[seed]["ASR"])
        row[f"clean_train_ACC_seed_{seed}"] = float(clean_accs[seed])
    return row

def data_to_cpu(data: Data) -> Data:
    data_cpu = Data(x=data.x.detach().cpu(), edge_index=data.edge_index.detach().cpu(), y=data.y.detach().cpu())
    for name in ("train_mask", "val_mask", "test_mask", "clean_idx", "trigger_idx", "target_class"):
        if hasattr(data, name):
            value = getattr(data, name)
            setattr(data_cpu, name, value.detach().cpu() if torch.is_tensor(value) else value)
    return data_cpu

def split_context_to_device(split_ctx: SplitContext, device: torch.device) -> SplitContext:
    return SplitContext(
        train_edge_index=split_ctx.train_edge_index.to(device),
        idx_train=split_ctx.idx_train.to(device),
        idx_val=split_ctx.idx_val.to(device),
        idx_clean_test=split_ctx.idx_clean_test.to(device),
        idx_atk=split_ctx.idx_atk.to(device),
        unlabeled_idx=split_ctx.unlabeled_idx.to(device),
        induct_edge_index=split_ctx.induct_edge_index.to(device),
        induct_edge_weights=split_ctx.induct_edge_weights.to(device),
    )

def split_context_to_cpu(split_ctx: SplitContext) -> SplitContext:
    return split_context_to_device(split_ctx, torch.device("cpu"))

def model_state_dicts_to_cpu(models: dict[int, nn.Module]) -> dict[int, dict[str, Tensor]]:
    return {
        seed: {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        for seed, model in models.items()
    }

def run_one_generator_combo_process(payload: dict[str, Any]) -> dict[str, Any]:
    device_text = payload["device"]
    device = torch.device(device_text if torch.cuda.is_available() or not str(device_text).startswith("cuda") else "cpu")
    data = payload["data"].to(device)
    split_ctx = split_context_to_device(payload["split_ctx"], device)
    target_record = payload["target_record"]
    seeds = payload["seeds"]
    trained_models = {}
    for seed in seeds:
        model = build_target_model(target_record, data, payload["hidden_dim"], payload["dropout"]).to(device)
        model.load_state_dict(payload["target_state_dicts"][seed])
        model.eval()
        trained_models[seed] = model
    return run_one_generator_combo(
        target_record,
        payload["generator_record"],
        trained_models,
        payload["clean_accs"],
        payload["ntk_metrics"],
        data,
        split_ctx,
        payload["target_cfg"],
        payload["trigger_cfg"],
        payload["eval_cfg"],
        payload["hidden_dim"],
        device,
        seeds,
    )

def write_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def write_jsonl_row(jsonl_path: Path, row: dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_completed_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8") as f:
        return {row["architecture_id"] for row in csv.DictReader(f) if row.get("architecture_id")}

def parse_seeds(seed_text: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in seed_text.split(",") if item.strip())
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for NTK similarity")
    return seeds

def dataset_output_name(dataset: str) -> str:
    return dataset.strip().replace("/", "_").replace("\\", "_").lower()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full NAS-Bench-Graph backdoor architecture benchmark.")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--data-root", type=str, default=str(EXAMPLES_DIR / "data"))
    parser.add_argument("--output-dir", type=str, default="./results/nbgraph_backdoor_benchmark")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--parallel-combos", type=int, default=1, help="Number of target/generator combinations to run concurrently per target architecture")
    parser.add_argument("--use-proteins-space", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--train-lr", type=float, default=0.01)
    parser.add_argument("--train-wd", type=float, default=5e-4)
    parser.add_argument("--trojan-epochs", type=int, default=200)
    parser.add_argument("--trojan-lr", type=float, default=0.01)
    parser.add_argument("--trojan-wd", type=float, default=5e-4)
    parser.add_argument("--trigger-size", type=int, default=3)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--homo-loss-weight", type=float, default=100.0)
    parser.add_argument("--homo-boost-thrd", type=float, default=0.5)
    parser.add_argument("--thrd", type=float, default=0.5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--defense-mode", type=str, default="prune", choices=["prune", "isolate", "none"])
    parser.add_argument("--prune-thr", type=float, default=0.1)
    parser.add_argument("--ntk-compute-at", type=str, default="init", choices=["init", "trained"])
    parser.add_argument("--ntk-proxy-split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--ntk-proxy-batch-size", type=int, default=16)
    parser.add_argument("--ntk-similarity", type=str, default="fro_cos", choices=["fro_cos", "cka"])
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    output_dir = Path(args.output_dir) / dataset_output_name(args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    jsonl_path = output_dir / "results.jsonl"
    target_cfg = TargetTrainConfig(args.train_epochs, args.train_lr, args.train_wd)
    trigger_cfg = TriggerTrainConfig(args.trojan_epochs, args.trojan_lr, args.trojan_wd, args.trigger_size, args.target_class, args.target_loss_weight, args.homo_loss_weight, args.homo_boost_thrd, args.thrd, args.debug)
    eval_cfg = EvalConfig(args.defense_mode, args.prune_thr)
    ntk_cfg = NTKConfig(args.ntk_compute_at, args.ntk_proxy_split, args.ntk_proxy_batch_size, args.ntk_similarity)
    nb_arch = load_nbgraph_arch_module()
    target_records = enumerate_target_architectures(nb_arch, args.use_proteins_space)
    op_candidates = list(nb_arch.gnn_list_proteins if args.use_proteins_space else nb_arch.gnn_list)
    generator_records = enumerate_trigger_generator_architectures(op_candidates)
    set_seed(seeds[0])
    data = load_backdoor_dataset(args.dataset, Path(args.data_root), device, trigger_cfg)
    split_ctx = build_split_context(data, device)
    config_dump = {
        "args": vars(args),
        "target_train_config": asdict(target_cfg),
        "trigger_train_config": asdict(trigger_cfg),
        "eval_config": asdict(eval_cfg),
        "ntk_config": asdict(ntk_cfg),
        "seeds": seeds,
        "num_target_architectures": len(target_records),
        "num_generator_architectures": len(generator_records),
        "op_alias": {"graph": "gcn", "fc": "linear"},
        "data_root": str(Path(args.data_root).resolve()),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)
    completed = load_completed_ids(csv_path) if args.resume else set()
    total_jobs = len(target_records) * len(generator_records)
    job_pos = 0
    for target_pos, target_record in enumerate(target_records, start=1):
        target_id = int(target_record["architecture_id"])
        combo_ids = {f"{target_id}__gen_{g['generator_architecture_id']}" for g in generator_records}
        if combo_ids.issubset(completed):
            job_pos += len(generator_records)
            continue
        print(f"[target {target_pos}/{len(target_records)}] id={target_id} link={target_record['link']} ops={target_record['ops']}")
        trained_models, clean_accs, ntk_metrics = prepare_target_architecture(target_record, data, split_ctx, target_cfg, ntk_cfg, args.hidden_dim, args.dropout, device, seeds)
        pending_jobs = []
        for generator_record in generator_records:
            generator_id = int(generator_record["generator_architecture_id"])
            combo_id = f"{target_id}__gen_{generator_id}"
            job_pos += 1
            if combo_id in completed:
                continue
            pending_jobs.append((job_pos, generator_record))

        if not pending_jobs:
            continue

        max_workers = max(1, int(args.parallel_combos))
        data_cpu = data_to_cpu(data)
        split_ctx_cpu = split_context_to_cpu(split_ctx)
        target_state_dicts = model_state_dicts_to_cpu(trained_models)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
            future_to_job = {}
            for pos, generator_record in pending_jobs:
                generator_id = int(generator_record["generator_architecture_id"])
                print(f"[submit {pos}/{total_jobs}] target={target_id} generator={generator_id} {generator_record['link']}/{generator_record['ops']}")
                future = executor.submit(
                    run_one_generator_combo_process,
                    {
                        "target_record": target_record,
                        "generator_record": generator_record,
                        "target_state_dicts": target_state_dicts,
                        "clean_accs": clean_accs,
                        "ntk_metrics": ntk_metrics,
                        "data": data_cpu,
                        "split_ctx": split_ctx_cpu,
                        "target_cfg": target_cfg,
                        "trigger_cfg": trigger_cfg,
                        "eval_cfg": eval_cfg,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "device": str(device),
                        "seeds": seeds,
                    },
                )
                future_to_job[future] = (pos, generator_id)

            for future in concurrent.futures.as_completed(future_to_job):
                pos, generator_id = future_to_job[future]
                row = future.result()
                write_csv_row(csv_path, row)
                write_jsonl_row(jsonl_path, row)
                completed.add(row["architecture_id"])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[done {pos}/{total_jobs}] target={target_id} generator={generator_id}")

if __name__ == "__main__":
    main()
