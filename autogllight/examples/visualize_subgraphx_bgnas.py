#!/usr/bin/env python3

import argparse
import json
import os
import signal
import sys
import warnings

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, k_hop_subgraph
from torch_geometric.utils import subgraph as tg_subgraph

from autogllight.utils import set_seed
from utils.data_atk import GraphDataLoader
from autogllight.nas.space.graph_nas import GraphBenchmarkingSpace
from models.UGBA import Backdoor

from dig.xgraph.method import SubgraphX
from dig.xgraph.method.subgraphx import find_closest_node_result

def get_args():
    p = argparse.ArgumentParser(
        description="DIG SubgraphX node-level clean-vs-triggered local "
                    "explanation visualisation for a BGNAS backdoor model."
    )
    p.add_argument("--dataset", type=str, default="Cora",
                   choices=["Citeseer", "Cora", "Pubmed", "Photo",
                            "Computers", "Flickr", "Actor", "chameleon"])
    p.add_argument("--seed", type=int, default=1047,
                   help="Seed identifying the saved model AND reproducing the "
                        "data split (saved_models1/{dataset}/seed_{seed}/).")
    p.add_argument("--target_class", type=int, default=0,
                   help="Backdoor target class (used for data split + "
                        "'--explain_class backdoor').")
    p.add_argument("--models_root", type=str, default="saved_models1",
                   help="Root dir holding {dataset}/seed_{seed}/"
                        "{alpha_star.json, fixed_model_state_dict.pt, "
                        "fixed_trigger_obj.pt}.")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--layer_number", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--vs_size", type=int, default=40)
    p.add_argument("--trigger_size", type=int, default=3)
    p.add_argument("--thrd", type=float, default=0.5)
    p.add_argument("--num_hops", type=int, default=2,
                   help="k-hop computation graph / ego-graph radius.")
    p.add_argument("--max_nodes", type=int, default=6,
                   help="Max #nodes of the SubgraphX important sub-graph.")
    p.add_argument("--rollout", type=int, default=10,
                   help="SubgraphX MCTS rollouts (higher = slower/better).")
    p.add_argument("--min_atoms", type=int, default=3)
    p.add_argument("--expand_atoms", type=int, default=10)
    p.add_argument("--reward_method", type=str, default="mc_l_shapley")
    p.add_argument("--explain_class", type=str, default="pred",
                   help="'pred' (model prediction on each graph), 'backdoor' "
                        "(the --target_class), or an integer class id.")
    p.add_argument("--max_test_nodes", type=int, default=60,
                   help="Explain at most this many test nodes (0 = all).")
    p.add_argument("--node_list", type=str, default=None,
                   help="Comma-separated GLOBAL node ids to explain instead of "
                        "iterating the test set.")
    p.add_argument("--max_plot", type=int, default=5,
                   help="Max number of matched nodes to render.")
    p.add_argument("--plot_all_tested", action="store_true", default=False,
                   help="Also render nodes that do NOT satisfy the trigger "
                        "criterion (for debugging).")
    p.add_argument("--save_dir", type=str, default="results/subgraphx_vis")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed_vis", type=int, default=0,
                   help="Seed for the spring layout (reproducible figures).")
    return p.parse_args()

def resolve_device(spec):
    if isinstance(spec, torch.device):
        return spec
    if str(spec).isdigit():
        spec = f"cuda:{spec}"
    if "cuda" in str(spec) and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(spec)

def build_dataset(dataset, seed, device, trigger_size, vs_size, target_class,
                  data_root):

    set_seed(int(seed))
    loader = GraphDataLoader(
        device=device,
        dataset_name=dataset,
        root=data_root,
        trigger_size=trigger_size,
        vs_size=vs_size,
        target_class=target_class,
        split=True,
    )
    return loader.load_data()

def build_induct_edge_index(data):

    data.edge_index = to_undirected(data.edge_index)
    train_edge_index, _, edge_mask = tg_subgraph(
        torch.bitwise_not(data.test_mask), data.edge_index, relabel_nodes=False,
        return_edge_mask=True,
    )
    mask_edge_index = data.edge_index[:, torch.bitwise_not(edge_mask)]
    induct_edge_index = torch.cat([train_edge_index, mask_edge_index], dim=1)
    return induct_edge_index

def load_bgnas_model(dataset, seed, data, device, models_root,
                     hidden_dim=64, layer_number=4, dropout=0.5):

    save_dir = os.path.join(_THIS_DIR, models_root, dataset, f"seed_{seed}")
    arch_path = os.path.join(save_dir, "alpha_star.json")
    sd_path = os.path.join(save_dir, "fixed_model_state_dict.pt")
    if not os.path.exists(arch_path):
        raise FileNotFoundError(f"architecture not found: {arch_path}")
    if not os.path.exists(sd_path):
        raise FileNotFoundError(f"state_dict not found: {sd_path}")

    with open(arch_path) as f:
        selection = json.load(f)

    num_classes = int(data.y.max().item()) + 1
    space = GraphBenchmarkingSpace(
        hidden_dim=hidden_dim,
        layer_number=layer_number,
        dropout=dropout,
        input_dim=data.num_node_features,
        output_dim=num_classes,
    )
    space.instantiate()
    box = space.parse_model(selection)
    model = box._model
    state_dict = torch.load(sd_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, selection, num_classes

def load_trigger_backdoor(dataset, seed, device, models_root, trigger_size,
                          target_class, thrd):

    save_dir = os.path.join(_THIS_DIR, models_root, dataset, f"seed_{seed}")
    trigger_path = os.path.join(save_dir, "fixed_trigger_obj.pt")
    if not os.path.exists(trigger_path):
        raise FileNotFoundError(f"trigger not found: {trigger_path}")
    trojan = torch.load(trigger_path, map_location=device).to(device)
    trojan.eval()

    class _Args:
        pass

    a = _Args()
    a.trigger_size = int(trigger_size)
    a.thrd = float(thrd)
    a.target_class = int(target_class)

    backdoor = Backdoor(a, device)
    backdoor.trojan = trojan
    return backdoor

def get_2hop_ego_graph(data, node_idx, relabel_nodes=True, num_hops=2,
                       num_nodes=None):

    edge_index = data.edge_index if hasattr(data, "edge_index") else data
    if num_nodes is None and hasattr(data, "num_nodes"):
        num_nodes = data.num_nodes
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=int(node_idx),
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=relabel_nodes,
        num_nodes=num_nodes,
        flow="source_to_target",
    )
    return subset, sub_edge_index, mapping, edge_mask

class DIGModelWrapper(nn.Module):

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x=None, edge_index=None, data=None, **kwargs):
        if data is None:
            data = Data(x=x, edge_index=edge_index)
        return self.base_model(data)

def _induced_edges(important_nodes, edge_index):

    imp = set(int(n) for n in important_nodes)
    ei = edge_index.detach().cpu().numpy()
    edges = set()
    for u, v in zip(ei[0], ei[1]):
        u, v = int(u), int(v)
        if u in imp and v in imp:
            edges.add((u, v))
    return edges

def run_subgraphx_node_explanation(model, data, node_idx, num_classes, device,
                                   num_hops=2, target_class=None,
                                   max_nodes=6, rollout=10, min_atoms=3,
                                   expand_atoms=10, reward_method="mc_l_shapley"):

    edge_index = data.edge_index if hasattr(data, "edge_index") else data
    x = data.x

    model.eval()
    if target_class is None:
        with torch.no_grad():
            logits = model(Data(x=x, edge_index=edge_index))
            target_class = int(logits[int(node_idx)].argmax().item())

    wrapper = DIGModelWrapper(model).to(device).eval()
    explainer = SubgraphX(
        wrapper, num_classes, device,
        num_hops=num_hops,
        explain_graph=False,
        rollout=rollout,
        min_atoms=min_atoms,
        expand_atoms=expand_atoms,
        reward_method=reward_method,
        vis=False,
    )

    results, related_pred = explainer.explain(
        x, edge_index, label=int(target_class),
        max_nodes=max_nodes, node_idx=int(node_idx),
    )

    subset = explainer.mcts_state_map.subset
    new_node_idx = int(explainer.mcts_state_map.new_node_idx)

    node_results = explainer.read_from_MCTSInfo_list(results)
    best = find_closest_node_result(node_results, max_nodes=max_nodes)
    coalition_local = list(best.coalition)

    important_nodes = set(int(subset[c]) for c in coalition_local)
    important_nodes.add(int(node_idx))
    important_edges = _induced_edges(important_nodes, edge_index)

    raw_result = {
        "target_class": int(target_class),
        "subset_global": [int(s) for s in subset.tolist()],
        "new_node_idx_local": new_node_idx,
        "coalition_local": coalition_local,
        "related_pred": related_pred,
    }
    return important_nodes, important_edges, raw_result

def inject_single_trigger(backdoor, node_idx, x, edge_index, device,
                          trigger_size):

    n_before = x.shape[0]
    edge_weight = torch.ones([edge_index.shape[1]], device=device)
    idx_attach = torch.tensor([int(node_idx)], device=device)

    with torch.no_grad():
        poison_x, poison_edge_index, _ = backdoor.inject_trigger(
            idx_attach, x, edge_index, edge_weight, device
        )
    trojan_edge = backdoor.get_trojan_edge(
        n_before, idx_attach, int(trigger_size)
    ).to(device)

    te = trojan_edge.detach().cpu().numpy()
    trigger_nodes = sorted(
        {int(v) for v in te.reshape(-1) if int(v) >= n_before}
    )
    trigger_edges, connection_edges = set(), set()
    for u, v in zip(te[0], te[1]):
        u, v = int(u), int(v)
        trigger_edges.add((u, v))
        if u < n_before or v < n_before:
            connection_edges.add((u, v))
    return (poison_x, poison_edge_index, trigger_nodes, trigger_edges,
            connection_edges)

def _key(u, v):
    return (u, v) if u <= v else (v, u)

def _edge_key_set(edges):
    return {_key(int(u), int(v)) for (u, v) in edges}

def _draw_single(ax, title, ego_edges, target_node, neighbour_nodes,
                 trigger_nodes, trigger_edge_keys, connection_edge_keys,
                 important_nodes, important_edge_keys, pos):

    g = nx.Graph()
    all_nodes = set(neighbour_nodes) | {target_node} | set(trigger_nodes)
    g.add_nodes_from(all_nodes)
    for (u, v) in ego_edges:
        g.add_edge(int(u), int(v))

    trigger_set = set(int(t) for t in trigger_nodes)
    important_set = set(int(n) for n in important_nodes)

    node_face, node_size, node_edgecol, node_edgew = [], [], [], []
    for n in g.nodes():
        if n == target_node:
            node_face.append("#FFD400")
            node_size.append(900)
        elif n in trigger_set:
            node_face.append("#E8482B")
            node_size.append(430)
        else:
            node_face.append("#8FB8DE")
            node_size.append(300)
        if n in important_set:
            node_edgecol.append("#1B7F1B")
            node_edgew.append(3.0)
        else:
            node_edgecol.append("#333333")
            node_edgew.append(0.8)

    normal_e, important_e, trigger_e, trigger_imp_e = [], [], [], []
    for (u, v) in g.edges():
        k = _key(u, v)
        is_trig = (k in trigger_edge_keys) or (k in connection_edge_keys)
        is_imp = k in important_edge_keys
        if is_trig and is_imp:
            trigger_imp_e.append((u, v))
        elif is_trig:
            trigger_e.append((u, v))
        elif is_imp:
            important_e.append((u, v))
        else:
            normal_e.append((u, v))

    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=normal_e,
                           edge_color="#CCCCCC", width=1.0)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=important_e,
                           edge_color="#000000", width=3.2)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=trigger_e,
                           edge_color="#E8482B", width=2.0, style="dashed")
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=trigger_imp_e,
                           edge_color="#E8482B", width=3.4, style="solid")

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_face,
                           node_size=node_size, edgecolors=node_edgecol,
                           linewidths=node_edgew)
    labels = {n: str(n) for n in g.nodes()
              if n == target_node or n in trigger_set}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=8,
                            font_color="black")

    ax.set_title(title, fontsize=11)
    ax.axis("off")

def visualize_clean_vs_triggered(node_idx, clean_info, trig_info, save_path,
                                  layout_seed=0):

    c_subset = clean_info["subset"]
    c_edges = clean_info["ego_edges"]
    c_important = clean_info["important_nodes"]
    c_important_ek = _edge_key_set(clean_info["important_edges"])
    c_neighbours = [int(n) for n in c_subset if int(n) != node_idx]

    base_g = nx.Graph()
    base_g.add_nodes_from(int(n) for n in c_subset)
    for (u, v) in c_edges:
        base_g.add_edge(int(u), int(v))
    if base_g.number_of_nodes() == 0:
        base_g.add_node(node_idx)
    pos_clean = nx.spring_layout(base_g, seed=layout_seed, k=0.9)

    t_subset = trig_info["subset"]
    t_edges = trig_info["ego_edges"]
    t_important = trig_info["important_nodes"]
    t_important_ek = _edge_key_set(trig_info["important_edges"])
    trigger_nodes = trig_info["trigger_nodes"]
    trigger_ek = _edge_key_set(trig_info["trigger_edges"])
    connection_ek = _edge_key_set(trig_info["connection_edges"])
    t_neighbours = [int(n) for n in t_subset
                    if int(n) != node_idx and int(n) not in set(trigger_nodes)]

    pos_trig = dict(pos_clean)
    tx, ty = pos_clean.get(node_idx, np.array([0.0, 0.0]))
    rng = np.random.RandomState(layout_seed + 1)
    for i, tn in enumerate(sorted(trigger_nodes)):
        ang = 2 * np.pi * i / max(1, len(trigger_nodes))
        pos_trig[tn] = np.array([tx + 0.45 * np.cos(ang) + 0.05 * rng.randn(),
                                 ty + 0.45 * np.sin(ang) + 0.05 * rng.randn()])
    trig_all = set(t_neighbours) | {node_idx} | set(trigger_nodes)
    for n in trig_all:
        if n not in pos_trig:
            pos_trig[n] = np.array([0.6 * rng.randn(), 0.6 * rng.randn()])

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    _draw_single(
        axes[0],
        f"CLEAN  |  node {node_idx}  (explain class {clean_info['target_class']})",
        c_edges, node_idx, c_neighbours, [], set(), set(),
        c_important, c_important_ek, pos_clean,
    )
    _draw_single(
        axes[1],
        f"TRIGGERED  |  node {node_idx}  (explain class {trig_info['target_class']})",
        t_edges, node_idx, t_neighbours, trigger_nodes, trigger_ek,
        connection_ek, t_important, t_important_ek, pos_trig,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFD400",
               markeredgecolor="#333333", markersize=15, label="target node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#8FB8DE",
               markeredgecolor="#333333", markersize=11, label="original neighbour"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E8482B",
               markeredgecolor="#333333", markersize=11, label="trigger node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#DDDDDD",
               markeredgecolor="#1B7F1B", markeredgewidth=3, markersize=13,
               label="SubgraphX important node"),
        Line2D([0], [0], color="#CCCCCC", lw=1.5, label="normal edge"),
        Line2D([0], [0], color="#000000", lw=3.2, label="SubgraphX important edge"),
        Line2D([0], [0], color="#E8482B", lw=2.0, ls="--", label="trigger edge"),
        Line2D([0], [0], color="#E8482B", lw=3.4, label="trigger & important edge"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"DIG SubgraphX 2-hop local explanation — node {node_idx}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _resolve_explain_class(explain_class, model, x, edge_index, node_idx,
                           backdoor_target):
    if explain_class == "pred":
        return None
    if explain_class == "backdoor":
        return int(backdoor_target)
    return int(explain_class)

def explain_clean_node(model, clean_x, induct_edge_index, node_idx, num_classes,
                       device, args):
    tclass = _resolve_explain_class(
        args.explain_class, model, clean_x, induct_edge_index, node_idx,
        args.target_class
    )
    clean_graph = Data(x=clean_x, edge_index=induct_edge_index)
    imp_nodes, imp_edges, raw = run_subgraphx_node_explanation(
        model, clean_graph, node_idx, num_classes, device,
        num_hops=args.num_hops, target_class=tclass,
        max_nodes=args.max_nodes, rollout=args.rollout,
        min_atoms=args.min_atoms, expand_atoms=args.expand_atoms,
        reward_method=args.reward_method,
    )
    subset, ego_ei, _, _ = get_2hop_ego_graph(
        clean_graph, node_idx, relabel_nodes=False, num_hops=args.num_hops
    )
    ego_edges = list(zip(ego_ei[0].tolist(), ego_ei[1].tolist()))
    return {
        "subset": [int(s) for s in subset.tolist()],
        "ego_edges": ego_edges,
        "important_nodes": imp_nodes,
        "important_edges": imp_edges,
        "target_class": raw["target_class"],
        "raw": raw,
    }

def explain_triggered_node(model, backdoor, clean_x, induct_edge_index,
                           node_idx, num_classes, device, args):
    (poison_x, poison_ei, trigger_nodes, trigger_edges,
     connection_edges) = inject_single_trigger(
        backdoor, node_idx, clean_x, induct_edge_index, device,
        args.trigger_size
    )
    tclass = _resolve_explain_class(
        args.explain_class, model, poison_x, poison_ei, node_idx,
        args.target_class
    )
    trig_graph = Data(x=poison_x, edge_index=poison_ei)
    imp_nodes, imp_edges, raw = run_subgraphx_node_explanation(
        model, trig_graph, node_idx, num_classes, device,
        num_hops=args.num_hops, target_class=tclass,
        max_nodes=args.max_nodes, rollout=args.rollout,
        min_atoms=args.min_atoms, expand_atoms=args.expand_atoms,
        reward_method=args.reward_method,
    )
    subset, ego_ei, _, _ = get_2hop_ego_graph(
        trig_graph, node_idx, relabel_nodes=False, num_hops=args.num_hops
    )
    ego_edges = list(zip(ego_ei[0].tolist(), ego_ei[1].tolist()))
    return {
        "subset": [int(s) for s in subset.tolist()],
        "ego_edges": ego_edges,
        "important_nodes": imp_nodes,
        "important_edges": imp_edges,
        "trigger_nodes": trigger_nodes,
        "trigger_edges": trigger_edges,
        "connection_edges": connection_edges,
        "target_class": raw["target_class"],
        "raw": raw,
    }

def triggered_hits_trigger(trig_info):

    trig_nodes = set(int(t) for t in trig_info["trigger_nodes"])
    node_hit = len(set(trig_info["important_nodes"]) & trig_nodes) > 0
    imp_ek = _edge_key_set(trig_info["important_edges"])
    trig_ek = _edge_key_set(trig_info["trigger_edges"]) | \
        _edge_key_set(trig_info["connection_edges"])
    edge_hit = len(imp_ek & trig_ek) > 0
    return node_hit or edge_hit, node_hit, edge_hit

def main():
    args = get_args()
    device = resolve_device(args.device)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    out_dir = os.path.join(_THIS_DIR, args.save_dir, args.dataset,
                           f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[subgraphx-vis] dataset={args.dataset} seed={args.seed} "
          f"target_class={args.target_class} device={device}")
    print(f"[subgraphx-vis] output dir: {out_dir}")

    data = build_dataset(
        args.dataset, args.seed, device, args.trigger_size, args.vs_size,
        args.target_class, args.data_root,
    )
    induct_edge_index = build_induct_edge_index(data)
    clean_x = data.x

    model, selection, num_classes = load_bgnas_model(
        args.dataset, args.seed, data, device, args.models_root,
        hidden_dim=args.hidden_dim, layer_number=args.layer_number,
        dropout=args.dropout,
    )
    print(f"[subgraphx-vis] loaded victim model; alpha_star={selection}")
    backdoor = load_trigger_backdoor(
        args.dataset, args.seed, device, args.models_root, args.trigger_size,
        args.target_class, args.thrd,
    )
    print(f"[subgraphx-vis] loaded trigger generator (trigger_size="
          f"{args.trigger_size}); num_classes={num_classes}")

    if args.node_list:
        node_ids = [int(t) for t in args.node_list.split(",") if t.strip() != ""]
    else:
        node_ids = data.test_mask.nonzero(as_tuple=False).flatten().tolist()
        if args.max_test_nodes and args.max_test_nodes > 0:
            node_ids = node_ids[: args.max_test_nodes]
    print(f"[subgraphx-vis] explaining {len(node_ids)} candidate test nodes ...")

    summary_path = os.path.join(out_dir, "summary.json")
    matched, records, n_plotted = [], [], 0

    def save_summary(status):

        summary = {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "target_class": int(args.target_class),
            "num_hops": int(args.num_hops),
            "explain_class": args.explain_class,
            "status": status,
            "num_candidate_nodes": len(node_ids),
            "num_processed_nodes": len(records),
            "num_matched_nodes": len(matched),
            "matched_nodes": matched,
            "num_figures": n_plotted,
            "records": records,
        }
        tmp_path = summary_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(summary, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, summary_path)

    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt()

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except Exception:
        pass

    save_summary("running")
    interrupted = False
    try:
        for i, node_idx in enumerate(node_ids):
            node_idx = int(node_idx)
            try:
                clean_info = explain_clean_node(
                    model, clean_x, induct_edge_index, node_idx, num_classes,
                    device, args,
                )
                trig_info = explain_triggered_node(
                    model, backdoor, clean_x, induct_edge_index, node_idx,
                    num_classes, device, args,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"  [{i+1}/{len(node_ids)}] node {node_idx}: FAILED ({exc})")
                continue

            hit, node_hit, edge_hit = triggered_hits_trigger(trig_info)
            records.append({
                "node": node_idx,
                "clean_class": clean_info["target_class"],
                "trig_class": trig_info["target_class"],
                "trigger_in_important": bool(hit),
                "trigger_node_hit": bool(node_hit),
                "trigger_edge_hit": bool(edge_hit),
                "clean_important_nodes": sorted(int(n) for n in clean_info["important_nodes"]),
                "trig_important_nodes": sorted(int(n) for n in trig_info["important_nodes"]),
                "trigger_nodes": [int(n) for n in trig_info["trigger_nodes"]],
            })
            flag = "HIT " if hit else "    "
            print(f"  [{i+1}/{len(node_ids)}] node {node_idx} {flag}"
                  f"clean_cls={clean_info['target_class']} "
                  f"trig_cls={trig_info['target_class']} "
                  f"node_hit={node_hit} edge_hit={edge_hit}")

            if hit or args.plot_all_tested:
                if n_plotted < args.max_plot:
                    save_path = os.path.join(out_dir, f"node_{node_idx}.png")
                    visualize_clean_vs_triggered(
                        node_idx, clean_info, trig_info, save_path,
                        layout_seed=args.seed_vis,
                    )
                    print(f"        -> figure saved: {save_path}")
                    n_plotted += 1
                if hit:
                    matched.append(node_idx)

            save_summary("running")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[subgraphx-vis] interrupted — flushing partial summary ...")
    finally:
        save_summary("interrupted" if interrupted else "completed")

    print("\n========== SUBGRAPHX VISUALISATION SUMMARY ==========")
    print(f"status                    : "
          f"{'interrupted' if interrupted else 'completed'}")
    print(f"candidate nodes           : {len(node_ids)}")
    print(f"nodes processed           : {len(records)}")
    print(f"nodes whose triggered important sub-graph contains the trigger: "
          f"{len(matched)}")
    print(f"matched nodes: {matched}")
    print(f"figures rendered: {n_plotted} (in {out_dir})")
    print(f"summary: {summary_path}")
    print("=====================================================\n")

if __name__ == "__main__":
    main()
