#!/usr/bin/env python3

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_EXAMPLES_DIR, "..", ".."))
for _p in (_PROJECT_ROOT, _EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visualize_subgraphx_bgnas import (
    resolve_device,
    build_dataset,
    build_induct_edge_index,
    load_trigger_backdoor,
    get_2hop_ego_graph,
    inject_single_trigger,
)
from torch_geometric.data import Data

C_TARGET = "#FFD400"
C_TRIGGER = "#E8482B"
C_NEIGHBOUR = "#8FB8DE"
C_IMPORTANT_RING = "#1B7F1B"
C_NORMAL_EDGE = "#B4B4B4"
C_IMPORTANT_EDGE = "#000000"

def get_args():
    p = argparse.ArgumentParser(
        description="Clean-vs-triggered 2-hop sub-graph figures (plain and/or "
                    "SubgraphX-attributed) for every node in each summary.json."
    )
    p.add_argument("--root", type=str,
                   default=os.path.join(_EXAMPLES_DIR, "results", "subgraphx_vis"),
                   help="Root dir holding {dataset}/seed_{seed}/summary.json.")
    p.add_argument("--which", type=str, default="both",
                   choices=["both", "plain", "subgraphx"],
                   help="Which flavour(s) to render.")
    p.add_argument("--out_subdir_plain", type=str, default="plain_subgraphs")
    p.add_argument("--out_subdir_subgraphx", type=str,
                   default="subgraphx_subgraphs")
    p.add_argument("--only_matched", action="store_true", default=False,
                   help="Only draw nodes in 'matched_nodes' (default: all "
                        "nodes in 'records').")
    p.add_argument("--models_root", type=str, default="saved_models1")
    p.add_argument("--data_root", type=str,
                   default=os.path.join(_EXAMPLES_DIR, "data"))
    p.add_argument("--vs_size", type=int, default=40)
    p.add_argument("--trigger_size", type=int, default=3)
    p.add_argument("--thrd", type=float, default=0.5)
    p.add_argument("--seed_vis", type=int, default=0,
                   help="Spring-layout seed (must match the original run; the "
                        "default 0 matches visualize_subgraphx_bgnas.py).")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Use the same device as the original run; the saved "
                        "trigger generator stores an internal cuda device.")
    p.add_argument("--fig_w", type=float, default=11.0,
                   help="Figure width (both panels).")
    p.add_argument("--fig_h", type=float, default=5.2,
                   help="Figure height.")
    p.add_argument("--layout_k", type=float, default=0.42,
                   help="spring_layout k: smaller = tighter node spacing.")
    p.add_argument("--layout_scale", type=float, default=0.72,
                   help="Shrink the layout towards its centre (<1 = tighter).")
    p.add_argument("--margin", type=float, default=0.14,
                   help="Axes margin fraction around the drawn graph.")
    p.add_argument("--trigger_radius", type=float, default=0.30,
                   help="Radius at which trigger nodes are placed around target.")
    p.add_argument("--edge_scale", type=float, default=1.0,
                   help="Global multiplier on all edge widths.")
    p.add_argument("--node_scale", type=float, default=1.0,
                   help="Global multiplier on all node sizes.")
    return p.parse_args()

def build_clean_info(clean_x, induct_edge_index, node_idx, num_hops):
    clean_graph = Data(x=clean_x, edge_index=induct_edge_index)
    subset, ego_ei, _, _ = get_2hop_ego_graph(
        clean_graph, node_idx, relabel_nodes=False, num_hops=num_hops
    )
    ego_edges = list(zip(ego_ei[0].tolist(), ego_ei[1].tolist()))
    return {
        "subset": [int(s) for s in subset.tolist()],
        "ego_edges": ego_edges,
    }

def build_triggered_info(backdoor, clean_x, induct_edge_index, node_idx,
                         device, trigger_size, num_hops):
    (poison_x, poison_ei, trigger_nodes, trigger_edges,
     connection_edges) = inject_single_trigger(
        backdoor, node_idx, clean_x, induct_edge_index, device, trigger_size
    )
    trig_graph = Data(x=poison_x, edge_index=poison_ei)
    subset, ego_ei, _, _ = get_2hop_ego_graph(
        trig_graph, node_idx, relabel_nodes=False, num_hops=num_hops
    )
    ego_edges = list(zip(ego_ei[0].tolist(), ego_ei[1].tolist()))
    return {
        "subset": [int(s) for s in subset.tolist()],
        "ego_edges": ego_edges,
        "trigger_nodes": trigger_nodes,
        "trigger_edges": trigger_edges,
        "connection_edges": connection_edges,
    }

def _key(u, v):
    u, v = int(u), int(v)
    return (u, v) if u <= v else (v, u)

def _edge_key_set(edges):
    return {_key(u, v) for (u, v) in edges}

def _induced_edge_keys(important_nodes, ego_edges):

    imp = set(int(n) for n in important_nodes)
    return {_key(u, v) for (u, v) in ego_edges
            if int(u) in imp and int(v) in imp}

def compute_layouts(node_idx, clean_info, trig_info, layout_seed,
                    layout_k, layout_scale, trigger_radius):
    c_subset = clean_info["subset"]
    base_g = nx.Graph()
    base_g.add_nodes_from(int(n) for n in c_subset)
    for (u, v) in clean_info["ego_edges"]:
        base_g.add_edge(int(u), int(v))
    if base_g.number_of_nodes() == 0:
        base_g.add_node(node_idx)
    pos_clean = nx.spring_layout(base_g, seed=layout_seed, k=layout_k)

    if len(pos_clean) > 0:
        centre = np.mean(np.stack(list(pos_clean.values())), axis=0)
        for n in pos_clean:
            pos_clean[n] = centre + (pos_clean[n] - centre) * layout_scale

    trigger_nodes = trig_info["trigger_nodes"]
    t_neighbours = [int(n) for n in trig_info["subset"]
                    if int(n) != node_idx and int(n) not in set(trigger_nodes)]
    pos_trig = dict(pos_clean)
    tx, ty = pos_clean.get(node_idx, np.array([0.0, 0.0]))
    rng = np.random.RandomState(layout_seed + 1)
    for i, tn in enumerate(sorted(trigger_nodes)):
        ang = 2 * np.pi * i / max(1, len(trigger_nodes))
        pos_trig[tn] = np.array([
            tx + trigger_radius * np.cos(ang) + 0.035 * rng.randn(),
            ty + trigger_radius * np.sin(ang) + 0.035 * rng.randn(),
        ])
    for n in set(t_neighbours) | {node_idx} | set(trigger_nodes):
        if n not in pos_trig:
            pos_trig[n] = np.array([0.5 * rng.randn(), 0.5 * rng.randn()])
    return pos_clean, pos_trig

def _draw_panel(ax, title, ego_edges, target_node, neighbour_nodes,
                trigger_nodes, trigger_ek, connection_ek,
                important_nodes, important_ek, pos, style):
    g = nx.Graph()
    all_nodes = set(neighbour_nodes) | {target_node} | set(trigger_nodes)
    g.add_nodes_from(all_nodes)
    for (u, v) in ego_edges:
        g.add_edge(int(u), int(v))

    trigger_set = set(int(t) for t in trigger_nodes)
    important_set = set(int(n) for n in important_nodes)
    ns = style["node_scale"]
    ew = style["edge_scale"]

    face, size, ecol, elw = [], [], [], []
    for n in g.nodes():
        if n == target_node:
            face.append(C_TARGET); size.append(1100 * ns)
        elif n in trigger_set:
            face.append(C_TRIGGER); size.append(560 * ns)
        else:
            face.append(C_NEIGHBOUR); size.append(420 * ns)
        if n in important_set:
            ecol.append(C_IMPORTANT_RING); elw.append(3.6)
        else:
            ecol.append("#333333"); elw.append(1.0)

    normal_e, important_e, trigger_e, trigger_imp_e = [], [], [], []
    for (u, v) in g.edges():
        k = _key(u, v)
        is_trig = (k in trigger_ek) or (k in connection_ek)
        is_imp = k in important_ek
        if is_trig and is_imp:
            trigger_imp_e.append((u, v))
        elif is_trig:
            trigger_e.append((u, v))
        elif is_imp:
            important_e.append((u, v))
        else:
            normal_e.append((u, v))

    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=normal_e,
                           edge_color=C_NORMAL_EDGE, width=2.0 * ew)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=important_e,
                           edge_color=C_IMPORTANT_EDGE, width=4.8 * ew)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=trigger_e,
                           edge_color=C_TRIGGER, width=3.4 * ew, style="dashed")
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=trigger_imp_e,
                           edge_color=C_TRIGGER, width=5.4 * ew, style="solid")

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=face, node_size=size,
                           edgecolors=ecol, linewidths=elw)
    labels = {target_node: str(target_node)} if target_node in g.nodes() else {}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax,
                            font_size=style["label_fontsize"],
                            font_color="black")

    ax.set_title(title, fontsize=style["title_fontsize"])
    ax.margins(style["margin"])
    ax.axis("off")

def render_pair(node_idx, clean_info, trig_info, pos_clean, pos_trig,
                clean_important, trig_important, show_important, save_path,
                style, suptitle):

    c_neighbours = [int(n) for n in clean_info["subset"] if int(n) != node_idx]
    trigger_nodes = trig_info["trigger_nodes"]
    t_neighbours = [int(n) for n in trig_info["subset"]
                    if int(n) != node_idx and int(n) not in set(trigger_nodes)]
    trigger_ek = _edge_key_set(trig_info["trigger_edges"])
    connection_ek = _edge_key_set(trig_info["connection_edges"])

    if show_important:
        c_imp_nodes = clean_important["nodes"]
        c_imp_ek = clean_important["edge_keys"]
        t_imp_nodes = trig_important["nodes"]
        t_imp_ek = trig_important["edge_keys"]
        tag = "SubgraphX"
    else:
        c_imp_nodes, c_imp_ek = set(), set()
        t_imp_nodes, t_imp_ek = set(), set()
        tag = "2-hop sub-graph"

    fig, axes = plt.subplots(1, 2, figsize=(style["fig_w"], style["fig_h"]))
    _draw_panel(axes[0], f"CLEAN  |  node {node_idx}  ({tag})",
                clean_info["ego_edges"], node_idx, c_neighbours,
                [], set(), set(), c_imp_nodes, c_imp_ek, pos_clean, style)
    _draw_panel(axes[1], f"TRIGGERED  |  node {node_idx}  ({tag})",
                trig_info["ego_edges"], node_idx, t_neighbours, trigger_nodes,
                trigger_ek, connection_ek, t_imp_nodes, t_imp_ek, pos_trig,
                style)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TARGET,
               markeredgecolor="#333333", markersize=14, label="target node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NEIGHBOUR,
               markeredgecolor="#333333", markersize=11, label="original neighbour"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TRIGGER,
               markeredgecolor="#333333", markersize=11, label="trigger node"),
        Line2D([0], [0], color=C_NORMAL_EDGE, lw=2.4, label="normal edge"),
        Line2D([0], [0], color=C_TRIGGER, lw=3.2, ls="--", label="trigger edge"),
    ]
    if show_important:
        handles += [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#DDDDDD",
                   markeredgecolor=C_IMPORTANT_RING, markeredgewidth=3,
                   markersize=13, label="SubgraphX important node"),
            Line2D([0], [0], color=C_IMPORTANT_EDGE, lw=4.6,
                   label="SubgraphX important edge"),
            Line2D([0], [0], color=C_TRIGGER, lw=5.0,
                   label="trigger & important edge"),
        ]
    ncol = 4 if show_important else 5
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

def process_summary(summary_path, args, device):
    with open(summary_path) as f:
        summary = json.load(f)

    dataset = summary["dataset"]
    seed = int(summary["seed"])
    target_class = int(summary.get("target_class", 0))
    num_hops = int(summary.get("num_hops", 2))

    records = summary.get("records", [])
    rec_by_node = {int(r["node"]): r for r in records}
    if args.only_matched:
        matched = set(int(n) for n in summary.get("matched_nodes", []))
        node_ids = [int(r["node"]) for r in records if int(r["node"]) in matched]
    else:
        node_ids = [int(r["node"]) for r in records]

    seed_dir = os.path.dirname(summary_path)
    do_plain = args.which in ("both", "plain")
    do_sgx = args.which in ("both", "subgraphx")
    out_plain = os.path.join(seed_dir, args.out_subdir_plain)
    out_sgx = os.path.join(seed_dir, args.out_subdir_subgraphx)
    if do_plain:
        os.makedirs(out_plain, exist_ok=True)
    if do_sgx:
        os.makedirs(out_sgx, exist_ok=True)

    print(f"\n[subgraph-vis] {dataset} seed={seed}: {len(node_ids)} nodes "
          f"(which={args.which}) -> {seed_dir}", flush=True)

    data = build_dataset(
        dataset, seed, device, args.trigger_size, args.vs_size,
        target_class, args.data_root,
    )
    induct_edge_index = build_induct_edge_index(data)
    clean_x = data.x
    backdoor = load_trigger_backdoor(
        dataset, seed, device, args.models_root, args.trigger_size,
        target_class, args.thrd,
    )
    try:
        backdoor.trojan.device = device
        if hasattr(backdoor.trojan, "mid_space"):
            backdoor.trojan.mid_space.device = device
    except Exception:
        pass

    style = {
        "fig_w": args.fig_w, "fig_h": args.fig_h, "margin": args.margin,
        "edge_scale": args.edge_scale, "node_scale": args.node_scale,
        "label_fontsize": 9, "title_fontsize": 12,
    }

    n_ok = 0
    for i, node_idx in enumerate(node_ids):
        try:
            clean_info = build_clean_info(
                clean_x, induct_edge_index, node_idx, num_hops
            )
            trig_info = build_triggered_info(
                backdoor, clean_x, induct_edge_index, node_idx, device,
                args.trigger_size, num_hops,
            )
            pos_clean, pos_trig = compute_layouts(
                node_idx, clean_info, trig_info, args.seed_vis,
                args.layout_k, args.layout_scale, args.trigger_radius,
            )

            rec = rec_by_node.get(node_idx, {})
            c_imp = rec.get("clean_important_nodes", [node_idx])
            t_imp = rec.get("trig_important_nodes", [node_idx])
            clean_important = {
                "nodes": set(int(n) for n in c_imp),
                "edge_keys": _induced_edge_keys(c_imp, clean_info["ego_edges"]),
            }
            trig_important = {
                "nodes": set(int(n) for n in t_imp),
                "edge_keys": _induced_edge_keys(t_imp, trig_info["ego_edges"]),
            }

            if do_plain:
                render_pair(
                    node_idx, clean_info, trig_info, pos_clean, pos_trig,
                    clean_important, trig_important, show_important=False,
                    save_path=os.path.join(out_plain, f"node_{node_idx}.png"),
                    style=style,
                    suptitle=f"Clean vs Triggered 2-hop local sub-graph "
                             f"— node {node_idx}",
                )
            if do_sgx:
                render_pair(
                    node_idx, clean_info, trig_info, pos_clean, pos_trig,
                    clean_important, trig_important, show_important=True,
                    save_path=os.path.join(out_sgx, f"node_{node_idx}.png"),
                    style=style,
                    suptitle=f"DIG SubgraphX 2-hop local explanation "
                             f"— node {node_idx}",
                )
            n_ok += 1
            print(f"  [{i+1}/{len(node_ids)}] node {node_idx} ok", flush=True)
        except Exception as exc:
            print(f"  [{i+1}/{len(node_ids)}] node {node_idx}: FAILED ({exc})",
                  flush=True)
    print(f"[subgraph-vis] {dataset} seed={seed}: {n_ok}/{len(node_ids)} nodes "
          f"rendered.", flush=True)
    return n_ok

def main():
    args = get_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    device = resolve_device(args.device)

    summaries = sorted(glob.glob(
        os.path.join(args.root, "*", "seed_*", "summary.json")
    ))
    if not summaries:
        print(f"[subgraph-vis] no summary.json found under {args.root}")
        return
    print(f"[subgraph-vis] found {len(summaries)} summary.json file(s):")
    for s in summaries:
        print(f"  - {s}")

    total = 0
    for s in summaries:
        try:
            total += process_summary(s, args, device)
        except Exception as exc:
            print(f"[subgraph-vis] FAILED for {s}: {exc}")
    print(f"\n[subgraph-vis] done. total nodes rendered: {total}")

if __name__ == "__main__":
    main()
