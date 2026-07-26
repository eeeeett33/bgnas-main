#!/usr/bin/env python3

import argparse
import copy
import itertools
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import torch

from autogllight.utils import set_seed
from autogllight.nas.space.graph_nas import (
    GraphBenchmarkingSpace,
    GRAPHNAS_DEFAULT_GNN_OPS,
)
from autogllight.examples.training_pipeline import run_pipeline_like_edgepruning

try:
    from path_attribution import loaders
except ModuleNotFoundError:
    from types import SimpleNamespace
    from utils.data_atk import GraphDataLoader

    def _normalise_selection(selection):
        out = {}
        for k, v in selection.items():
            if k.startswith("in_"):
                out[k] = ([int(i) for i in v]
                          if isinstance(v, (list, tuple)) else [int(v)])
            else:
                out[k] = int(v)
        return out

    def _load_architecture(dataset, seed, arch_path=None, device="cpu"):
        arch_cache = os.path.join(_THIS_DIR, "saved_models1", dataset,
                                  f"seed_{seed}", "alpha_star.json")
        full_obj = os.path.join(_THIS_DIR, "saved_models", dataset,
                                f"seed_{seed}", "fixed_model_obj.pt")
        if arch_path is not None:
            with open(arch_path) as f:
                return _normalise_selection(json.load(f))
        if os.path.exists(arch_cache):
            with open(arch_cache) as f:
                return _normalise_selection(json.load(f))
        obj = torch.load(full_obj, map_location=device, weights_only=False)
        selection = _normalise_selection(getattr(obj, "selection"))
        try:
            os.makedirs(os.path.dirname(arch_cache), exist_ok=True)
            with open(arch_cache, "w") as f:
                json.dump(selection, f, indent=2)
        except OSError:
            pass
        return selection

    def _build_dataset(dataset, device, trigger_size=3, vs_size=40,
                       target_class=0, root="./data"):
        loader = GraphDataLoader(
            device=device, dataset_name=dataset, root=root,
            trigger_size=trigger_size, vs_size=vs_size,
            target_class=target_class, split=True,
        )
        return loader.load_data()

    loaders = SimpleNamespace(load_architecture=_load_architecture,
                              build_dataset=_build_dataset)

from retrain_fixed_gen import _reset_module_parameters, load_saved_components

def get_args():
    p = argparse.ArgumentParser(
        description="Macro-architecture ablation (fixed operators, swapped "
                    "wiring) for searched backdoored GNNs."
    )
    p.add_argument("--dataset", type=str, default="Pubmed",
                   choices=["Citeseer", "Flickr", "Cora", "Pubmed", "Actor",
                            "chameleon", "Computers", "Photo", "CS"])
    p.add_argument("--arch_seed", type=int, default=1333,
                   help="Seed identifying the saved architecture + generator "
                        "(saved_models/{dataset}/seed_{arch_seed}/).")
    p.add_argument("--set_seed", type=int, default=0,
                   help="Random seed for this ablation run (re-init + training).")
    p.add_argument("--macros", type=str, default="others",
                   help="'others' (the 8 non-original topologies), 'all' (all 9),"
                        " or a comma-separated list of macro-class ids, e.g. '0,3,8'.")
    p.add_argument("--device", type=int, default=0)

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layer_number", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)

    p.add_argument("--trigger_size", type=int, default=3)
    p.add_argument("--target_class", type=int, default=0)
    p.add_argument("--vs_size", type=int, default=40)

    p.add_argument("--retrain_epochs", type=int, default=200,
                   help="Epochs to train the (target/shadow) model to convergence.")
    p.add_argument("--train_lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--retrain_lr", type=float, default=0.01)
    p.add_argument("--retrain_wd", type=float, default=5e-4)
    p.add_argument("--trojan_epochs", type=int, default=200,
                   help="Epochs to train / fine-tune the trigger generator.")
    p.add_argument("--freeze_trigger", action="store_true", default=False,
                   help="If set, do not train the generator (keep it fixed).")

    p.add_argument("--use_vs_number", action="store_true", default=True)
    p.add_argument("--vs_ratio", type=float, default=0.0)
    p.add_argument("--vs_number", type=int, default=40)
    p.add_argument("--selection_method", type=str, default="cluster_degree",
                   choices=["loss", "conf", "cluster", "none", "cluster_degree"])
    p.add_argument("--defense_mode", type=str, default="prune",
                   choices=["prune", "isolate", "none"])
    p.add_argument("--prune_thr", type=float, default=0.1)
    p.add_argument("--target_loss_weight", type=float, default=1.0)
    p.add_argument("--homo_loss_weight", type=float, default=100.0)
    p.add_argument("--homo_boost_thrd", type=float, default=0.5)
    p.add_argument("--dis_weight", type=float, default=1.0)
    p.add_argument("--thrd", type=float, default=0.5)
    p.add_argument("--debug", type=bool, default=True)

    p.add_argument("--save_dir", type=str, default="results/macro_arch_ablation")
    return p.parse_args()

def _canonical_rooted_tree(wiring):

    children = {n: [] for n in range(5)}
    for k in range(4):
        children[wiring[k]].append(k + 1)

    def encode(n):
        return "(" + "".join(sorted(encode(c) for c in children[n])) + ")"

    return encode(0)

def _describe_wiring(wiring):

    children = {n: [] for n in range(5)}
    parent = {}
    for k in range(4):
        children[wiring[k]].append(k + 1)
        parent[k + 1] = wiring[k]

    depth = {0: 0}
    for node in range(1, 5):
        chain, cur = 0, node
        while cur != 0:
            chain += 1
            cur = parent[cur]
        depth[node] = chain
    max_depth = max(depth[n] for n in range(1, 5))
    input_fanout = len(children[0])
    internal_edges = sum(1 for k in range(4) if wiring[k] != 0)

    if max_depth == 4:
        label = "chain (sparsest)"
    elif input_fanout == 4:
        label = "star (densest)"
    else:
        label = f"depth{max_depth}_infanout{input_fanout}"
    return {
        "max_depth": int(max_depth),
        "input_fanout": int(input_fanout),
        "internal_edges": int(internal_edges),
        "label": label,
    }

def enumerate_macro_topologies():

    wirings = [
        (0, i1, i2, i3)
        for i1 in range(2) for i2 in range(3) for i3 in range(4)
    ]
    groups = {}
    for w in wirings:
        groups.setdefault(_canonical_rooted_tree(w), []).append(w)

    classes = []
    for canon, members in groups.items():
        rep = min(members)
        desc = _describe_wiring(rep)
        classes.append({
            "canonical_form": canon,
            "rep_wiring": rep,
            "members": members,
            **desc,
        })
    classes.sort(key=lambda c: (-c["input_fanout"], c["max_depth"],
                                c["canonical_form"]))
    for cid, c in enumerate(classes):
        c["class_id"] = cid
    return classes

def wiring_from_selection(selection):

    def _idx(key):
        v = selection[key]
        return int(v[0]) if isinstance(v, (list, tuple)) else int(v)
    return tuple(_idx(f"in_{k}") for k in range(4))

def selection_with_wiring(selection, wiring):

    new = copy.deepcopy(selection)
    for k in range(4):
        new[f"in_{k}"] = [int(wiring[k])]
    return new

def find_class_id(classes, wiring):
    canon = _canonical_rooted_tree(tuple(wiring))
    for c in classes:
        if c["canonical_form"] == canon:
            return c["class_id"]
    return None

def build_model_from_selection(selection, data, device, args):

    input_dim = int(data.num_node_features)
    output_dim = int(data.y.max().item() + 1)
    space = GraphBenchmarkingSpace(
        hidden_dim=args.hidden,
        layer_number=args.layer_number,
        dropout=args.dropout,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    box = space.parse_model(selection)
    return box._model.to(device)

def ops_of_selection(selection):

    ops = []
    for k in range(4):
        idx = int(selection[f"op_{k}"])
        name = (GRAPHNAS_DEFAULT_GNN_OPS[idx]
                if 0 <= idx < len(GRAPHNAS_DEFAULT_GNN_OPS) else str(idx))
        ops.append(name)
    return ops

def run_one_macro(args, cls, wiring, base_selection, base_generator, device):

    selection = selection_with_wiring(base_selection, wiring)

    set_seed(int(args.arch_seed))
    data = loaders.build_dataset(
        args.dataset, device, trigger_size=args.trigger_size,
        vs_size=args.vs_size, target_class=args.target_class, root="./data",
    )

    set_seed(int(args.set_seed))
    model = build_model_from_selection(selection, data, device, args)
    _reset_module_parameters(model)
    for pmt in model.parameters():
        pmt.requires_grad = True
    model.train()

    generator = copy.deepcopy(base_generator).to(device)
    for pmt in generator.parameters():
        pmt.requires_grad = (not args.freeze_trigger)
    generator.train() if not args.freeze_trigger else generator.eval()

    result = run_pipeline_like_edgepruning(
        args, data, device, model, fixed_trigger=generator
    )
    return float(result.get("CA", 0.0)), float(result.get("ASR", 0.0))

def resolve_targets(macros_arg, classes, original_id):

    macros_arg = str(macros_arg).strip().lower()
    all_ids = [c["class_id"] for c in classes]
    if macros_arg == "all":
        return all_ids
    if macros_arg == "others":
        return [cid for cid in all_ids if cid != original_id]
    ids = [int(x) for x in macros_arg.split(",") if x.strip() != ""]
    for cid in ids:
        if cid not in all_ids:
            raise ValueError(f"Unknown macro class id {cid}; valid ids: {all_ids}")
    return ids

def main():
    args = get_args()
    device = torch.device(
        ("cuda:{}" if torch.cuda.is_available() else "cpu").format(args.device)
    )

    out_dir = os.path.join(args.save_dir, args.dataset,
                           f"arch_seed_{args.arch_seed}",
                           f"set_seed_{args.set_seed}")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(int(args.arch_seed))
    base_selection = loaders.load_architecture(
        args.dataset, args.arch_seed, device=device
    )
    _, base_generator = load_saved_components(args.dataset, int(args.arch_seed), device)

    ops = ops_of_selection(base_selection)
    original_wiring = wiring_from_selection(base_selection)

    classes = enumerate_macro_topologies()
    original_id = find_class_id(classes, original_wiring)

    print(f"[macro_ablation] dataset={args.dataset} arch_seed={args.arch_seed} "
          f"set_seed={args.set_seed} device={device}")
    print(f"[macro_ablation] fixed operators (op_0..op_3): {ops}")
    print(f"[macro_ablation] original wiring (in_0..in_3): {original_wiring} "
          f"-> macro class {original_id}")
    print(f"[macro_ablation] {len(classes)} macro topologies:")
    for c in classes:
        star = " <== ORIGINAL" if c["class_id"] == original_id else ""
        print(f"    class {c['class_id']}: rep_wiring={c['rep_wiring']} "
              f"label={c['label']:<18} depth={c['max_depth']} "
              f"input_fanout={c['input_fanout']} form={c['canonical_form']}{star}")

    targets = resolve_targets(args.macros, classes, original_id)
    print(f"[macro_ablation] running macro classes: {targets}")

    rows = []
    cls_by_id = {c["class_id"]: c for c in classes}
    for cid in targets:
        cls = cls_by_id[cid]
        is_original = (cid == original_id)
        wiring = original_wiring if is_original else cls["rep_wiring"]
        print(f"\n[macro_ablation] === macro class {cid} "
              f"({cls['label']}, wiring={wiring}) ===")
        ca, asr = run_one_macro(
            args, cls, wiring, base_selection, base_generator, device
        )
        print(f"[macro_ablation] class {cid}: CA={ca:.4f} ASR={asr:.4f}")
        rows.append({
            "dataset": args.dataset,
            "arch_seed": int(args.arch_seed),
            "set_seed": int(args.set_seed),
            "macro_class": int(cid),
            "is_original": bool(is_original),
            "label": cls["label"],
            "canonical_form": cls["canonical_form"],
            "wiring": "-".join(str(w) for w in wiring),
            "max_depth": cls["max_depth"],
            "input_fanout": cls["input_fanout"],
            "internal_edges": cls["internal_edges"],
            "ops": "|".join(ops),
            "CA": ca,
            "ASR": asr,
        })

    df = pd.DataFrame(rows).sort_values("macro_class").reset_index(drop=True)
    csv_path = os.path.join(out_dir, "macro_ablation_results.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "dataset": args.dataset,
        "arch_seed": int(args.arch_seed),
        "set_seed": int(args.set_seed),
        "target_class": int(args.target_class),
        "fixed_operators": ops,
        "original_wiring": list(original_wiring),
        "original_macro_class": original_id,
        "num_macro_topologies": len(classes),
        "macro_topologies": [
            {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in c.items() if k != "members"}
            for c in classes
        ],
        "results": rows,
        "outputs": {"table": "macro_ablation_results.csv"},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== MACRO ARCHITECTURE ABLATION ==========")
    print(f"dataset={args.dataset} arch_seed={args.arch_seed} "
          f"set_seed={args.set_seed}")
    print(f"fixed operators: {ops}")
    print(f"{'class':>5} {'orig':>5} {'label':<18} {'wiring':<10} "
          f"{'depth':>5} {'fanin0':>6} {'CA':>8} {'ASR':>8}")
    for _, r in df.iterrows():
        print(f"{r['macro_class']:>5} {'*' if r['is_original'] else '':>5} "
              f"{r['label']:<18} {r['wiring']:<10} {r['max_depth']:>5} "
              f"{r['input_fanout']:>6} {r['CA']:>8.4f} {r['ASR']:>8.4f}")
    print(f"\nAll outputs written to: {os.path.abspath(out_dir)}")
    print("=================================================\n")

if __name__ == "__main__":
    main()
