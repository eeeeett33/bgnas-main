#!/usr/bin/env python3

import argparse
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
from path_attribution import loaders, branch_meta as bmeta, metrics, masking, plots
from path_attribution.scoring import compute_branch_final_scores

def get_args():
    p = argparse.ArgumentParser(
        description="BGNAS branch/path attribution + causal masking analysis."
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--seed", type=int, default=1047,
                   help="Seed identifying the saved model AND reproducing the split.")
    p.add_argument("--target_class", type=int, default=0)
    p.add_argument("--arch_path", type=str, default=None,
                   help="Optional alpha_star.json override.")
    p.add_argument("--model_ckpt", type=str, default=None,
                   help="Optional victim state_dict override.")
    p.add_argument("--generator_ckpt", type=str, default=None,
                   help="Optional trigger generator override.")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--layer_number", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--vs_size", type=int, default=40)
    p.add_argument("--trigger_size", type=int, default=None,
                   help="Trigger size; inferred from the generator when omitted.")
    p.add_argument("--thrd", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=0,
                   help="0 = no batching (inject all attack nodes at once).")
    p.add_argument("--ig_steps", type=int, default=20)
    p.add_argument("--topk_branches", type=int, default=3)
    p.add_argument("--random_mask_trials", type=int, default=10)
    p.add_argument("--channel_topk", type=float, default=0.1,
                   help="Top fraction of channels to mask in channel analysis.")
    p.add_argument("--no_channel", action="store_true",
                   help="Disable the (optional) channel-level analysis.")
    p.add_argument("--save_dir", type=str, default="results/path_attribution")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()

def _batch_size(args):
    return None if args.batch_size in (None, 0) else int(args.batch_size)

def _highlight(df, baseline, asr_drop_thresh=0.5, acc_drop_thresh=0.1):

    flagged = []
    base_asr = baseline["ASR"]
    for _, r in df.iterrows():
        d_asr = r.get("delta_ASR_after_masking", float("nan"))
        d_acc = r.get("delta_ACC_after_masking", float("nan"))
        if not np.isfinite(d_asr) or not np.isfinite(d_acc):
            continue
        rel_asr = d_asr / base_asr if base_asr > 1e-8 else 0.0
        if rel_asr >= asr_drop_thresh and d_acc <= acc_drop_thresh:
            flagged.append({
                "branch_id": r["branch_id"],
                "op": r["op"],
                "delta_ASR_after_masking": float(d_asr),
                "delta_ACC_after_masking": float(d_acc),
                "asr_relative_drop": float(rel_asr),
            })
    return flagged

def main():
    args = get_args()
    device = loaders.resolve_device(args.device)
    set_seed(int(args.seed))

    out_dir = os.path.join(args.save_dir, args.dataset, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[path_attribution] dataset={args.dataset} seed={args.seed} "
          f"target_class={args.target_class} device={device}")
    print(f"[path_attribution] output dir: {os.path.abspath(out_dir)}")

    ctx = loaders.build_eval_context(
        args.dataset, args.seed, device,
        target_class=args.target_class, vs_size=args.vs_size,
        data_root=args.data_root,
    )
    data = ctx["data"]
    induct_edge_index = ctx["induct_edge_index"]
    idx_clean_test = ctx["idx_clean_test"]
    idx_atk = ctx["idx_atk"]
    y = data.y
    print(f"[path_attribution] |idx_clean_test|={len(idx_clean_test)} "
          f"|idx_atk (ASR subset)|={len(idx_atk)}")

    selection = loaders.load_architecture(
        args.dataset, args.seed, arch_path=args.arch_path, device=device
    )
    print(f"[path_attribution] selection (alpha_star): {selection}")

    paths = loaders.resolve_paths(
        args.dataset, args.seed, arch_path=args.arch_path,
        model_ckpt=args.model_ckpt, generator_ckpt=args.generator_ckpt,
    )
    model = loaders.build_victim_model(
        selection, data, device, paths["model_ckpt"],
        hidden_dim=args.hidden_dim, layer_number=args.layer_number,
        dropout=args.dropout,
    )
    backdoor, trigger_size = loaders.load_trigger(
        paths["generator_ckpt"], data, device,
        target_class=args.target_class, thrd=args.thrd,
        trigger_size=args.trigger_size,
    )
    print(f"[path_attribution] trigger_size={trigger_size}")

    branch_meta = bmeta.build_branch_metadata(selection, args.hidden_dim)
    gate_keys = [b["gate_key"] for b in branch_meta]
    meta_by_key = {b["gate_key"]: b for b in branch_meta}
    pd.DataFrame(branch_meta).to_csv(
        os.path.join(out_dir, "branch_metadata.csv"), index=False
    )
    print(f"[path_attribution] {len(branch_meta)} branches: "
          f"{[b['branch_id'] for b in branch_meta]}")

    clean_graph = loaders.make_clean_graph(data, induct_edge_index)
    triggered_graph = loaders.make_triggered_graph(
        data, induct_edge_index, idx_atk, backdoor, device
    )

    def make_triggered(nodes):
        return loaders.make_triggered_graph(
            data, induct_edge_index, nodes, backdoor, device
        )

    bs = _batch_size(args)

    print("[path_attribution] computing branch gradient attribution ...")
    grad_attr = metrics.compute_branch_gradient_attribution(
        model, clean_graph, make_triggered, idx_atk, args.target_class,
        gate_keys, device, batch_size=bs,
    )
    grad_rows = []
    for k in gate_keys:
        b = meta_by_key[k]
        grad_rows.append({
            "dataset": args.dataset,
            "branch_id": b["branch_id"], "source": b["source"],
            "target": b["target"], "op": b["op"],
            **grad_attr[k],
        })
    pd.DataFrame(grad_rows).to_csv(
        os.path.join(out_dir, "branch_gradient_attribution.csv"), index=False
    )

    print("[path_attribution] computing branch integrated gradients ...")
    ig_attr = metrics.compute_branch_integrated_gradients(
        model, clean_graph, make_triggered, idx_atk, args.target_class,
        gate_keys, device, ig_steps=args.ig_steps, batch_size=bs,
    )
    ig_rows = []
    for k in gate_keys:
        b = meta_by_key[k]
        ig_rows.append({
            "dataset": args.dataset,
            "branch_id": b["branch_id"], "source": b["source"],
            "target": b["target"], "op": b["op"],
            **ig_attr[k],
        })
    pd.DataFrame(ig_rows).to_csv(
        os.path.join(out_dir, "branch_integrated_gradients.csv"), index=False
    )

    print("[path_attribution] computing branch activation analysis ...")
    act_attr = metrics.compute_branch_activation_analysis(
        model, clean_graph, triggered_graph, idx_atk, args.target_class,
        branch_meta, device,
    )
    act_rows = []
    for k in gate_keys:
        b = meta_by_key[k]
        act_rows.append({
            "dataset": args.dataset,
            "branch_id": b["branch_id"], "source": b["source"],
            "target": b["target"], "op": b["op"],
            **act_attr[k],
        })
    pd.DataFrame(act_rows).to_csv(
        os.path.join(out_dir, "branch_activation_analysis.csv"), index=False
    )

    print("[path_attribution] running causal masking validation ...")
    baseline, mask_rows = masking.evaluate_branch_masking(
        model, clean_graph, triggered_graph, idx_clean_test, idx_atk, y,
        args.target_class, branch_meta, device, dataset=args.dataset,
        random_trials=args.random_mask_trials, seed=args.seed,
    )
    pd.DataFrame(mask_rows).to_csv(
        os.path.join(out_dir, "branch_masking_validation.csv"), index=False
    )
    print(f"[path_attribution] baseline ACC={baseline['ACC']:.4f} "
          f"ASR={baseline['ASR']:.4f} ASR-C={baseline['ASR_C']:.4f}")

    mask_by_branch = {
        r["masked_branches"]: r for r in mask_rows
        if r["masking_type"] == "single_branch"
    }

    final_rows = []
    for b in branch_meta:
        k = b["gate_key"]
        bid = b["branch_id"]
        mrow = mask_by_branch.get(bid, {})
        final_rows.append({
            "dataset": args.dataset,
            "branch_id": bid, "source": b["source"], "target": b["target"],
            "op": b["op"],
            "grad_attr_trigger": grad_attr[k]["grad_attr_trigger"],
            "grad_attr_clean": grad_attr[k]["grad_attr_clean"],
            "grad_attr_trigger_specific": grad_attr[k]["grad_attr_trigger_specific"],
            "ig_trigger_specific": ig_attr[k]["ig_trigger_specific"],
            "activation_diff_ratio": act_attr[k]["activation_diff_ratio"],
            "target_aligned_activation_score":
                act_attr[k]["target_aligned_activation_score"],
            "ACC_after_masking": mrow.get("ACC", float("nan")),
            "ASR_after_masking": mrow.get("ASR", float("nan")),
            "delta_ACC_after_masking": mrow.get("delta_ACC", float("nan")),
            "delta_ASR_after_masking": mrow.get("delta_ASR", float("nan")),
        })
    final_df = pd.DataFrame(final_rows)
    final_df = compute_branch_final_scores(final_df)
    final_df = final_df.sort_values("final_rank").reset_index(drop=True)
    final_df.to_csv(os.path.join(out_dir, "branch_final_scores.csv"), index=False)

    top_df = final_df.head(max(1, args.topk_branches))
    top_ids = set(top_df["branch_id"].tolist())
    top_keys = [b["gate_key"] for b in branch_meta if b["branch_id"] in top_ids]

    channel_masking = {}
    if not args.no_channel:
        print("[path_attribution] computing branch-channel attribution "
              f"(top {args.topk_branches} branches) ...")
        try:
            ch_rows, channel_masking = masking.compute_branch_channel_attribution(
                model, clean_graph, triggered_graph, idx_atk, idx_clean_test, y,
                args.target_class, branch_meta, args.hidden_dim, device,
                branch_keys=top_keys, channel_topk=args.channel_topk,
            )
            pd.DataFrame(ch_rows).to_csv(
                os.path.join(out_dir, "branch_channel_attribution.csv"), index=False
            )
        except RuntimeError as exc:
            print(f"[path_attribution] channel analysis skipped (likely OOM): {exc}")
    else:
        print("[path_attribution] channel analysis disabled (--no_channel). "
              "Interface available via masking.compute_branch_channel_attribution.")

    print("[path_attribution] rendering figures ...")
    plots.plot_branch_attribution_heatmap(
        final_df, os.path.join(out_dir, "path_attribution_heatmap.png")
    )
    pearson, spearman = plots.plot_attribution_vs_asr_drop(
        final_df, os.path.join(out_dir, "attribution_vs_asr_drop.png")
    )
    final_scores = dict(zip(final_df["branch_id"], final_df["final_score"]))
    plots.plot_architecture_attribution_graph(
        branch_meta, final_scores, top_ids,
        os.path.join(out_dir, "architecture_attribution_graph.png"),
        os.path.join(out_dir, "architecture_attribution_graph.dot"),
    )

    flagged = _highlight(final_df, baseline)
    summary = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "target_class": int(args.target_class),
        "device": str(device),
        "trigger_size": int(trigger_size),
        "num_branches": len(branch_meta),
        "selection_alpha_star": selection,
        "asr_test_subset_size": int(len(idx_atk)),
        "clean_test_subset_size": int(len(idx_clean_test)),
        "baseline": baseline,
        "attribution_vs_asr_drop_correlation": {
            "pearson": pearson, "spearman": spearman,
        },
        "top_branches": top_df[[
            "branch_id", "op", "final_score", "final_rank",
            "grad_attr_trigger_specific", "ig_trigger_specific",
            "delta_ASR_after_masking", "delta_ACC_after_masking",
        ]].to_dict(orient="records"),
        "causal_highlight": {
            "description": ("Branches whose masking collapses ASR (relative drop "
                            ">= 0.5) while ACC drop <= 0.1 -> strong causal "
                            "backdoor evidence."),
            "branches": flagged,
        },
        "channel_masking": channel_masking,
        "outputs": {
            "branch_metadata": "branch_metadata.csv",
            "branch_gradient_attribution": "branch_gradient_attribution.csv",
            "branch_integrated_gradients": "branch_integrated_gradients.csv",
            "branch_activation_analysis": "branch_activation_analysis.csv",
            "branch_final_scores": "branch_final_scores.csv",
            "branch_masking_validation": "branch_masking_validation.csv",
            "branch_channel_attribution": (
                "branch_channel_attribution.csv" if (not args.no_channel
                                                     and channel_masking) else None),
            "heatmap": "path_attribution_heatmap.png",
            "attribution_vs_asr_drop": "attribution_vs_asr_drop.png",
            "architecture_graph_png": "architecture_attribution_graph.png",
            "architecture_graph_dot": "architecture_attribution_graph.dot",
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== PATH ATTRIBUTION SUMMARY ==========")
    print(f"baseline: ACC={baseline['ACC']:.4f} ASR={baseline['ASR']:.4f} "
          f"ASR-C={baseline['ASR_C']:.4f}")
    print(f"attribution vs ASR-drop correlation: "
          f"Pearson={pearson:.3f} Spearman={spearman:.3f}")
    print("top branches by final attribution score:")
    for _, r in top_df.iterrows():
        print(f"  - {r['branch_id']:<18} op={r['op']:<7} "
              f"final={r['final_score']:.3f} "
              f"dASR={r['delta_ASR_after_masking']:.3f} "
              f"dACC={r['delta_ACC_after_masking']:.3f}")
    if flagged:
        print("CAUSAL HIGHLIGHT (mask -> ASR collapses, ACC preserved):")
        for fl in flagged:
            print(f"  * {fl['branch_id']} (op={fl['op']}): "
                  f"dASR={fl['delta_ASR_after_masking']:.3f} "
                  f"(rel {fl['asr_relative_drop']:.2f}), "
                  f"dACC={fl['delta_ACC_after_masking']:.3f}")
    else:
        print("No single branch both collapses ASR and preserves ACC; the "
              "backdoor may be distributed across branches.")
    print(f"\nAll outputs written to: {os.path.abspath(out_dir)}")
    print("==============================================\n")

if __name__ == "__main__":
    main()
