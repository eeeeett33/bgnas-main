#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import random
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

import temp1 as bench

@contextmanager
def suppress_stdout():

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        yield

def _condition_number(ntk: Tensor) -> float:

    eigs = torch.linalg.eigvalsh(ntk)
    eigs_clipped = eigs.nan_to_num(nan=1e5, posinf=1e7, neginf=-1e7)
    return (eigs_clipped[-1] / eigs_clipped[0]).nan_to_num(nan=1e5).item()

def ntk_score_once(
    target_model: torch.nn.Module,
    generator: torch.nn.Module,
    data: Data,
    split_ctx: bench.SplitContext,
    proxy_nodes: Tensor,
    bd_args: Any,
    device: torch.device,
) -> float:

    target_model.eval()
    generator.eval()

    backdoor = bench.Backdoor(bd_args, device)
    backdoor.trojan = generator

    base_edge_index = split_ctx.train_edge_index
    edge_weight = torch.ones(base_edge_index.shape[1], device=device, dtype=torch.float)
    feat, edge_index, _ = backdoor.inject_trigger(
        proxy_nodes, data.x, base_edge_index, edge_weight, device
    )
    poisoned = Data(x=feat.detach(), edge_index=edge_index.detach())

    params = {
        name: p
        for name, p in target_model.named_parameters()
        if "weight" in name and p.requires_grad
    }
    if not params:
        raise ValueError("Target model exposes no trainable 'weight' parameters for NTK")
    names = list(params.keys())
    values = tuple(params[name] for name in names)

    def func(*flat_params: Tensor) -> Tensor:
        param_dict = {name: p for name, p in zip(names, flat_params)}
        logits = functional_call(
            target_model, param_dict, (poisoned,), {"return_logits": True}
        )
        return logits[proxy_nodes]

    batch_grads = torch.autograd.functional.jacobian(func, values)
    batch_grad = torch.cat([g.flatten(2).detach() for g in batch_grads], dim=-1)
    ntk = (batch_grad @ batch_grad.transpose(1, 2)).mean(0)
    return _condition_number(ntk)

class GraphArchSearchSpace:

    def __init__(self, nb_arch: Any, use_proteins: bool, mutate_link_prob: float):
        self.nb_arch = nb_arch
        self.records: list[dict[str, Any]] = bench.enumerate_target_architectures(
            nb_arch, use_proteins
        )
        self.op_candidates: list[str] = list(
            nb_arch.gnn_list_proteins if use_proteins else nb_arch.gnn_list
        )
        self.mutate_link_prob = float(mutate_link_prob)
        self.hash_to_index = {
            int(rec["architecture_id"]): idx for idx, rec in enumerate(self.records)
        }

    def __len__(self) -> int:
        return len(self.records)

    def record(self, index: int) -> dict[str, Any]:
        return self.records[index]

    def random_indices(self, n: int) -> list[int]:
        return torch.randperm(len(self.records))[:n].tolist()

    def mutate(self, index: int) -> int:
        record = self.records[index]
        link = list(record["link"])
        ops = list(record["ops"])

        if random.random() < self.mutate_link_prob:
            other_links = [l for l in self.nb_arch.link_list if l != link]
            link = list(random.choice(other_links))
        else:
            pos = random.randrange(len(ops))
            choices = [op for op in self.op_candidates if op != ops[pos]]
            ops[pos] = random.choice(choices)

        canonical_hash = int(self.nb_arch.Arch(list(link), list(ops)).valid_hash())
        return self.hash_to_index.get(canonical_hash, index)

def validate_architecture(
    record: dict[str, Any],
    gen_record: dict[str, Any],
    data: Data,
    split_ctx: bench.SplitContext,
    target_cfg: bench.TargetTrainConfig,
    trigger_cfg: bench.TriggerTrainConfig,
    eval_cfg: bench.EvalConfig,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
    seeds: tuple[int, ...],
) -> tuple[float, float]:

    primary_seed = seeds[0]
    bench.set_seed(primary_seed)
    primary = bench.build_target_model(record, data, hidden_dim, dropout).to(device)
    primary = bench.train_target_model(primary, data, split_ctx, target_cfg, device)

    generator = bench.build_trigger_generator(
        gen_record, data, hidden_dim, trigger_cfg, device
    ).to(device)
    with suppress_stdout():
        backdoor = bench.train_trigger_generator(
            primary, generator, data, split_ctx, target_cfg, trigger_cfg, eval_cfg, device
        )
    fixed_generator = deepcopy(backdoor.trojan).to(device).eval()
    eval_backdoor, eval_args = bench.make_eval_backdoor(
        fixed_generator, trigger_cfg, eval_cfg, device
    )

    asr_list, acc_list = [], []
    for seed in seeds:
        bench.set_seed(seed)
        model = bench.build_target_model(record, data, hidden_dim, dropout).to(device)
        model = bench.train_target_model(model, data, split_ctx, target_cfg, device)
        metrics = bench.evaluate_target(model, data, split_ctx, eval_backdoor, eval_args, device)
        asr_list.append(metrics["ASR"])
        acc_list.append(metrics["ACC"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return float(np.median(asr_list)), float(np.median(acc_list))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genetic + NTK-condition-number search for backdoor-vulnerable "
        "NAS-Bench-Graph architectures."
    )
    p.add_argument("--dataset", type=str, default="Cora")
    p.add_argument("--data-root", type=str, default=str(bench.EXAMPLES_DIR / "data"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output-dir", type=str, default="./result/nas_backdoor_graph")
    p.add_argument("--save-suffix", type=str, default="0")
    p.add_argument("--total-resume", action="store_true")
    p.add_argument("--use-proteins-space", action="store_true")

    p.add_argument("--pool-size", type=int, default=50)
    p.add_argument("--sample-size", type=int, default=10)
    p.add_argument("--total-epoch", type=int, default=1500)
    p.add_argument("--sample-freq", type=int, default=10)
    p.add_argument("--score-repeats", type=int, default=3,
                   help="re-init the target/generator this many times and take the median NTK score")
    p.add_argument("--mutate-link-prob", type=float, default=0.25,
                   help="probability that a mutation rewires the link pattern instead of an op")

    p.add_argument("--ntk-proxy-split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--ntk-proxy-batch-size", type=int, default=16)

    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--gen-link", type=int, nargs=2, default=[0, 0],
                   help="2-node trigger generator parent links, e.g. 0 0 or 0 1")
    p.add_argument("--gen-ops", type=str, nargs=2, default=["gcn", "gcn"],
                   help="2-node trigger generator operations")

    p.add_argument("--train-epochs", type=int, default=200)
    p.add_argument("--train-lr", type=float, default=0.01)
    p.add_argument("--train-wd", type=float, default=5e-4)

    p.add_argument("--trojan-epochs", type=int, default=200)
    p.add_argument("--trojan-lr", type=float, default=0.01)
    p.add_argument("--trojan-wd", type=float, default=5e-4)
    p.add_argument("--trigger-size", type=int, default=3)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--target-loss-weight", type=float, default=1.0)
    p.add_argument("--homo-loss-weight", type=float, default=100.0)
    p.add_argument("--homo-boost-thrd", type=float, default=0.5)
    p.add_argument("--thrd", type=float, default=0.5)
    p.add_argument("--debug", action="store_true")

    p.add_argument("--defense-mode", type=str, default="prune", choices=["prune", "isolate", "none"])
    p.add_argument("--prune-thr", type=float, default=0.1)

    p.add_argument("--seeds", type=str, default="666,777,888")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    device = torch.device(
        args.device
        if torch.cuda.is_available() or not args.device.startswith("cuda")
        else "cpu"
    )

    target_cfg = bench.TargetTrainConfig(args.train_epochs, args.train_lr, args.train_wd)
    trigger_cfg = bench.TriggerTrainConfig(
        epochs=args.trojan_epochs,
        lr=args.trojan_lr,
        weight_decay=args.trojan_wd,
        trigger_size=args.trigger_size,
        target_class=args.target_class,
        target_loss_weight=args.target_loss_weight,
        homo_loss_weight=args.homo_loss_weight,
        homo_boost_thrd=args.homo_boost_thrd,
        thrd=args.thrd,
        debug=args.debug,
    )
    eval_cfg = bench.EvalConfig(args.defense_mode, args.prune_thr)
    bd_args = bench.make_backdoor_args(target_cfg, trigger_cfg, eval_cfg, trojan_epochs=0)

    nb_arch = bench.load_nbgraph_arch_module()
    space = GraphArchSearchSpace(nb_arch, args.use_proteins_space, args.mutate_link_prob)
    print(f"NAS-Bench-Graph target search space: {len(space)} architectures")

    gen_record = {
        "generator_architecture_id": 0,
        "link": list(args.gen_link),
        "ops": list(args.gen_ops),
    }

    bench.set_seed(seeds[0])
    data = bench.load_backdoor_dataset(args.dataset, Path(args.data_root), device, trigger_cfg)
    split_ctx = bench.build_split_context(data, device)

    bench.set_seed(12345)
    proxy_nodes = bench.sample_proxy_nodes(data, args.ntk_proxy_split, args.ntk_proxy_batch_size).to(device)

    def get_score(arch_index: int) -> float:
        record = space.record(arch_index)
        scores: list[float] = []
        for _ in range(args.score_repeats):
            model = bench.build_target_model(record, data, args.hidden_dim, args.dropout).to(device)
            generator = bench.build_trigger_generator(
                gen_record, data, args.hidden_dim, trigger_cfg, device
            ).to(device)
            for p in model.parameters():
                p.requires_grad_(True)
            scores.append(
                ntk_score_once(model, generator, data, split_ctx, proxy_nodes, bd_args, device)
            )
            del model, generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return float(np.median(scores))

    out_dir = Path(args.output_dir) / bench.dataset_output_name(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"search_genetic_{args.save_suffix}.npz"

    pool_size = args.pool_size
    sample_size = args.sample_size
    epochs = args.total_epoch

    torch.manual_seed(int.from_bytes(os.urandom(4), "little"))

    score_result: list[float] = []
    index_result: list[int] = []
    asr: dict[int, float] = {}
    acc: dict[int, float] = {}

    if args.total_resume and file_path.is_file():
        prev = np.load(file_path, allow_pickle=True)
        ages = torch.as_tensor(prev["ages"], dtype=torch.int)
        pool = torch.as_tensor(prev["pool"], dtype=torch.long)
        scores = torch.as_tensor(prev["scores"], dtype=torch.float)
        _epoch = int(prev["_epoch"])
        score_result = prev["score"].tolist()
        index_result = prev["index"].tolist()
        best_element = int(pool[scores.argmin()])
        best_score = float(scores.min())
        asr = dict(prev["asr"].item())
        acc = dict(prev["acc"].item())
        print(f"Resumed from {file_path} at epoch {_epoch}")
    else:
        ages = torch.zeros(pool_size, dtype=torch.int)
        pool = torch.as_tensor(space.random_indices(pool_size), dtype=torch.long)
        scores = torch.tensor([get_score(int(el)) for el in pool], dtype=torch.float)
        _epoch = 0
        best_element = int(pool[scores.argmin()])
        best_score = float(scores.min())
        score_result.append(best_score)
        index_result.append(best_element)

    for i in range(_epoch, epochs):
        sample = torch.randperm(len(scores))[:sample_size]
        parent_pos = int(sample[int(scores[sample].argmin())])
        old_idx = int(ages.argmax())

        removed_element = int(pool[old_idx])
        removed_score = float(scores[old_idx])
        new_element = space.mutate(int(pool[parent_pos]))

        pool[old_idx] = new_element
        new_score = get_score(new_element)
        scores[old_idx] = new_score

        new_rec = space.record(new_element)
        removed_rec = space.record(removed_element)
        print(f"[{i + 1}/{epochs}]")
        print(f"    Add  score: {new_score:15.4f}  arch: id={new_rec['architecture_id']} "
              f"link={new_rec['link']} ops={new_rec['ops']}")
        print(f"    Del  score: {removed_score:15.4f}  arch: id={removed_rec['architecture_id']} "
              f"link={removed_rec['link']} ops={removed_rec['ops']}")

        score_result.append(new_score)
        index_result.append(new_element)
        if new_score < best_score:
            print(f"    Best updated!  previous: {best_score:10.3f}  new: {new_score:10.3f}")
            best_element = new_element
            best_score = new_score

        ages += 1
        ages[old_idx] = 0

        if i % args.sample_freq == 0:
            cand_asr, cand_acc = validate_architecture(
                space.record(new_element), gen_record, data, split_ctx,
                target_cfg, trigger_cfg, eval_cfg, args.hidden_dim, args.dropout, device, seeds,
            )
            asr[i] = cand_asr
            acc[i] = cand_acc
            print(f"    [validate] arch id={new_rec['architecture_id']} "
                  f"median ASR={cand_asr:.4f}  median ACC={cand_acc:.4f}")

    best_rec = space.record(best_element)
    print()
    print(f"best arch index (record id): {best_rec['architecture_id']}")
    print(f"best arch link={best_rec['link']} ops={best_rec['ops']}")
    print(f"best NTK condition-number score: {best_score}")

if __name__ == "__main__":
    main()
