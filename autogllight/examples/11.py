#!/usr/bin/env python3

import argparse
import json
import os

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from autogllight.utils import set_seed
from utils.data_atk import GraphDataLoader
from utils import subgraph
from models.UGBA import Backdoor
from help_funcs import prune_unrelated_edge

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, default='saved_models',
                        help='phase1 模型根目录（相对 examples 目录），对应 retrain_fixed_gen1.py 的保存路径')
    parser.add_argument('--target_class_root', type=str,
                        default='/mnt/HDD-data/saved_models_target_class',
                        help='不同 target_class 下保存模型的根目录')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='asr_c_results')

    parser.add_argument('--trigger_size', type=int, default=3)
    parser.add_argument('--target_class', type=int, default=0,
                        help='仅单条评测时使用；批量扫描 target_class 目录时会自动覆盖')
    parser.add_argument('--vs_size', type=int, default=40)
    parser.add_argument('--thrd', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--seed_load', type=int, default=None)
    return parser.parse_args()

def _ensure_trigger_compat(fixed_trigger):

    if not hasattr(fixed_trigger, 'searchable'):
        import torch.nn as nn
        mid = getattr(fixed_trigger, 'mid_space', None)
        fixed_trigger.searchable = not isinstance(mid, nn.Sequential)
    return fixed_trigger

def load_saved_components(save_dir: str, device: torch.device):

    model_path = os.path.join(save_dir, 'fixed_model_obj.pt')
    trigger_path = os.path.join(save_dir, 'fixed_trigger_obj.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    if not os.path.exists(trigger_path):
        raise FileNotFoundError(f'Trigger not found: {trigger_path}')

    fixed_model_obj = torch.load(model_path, map_location=device, weights_only=False)
    target_model = getattr(fixed_model_obj, '_model', fixed_model_obj)
    fixed_trigger = torch.load(trigger_path, map_location=device, weights_only=False)
    fixed_trigger = _ensure_trigger_compat(fixed_trigger)
    return target_model, fixed_trigger

def build_dataset(args, device):
    loader = GraphDataLoader(
        device=device,
        dataset_name=args.dataset,
        root="./data",
        trigger_size=args.trigger_size,
        vs_size=args.vs_size,
        target_class=args.target_class,
        split=True,
    )
    return loader.load_data()

@torch.no_grad()
def evaluate_asr_c(args, data, device, model, fixed_trigger):

    data.edge_index = to_undirected(data.edge_index)
    train_edge_index, _, edge_mask = subgraph(
        torch.bitwise_not(data.test_mask), data.edge_index, relabel_nodes=False
    )
    mask_edge_index = data.edge_index[:, torch.bitwise_not(edge_mask)]

    idx_test = data.test_mask.nonzero(as_tuple=False).flatten()
    half = int(len(idx_test) / 2)
    idx_clean_test = idx_test[:half]
    idx_atk = idx_test[half:]

    model = model.to(device)
    model.eval()

    backdoor = Backdoor(args, device)
    backdoor.trojan = fixed_trigger.to(device)
    backdoor.trojan.eval()

    induct_edge_index = torch.cat([train_edge_index, mask_edge_index], dim=1)
    induct_edge_weights = torch.ones([induct_edge_index.shape[1]], dtype=torch.float, device=device)

    data_induct_clean = Data(x=data.x, edge_index=induct_edge_index, y=data.y)
    out_clean = model(data_induct_clean)
    ca = (out_clean[idx_clean_test].argmax(1) == data.y[idx_clean_test]).float().mean().item()

    induct_x, induct_edge_index2, induct_edge_weights2 = backdoor.inject_trigger(
        idx_atk, data.x, induct_edge_index, induct_edge_weights, device
    )
    if args.defense_mode in ['prune', 'isolate']:
        induct_edge_index2, induct_edge_weights2 = prune_unrelated_edge(
            args, induct_edge_index2, induct_edge_weights2, induct_x, device
        )

    data_overall = Data(x=induct_x, edge_index=induct_edge_index2)
    out_overall = model(data_overall)
    pred_atk = out_overall[idx_atk].argmax(1)
    y_atk = data.y[idx_atk]

    asr = (pred_atk == args.target_class).float().mean().item()
    flip_mask = y_atk != args.target_class
    if int(flip_mask.sum().item()) == 0:
        asr_c = float('nan')
    else:
        asr_c = (pred_atk[flip_mask] == args.target_class).float().mean().item()

    return {
        'CA': float(ca),
        'ASR': float(asr),
        'ASR-C': float(asr_c),
    }

def _append_record(out_path, record):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def _iter_seed_dirs(root_dir, dataset=None, seed_load=None):
    if not os.path.isdir(root_dir):
        return
    datasets = [dataset] if dataset else sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    for ds in datasets:
        ds_dir = os.path.join(root_dir, ds)
        for name in sorted(os.listdir(ds_dir)):
            if not name.startswith('seed_'):
                continue
            seed = int(name.split('_', 1)[1])
            if seed_load is not None and seed != seed_load:
                continue
            save_dir = os.path.join(ds_dir, name)
            if os.path.exists(os.path.join(save_dir, 'fixed_model_obj.pt')):
                yield ds, seed, save_dir

def _iter_target_class_seed_dirs(root_dir, dataset=None, seed_load=None):
    if not os.path.isdir(root_dir):
        return
    datasets = [dataset] if dataset else sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    for ds in datasets:
        ds_dir = os.path.join(root_dir, ds)
        for tc_name in sorted(os.listdir(ds_dir), key=lambda x: int(x) if x.isdigit() else x):
            tc_dir = os.path.join(ds_dir, tc_name)
            if not os.path.isdir(tc_dir):
                continue
            try:
                target_class = int(tc_name)
            except ValueError:
                continue
            for name in sorted(os.listdir(tc_dir)):
                if not name.startswith('seed_'):
                    continue
                seed = int(name.split('_', 1)[1])
                if seed_load is not None and seed != seed_load:
                    continue
                save_dir = os.path.join(tc_dir, name)
                if os.path.exists(os.path.join(save_dir, 'fixed_model_obj.pt')):
                    yield ds, target_class, seed, save_dir

def eval_phase1(args, device, examples_dir, dataset_filter=None, seed_filter=None):
    root_dir = args.models_dir
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(examples_dir, root_dir)
    out_path = os.path.join(examples_dir, args.out_dir, 'saved_models_4_asr_c.jsonl')

    print(f'[phase1] root={root_dir}')
    for dataset, seed, save_dir in _iter_seed_dirs(root_dir, dataset_filter, seed_filter):
        args.dataset = dataset
        args.target_class = 0
        try:
            set_seed(seed)
            target_model, fixed_trigger = load_saved_components(save_dir, device)
            data = build_dataset(args, device)
            metrics = evaluate_asr_c(args, data, device, target_model, fixed_trigger)
            record = {
                'dataset': dataset,
                'seed': seed,
                'target_class': args.target_class,
                **metrics,
            }
            _append_record(out_path, record)
            print(f'  {dataset} seed={seed} | CA={metrics["CA"]:.4f} ASR={metrics["ASR"]:.4f} ASR-C={metrics["ASR-C"]:.4f}')
        except Exception as e:
            print(f'  [FAIL] {dataset} seed={seed}: {e}')

def eval_target_class(args, device, examples_dir, dataset_filter=None, seed_filter=None):
    root_dir = args.target_class_root
    out_path = os.path.join(examples_dir, args.out_dir, 'saved_models_target_class_asr_c.jsonl')

    print(f'[target_class] root={root_dir}')
    for dataset, target_class, seed, save_dir in _iter_target_class_seed_dirs(
        root_dir, dataset_filter, seed_filter
    ):
        args.dataset = dataset
        args.target_class = target_class
        try:
            set_seed(seed)
            target_model, fixed_trigger = load_saved_components(save_dir, device)
            data = build_dataset(args, device)
            metrics = evaluate_asr_c(args, data, device, target_model, fixed_trigger)
            record = {
                'dataset': dataset,
                'seed': seed,
                'target_class': target_class,
                **metrics,
            }
            _append_record(out_path, record)
            print(
                f'  {dataset} target={target_class} seed={seed} | '
                f'CA={metrics["CA"]:.4f} ASR={metrics["ASR"]:.4f} ASR-C={metrics["ASR-C"]:.4f}'
            )
        except Exception as e:
            print(f'  [FAIL] {dataset} target={target_class} seed={seed}: {e}')

def main():
    args = get_args()
    device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device))
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_filter = args.dataset
    seed_filter = args.seed_load

    if args.phase in ('phase1', 'all'):
        eval_phase1(args, device, examples_dir, dataset_filter, seed_filter)
    if args.phase in ('target_class', 'all'):
        eval_target_class(args, device, examples_dir, dataset_filter, seed_filter)

if __name__ == '__main__':
    main()
