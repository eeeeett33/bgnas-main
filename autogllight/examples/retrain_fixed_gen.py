#!/usr/bin/env python3
import argparse
import os
import math

import numpy as np
import torch
import torch.nn as nn

from autogllight.utils import set_seed
from utils.data_atk import GraphDataLoader
from autogllight.examples.training_pipeline import run_pipeline_like_edgepruning

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Flickr', choices=['Citeseer', 'Flickr', 'Cora', 'Pubmed', 'Actor', 'chameleon'])
    parser.add_argument('--seed_load', type=int, default=1012,  help='Seed used when saving fixed model/generator')
    parser.add_argument('--seed_retrain', type=int, default=0, help='New seed used to re-init and retrain the target model')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--trigger_size', type=int, default=3)
    parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--vs_size', type=int, default=40)

    parser.add_argument('--retrain_epochs', type=int, default=200)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--retrain_lr', type=float, default=0.01)
    parser.add_argument('--retrain_wd', type=float, default=5e-4)
    parser.add_argument('--trojan_epochs', type=int, default=200, help='Number of epochs to fine-tune generator')
    parser.add_argument('--freeze_trigger', action='store_true', default=True, help='If set, do not fine-tune generator')

    parser.add_argument('--use_vs_number', action='store_true', default=True)
    parser.add_argument('--vs_ratio', type=float, default=0.0)
    parser.add_argument('--vs_number', type=int, default=40)
    parser.add_argument('--selection_method', type=str, default='cluster_degree',
                        choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'])
    parser.add_argument('--defense_mode', type=str, default='none', choices=['prune', 'isolate', 'none'])
    parser.add_argument('--prune_thr', type=float, default=0.1)
    parser.add_argument('--target_loss_weight', type=float, default=1.0)
    parser.add_argument('--homo_loss_weight', type=float, default=100.0)
    parser.add_argument('--homo_boost_thrd', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dis_weight', type=float, default=1.0)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--thrd', type=float, default=0.5)
    return parser.parse_args()

def _reset_module_parameters(module: nn.Module):

    for m in module.modules():
        if hasattr(m, 'reset_parameters') and callable(getattr(m, 'reset_parameters')):
            try:
                m.reset_parameters()
                continue
            except Exception:
                pass
        if isinstance(m, (nn.Linear,)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

def load_saved_components(dataset: str, seed_load: int, device: torch.device):
    base_dir = os.path.dirname(__file__)
    save_dir = os.path.join(base_dir, 'saved_models', dataset, f'seed_{seed_load}')
    model_path = os.path.join(save_dir, 'fixed_model_obj.pt')
    trigger_path = os.path.join(save_dir, 'fixed_trigger_obj.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')
    if not os.path.exists(trigger_path):
        raise FileNotFoundError(f'Trigger not found: {trigger_path}')

    fixed_model_obj = torch.load(model_path, map_location=device)
    target_model = getattr(fixed_model_obj, '_model', fixed_model_obj)

    fixed_trigger = torch.load(trigger_path, map_location=device)

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

def retrain_model(args, target_model=None, fixed_trigger=None):

    device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device))

    seed_load = int(getattr(args, 'seed_load', getattr(args, 'seed', 0)))
    seed_retrain = int(getattr(args, 'seed_retrain', getattr(args, 'seed', seed_load)))
    freeze_trigger = bool(getattr(args, 'freeze_trigger', True))

    set_seed(seed_load)
    if target_model is None or fixed_trigger is None:
        target_model, fixed_trigger = load_saved_components(args.dataset, seed_load, device)
    target_model = target_model.to(device)
    try:
        for p in target_model.parameters():
            p.requires_grad = True
    except Exception:
        pass
    target_model.train()

    data = build_dataset(args, device)

    set_seed(seed_retrain)
    _reset_module_parameters(target_model)

    fixed_trigger = fixed_trigger.to(device)
    if freeze_trigger:
        args.trojan_epochs = 0
    for p in fixed_trigger.parameters():
        p.requires_grad = (not freeze_trigger)
    if not freeze_trigger:
        fixed_trigger.train()
    else:
        fixed_trigger.eval()
    result = run_pipeline_like_edgepruning(args, data, device, target_model, fixed_trigger=fixed_trigger)
    return result

if __name__ == '__main__':
    rs = np.random.RandomState(15)
    seeds1 = rs.randint(2000, size=50)
    asr_list = []
    for seed1 in seeds1:
        args = get_args()
        args.seed_retrain = int(seed1)
        r = retrain_model(args)
        asr_list.append(float(r.get('ASR', 0.0)))
    asr_mean = float(np.mean(asr_list)) if asr_list else 0.0
    print("Mean ASR: ", asr_mean)

