#!/usr/bin/env python3
import argparse
import copy
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

from autogllight.nas.space.graph_nas import GraphBenchmarkingSpace
from autogllight.nas.algorithm.darts_backdoor import DartsBackdoor
from autogllight.nas.estimator.base import BaseEstimator
from autogllight.utils import set_seed
from utils.data_atk import GraphDataLoader
from models.UGBA import GraphTrojanNet, HomoLoss
from autogllight.examples.training_pipeline import run_pipeline_like_edgepruning, analyze_architecture
from retrain_fixed_gen import retrain_model

import numpy as np
from torch_geometric.utils import k_hop_subgraph
import os

class UGBACompatTriggerGen(torch.nn.Module):
    def __init__(self, device, nfeat, trigger_size=3, target_class=0, thrd=0.5):
        super().__init__()
        self.device = device
        self.nfeat = nfeat
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.thrd = thrd
        self.trojan = GraphTrojanNet(self.device, nfeat, trigger_size, layernum=2).to(device)

    def _get_trigger_index(self, trigger_size):
        edge_list = []
        edge_list.append([0, 0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j, k])
        edge_index = torch.tensor(edge_list, device=self.device).long().t()
        return edge_index

    def _get_trojan_edge(self, start, idx_attach, trigger_size):
        edge_list = []
        base = self._get_trigger_index(trigger_size)
        for idx in idx_attach:
            edges = base.clone()
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start
            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        return torch.stack([row, col])

    def build_poisoned_val_graph(self, data: Data, detach_outputs: bool):
        device = self.device
        x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
        n = x.size(0)
        idx = data.trigger_idx.to(device)

        host_feat = x[idx]
        feat, weights = self.trojan(host_feat, self.thrd)
        feat = feat.view(-1, self.nfeat)
        if detach_outputs:
            feat = feat.detach()

        trojan_edge = self._get_trojan_edge(n, idx, self.trigger_size)
        x_poison = torch.cat([x, feat], dim=0)
        edge_index_poison = torch.cat([edge_index, trojan_edge], dim=1)
        y_poison = y.clone()
        y_poison[idx] = self.target_class
        return x_poison, edge_index_poison, y_poison

class MiniEstimator(BaseEstimator):
    def __init__(self, trainer, args=None, loss_f=None, evaluation=None):
        super().__init__(loss_f, evaluation)
        self.trainer = trainer
        self.args = args

    def infer(self, model: GraphBenchmarkingSpace, dataset, mask="train"):
        model = model.to(self.trainer.device)
        data: Data = dataset.to(self.trainer.device)

        if mask == "train":
            model.train()
            pred = model(data)
            loss = F.nll_loss(pred[data.train_mask], data.y[data.train_mask])
            with torch.no_grad():
                clean_acc = (pred[data.val_mask].argmax(1) == data.y[data.val_mask]).float().mean().item()
            return {"clean_acc": clean_acc}, loss

        assert self.trainer.trigger_gen is not None
        detach_outputs = False if mask in ("gen", "val") else True
        data_c = copy.deepcopy(data)
        x_p, ei_p, y_p = self.trainer.trigger_gen.build_poisoned_val_graph(data, detach_outputs)
        data_p = Data(x=x_p, edge_index=ei_p, y=y_p,
                      train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                      clean_idx=data.clean_idx, trigger_idx=data.trigger_idx)
        pred_p = model(data_p)
        pred_c = model(data_c)
        pred_clean = pred_c[data.clean_idx]
        y_clean = data_c.y[data.clean_idx]
        pred_trig = pred_p[data.trigger_idx]
        y_trig_poison = data_p.y[data.trigger_idx]

        loss = F.nll_loss(pred_clean, y_clean) + self.args.target_loss_weight * F.nll_loss(pred_trig, y_trig_poison)
        with torch.no_grad():
            clean_acc = (pred_clean.argmax(1) == y_clean).float().mean().item()
            asr = (pred_trig.argmax(1) == y_trig_poison).float().mean().item()
        return {"clean_acc": clean_acc, "ASR": asr}, loss

class TrainerBox:
    def __init__(self, device, trigger_gen,
                 hfir_radius: int = None, hfir_proj_q: int = 0, hfir_edge_sample_m: int = 0):
        self.device = device
        self.trigger_gen = trigger_gen

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
    data = loader.load_data()
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Photo')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=150, help='DARTS 搜索轮数')
    parser.add_argument('--model_lr', type=float, default=1e-3)
    parser.add_argument('--model_wd', type=float, default=5e-4)
    parser.add_argument('--arch_lr', type=float, default=1e-3)
    parser.add_argument('--arch_wd', type=float, default=1e-3)
    parser.add_argument('--gen_lr', type=float, default=1e-3)
    parser.add_argument('--gen_wd', type=float, default=5e-4)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--layer_number', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--trigger_size', type=int, default=3)
    parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--vs_size', type=int, default=40)
    parser.add_argument('--train_lr', type=float, default=0.01, help='干净阶段 shadow 模型学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--use_vs_number', action='store_true', default=True)
    parser.add_argument('--vs_ratio', type=float, default=0.0)
    parser.add_argument('--vs_number', type=int, default=40)
    parser.add_argument('--selection_method', type=str, default='cluster_degree',
                        choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'])
    parser.add_argument('--defense_mode', type=str, default='prune', choices=['prune', 'isolate', 'none'])
    parser.add_argument('--prune_thr', type=float, default=0)
    parser.add_argument('--target_loss_weight', type=float, default=1.0)
    parser.add_argument('--homo_loss_weight', type=float, default=100.0)
    parser.add_argument('--homo_boost_thrd', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dis_weight', type=float, default=1)
    parser.add_argument('--retrain_epochs', type=int, default=200)
    parser.add_argument('--retrain_lr', type=float, default=0.01)
    parser.add_argument('--retrain_wd', type=float, default=5e-4)
    parser.add_argument('--trojan_epochs', type=int, default=200)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--thrd', type=float, default=0.5)
    parser.add_argument('--freeze_trigger', action='store_true', default=True, help='If set, do not fine-tune generator')
    parser.add_argument('--seed_load', type=int, default=0, help='Seed used when saving fixed model/generator')
    parser.add_argument('--seed_retrain', type=int, default=888, help='New seed used to re-init and retrain the target model')

    args = parser.parse_args()
    return args

def main(args, seed):
    device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device))

    data = build_dataset(args, device)

    space = GraphBenchmarkingSpace(
        hidden_dim=args.hidden_dim,
        layer_number=args.layer_number,
        dropout=args.dropout,
        input_dim=data.num_node_features,
        output_dim=int(data.y.max().item() + 1),
    )
    space.instantiate()

    trigger_gen = UGBACompatTriggerGen(device, data.num_node_features, args.trigger_size, args.target_class)
    trainer = TrainerBox(
        device=device,
        trigger_gen=trigger_gen,
    )

    estimator = MiniEstimator(trainer, args=args)

    algo = DartsBackdoor(
        num_epochs=args.epochs,
        gradient_clip=args.grad_clip,
        model_lr=args.model_lr,
        model_wd=args.model_wd,
        arch_lr=args.arch_lr,
        arch_wd=args.arch_wd,
        gen_lr=args.gen_lr,
        gen_wd=args.gen_wd,
        device=device,
    )

    fixed_model, gen_selection = algo.search(space, data, estimator)
    fixed_trigger = None
    if hasattr(trigger_gen, 'trojan') and hasattr(trigger_gen.trojan, 'apply_selection'):
        trigger_gen.trojan.apply_selection(gen_selection)
        fixed_trigger = trigger_gen.trojan
    arch_summary = analyze_architecture(fixed_model, fixed_trigger)
    print('arch_summary:', arch_summary)
    result = run_pipeline_like_edgepruning(args, data, device, fixed_model._model, fixed_trigger=fixed_trigger)

    rs = np.random.RandomState(15)
    seeds1 = rs.randint(2000, size=50)
    args.seed_load = int(seed)
    retrain_results = []
    asr_list = []
    for seed1 in seeds1:
        fixed_model1 = copy.deepcopy(fixed_model._model)
        fixed_trigger1 = copy.deepcopy(fixed_trigger)
        args.seed_retrain = int(seed1)
        r = retrain_model(args, target_model=fixed_model1, fixed_trigger=fixed_trigger1)
        retrain_results.append((int(seed1), r))
        asr_list.append(float(r.get('ASR', 0.0)))

    asr_mean = float(np.mean(asr_list)) if asr_list else 0.0
    result['ASR_mean'] = asr_mean

    return result, fixed_model._model, fixed_trigger

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    result, fixed_model, fixed_trigger = main(args, args.seed)
    print(result)

    save_dir = os.path.join('saved_models', args.dataset, f'seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)

    fixed_model_obj_path = os.path.join(save_dir, 'fixed_model_obj.pt')
    torch.save(fixed_model, fixed_model_obj_path)

    if fixed_trigger is not None:
        fixed_trigger_obj_path = os.path.join(save_dir, 'fixed_trigger_obj.pt')
        torch.save(fixed_trigger, fixed_trigger_obj_path)

           
