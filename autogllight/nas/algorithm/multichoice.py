import logging

import torch
import torch.nn as nn

from tqdm import trange

from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from .base import BaseNAS
from ..space.nni import (
    replace_layer_choice,
    replace_input_choice,
    DartsLayerChoice,
    DartsInputChoice,
)

_logger = logging.getLogger(__name__)

class DartsBackdoor(BaseNAS):

    def __init__(
        self,
        num_epochs=30,
        workers=4,
        gradient_clip=5.0,
        model_lr=1e-3,
        model_wd=5e-4,
        arch_lr=1e-3,
        arch_wd=1e-3,
        gen_lr=1e-3,
        gen_wd=5e-4,
        device="auto",
        disable_progress=False,
        batch_search=False,
        num_parts=0,
        parts_per_batch=1,
    ):
        super().__init__(device=device)
        self.num_epochs = num_epochs
        self.workers = workers
        self.gradient_clip = gradient_clip
        self.model_lr = model_lr
        self.model_wd = model_wd
        self.arch_lr = arch_lr
        self.arch_wd = arch_wd
        self.gen_lr = gen_lr
        self.gen_wd = gen_wd
        self.model_optimizer = torch.optim.Adam
        self.arch_optimizer = torch.optim.Adam
        self.gen_optimizer = torch.optim.Adam
        self.disable_progress = disable_progress
        self.batch_search = batch_search
        self.num_parts = num_parts
        self.parts_per_batch = parts_per_batch

    def search(self, space, dataset, estimator, search_model_arch=True, search_gen_arch=True):
        nas_modules_model = []
        if search_model_arch:
            replace_layer_choice(space, DartsLayerChoice, nas_modules_model)
            replace_input_choice(space, DartsInputChoice, nas_modules_model)
        space = space.to(self.device)
        assert hasattr(estimator, 'trainer') or hasattr(estimator, 'trainer_obj'), "estimator 缺少 trainer 接口"
        trigger_gen = None
        if hasattr(estimator, 'trainer') and estimator.trainer is not None:
            trigger_gen = getattr(estimator.trainer, 'trigger_gen', None)
        if trigger_gen is None and hasattr(estimator, 'trainer_obj') and estimator.trainer_obj is not None:
            if hasattr(estimator.trainer_obj, 'trigger_gen'):
                trigger_gen = estimator.trainer_obj.trigger_gen
        assert trigger_gen is not None, "未找到触发器生成器 trigger_gen"
        nas_modules_gen = []
        if search_gen_arch and hasattr(trigger_gen, 'trojan') and hasattr(trigger_gen.trojan, 'mid_space'):
            replace_layer_choice(trigger_gen.trojan.mid_space, DartsLayerChoice, nas_modules_gen)
            replace_input_choice(trigger_gen.trojan.mid_space, DartsInputChoice, nas_modules_gen)
            trigger_gen.trojan.mid_space = trigger_gen.trojan.mid_space.to(self.device)
        ctrl_params_model = {}
        for _, m in nas_modules_model:
            if m.name in ctrl_params_model:
                assert m.alpha.size() == ctrl_params_model[m.name].size()
                m.alpha = ctrl_params_model[m.name]
            else:
                ctrl_params_model[m.name] = m.alpha

        ctrl_params_gen = {}
        for _, m in nas_modules_gen:
            if m.name in ctrl_params_gen:
                assert m.alpha.size() == ctrl_params_gen[m.name].size()
                m.alpha = ctrl_params_gen[m.name]
            else:
                ctrl_params_gen[m.name] = m.alpha

        arch_alpha_list = list(ctrl_params_model.values()) + list(ctrl_params_gen.values())
        arch_optim = (
            self.arch_optimizer(arch_alpha_list, lr=self.arch_lr, weight_decay=self.arch_wd)
            if len(arch_alpha_list) > 0 else None
        )
        model_optim = self.model_optimizer(space.parameters(), lr=self.model_lr, weight_decay=self.model_wd)
        gen_optim = self.gen_optimizer(trigger_gen.parameters(), lr=self.gen_lr, weight_decay=self.gen_wd)

        sub_graphs = None
        if self.batch_search and self.num_parts and self.num_parts > 0:
            sub_graphs = self._build_subgraphs(dataset)

        with trange(self.num_epochs, disable=self.disable_progress) as bar:
            for epoch in bar:
                if sub_graphs is not None:
                    metric, loss = self._train_one_epoch_batched(
                        epoch, space, sub_graphs, estimator, model_optim, gen_optim, arch_optim
                    )
                else:
                    metric, loss = self._train_one_epoch(
                        epoch, space, dataset, estimator, model_optim, gen_optim, arch_optim
                    )
                if loss is not None:
                    bar.set_postfix(loss=float(loss.item()), **metric)

        selection_model = self.export(nas_modules_model)
        selection_gen = self.export(nas_modules_gen)
        if search_model_arch:
            fixed_model = space.parse_model(selection_model)
        else:
            fixed_model = space
        return fixed_model, selection_gen

    def _train_one_epoch(self, epoch, model, dataset, estimator, model_optim, gen_optim, arch_optim):
        model.train()
        model_optim.zero_grad()
        metric_train, loss_train = estimator.infer(model, dataset=dataset, mask="train")
        loss_train.backward()
        if self.gradient_clip and self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        model_optim.step()
        gen_optim.zero_grad()
        metric_gen, loss_gen = estimator.infer(model, dataset=dataset, mask="gen")
        loss_gen.backward()
        gen_optim.step()
        metric_val, loss_val = estimator.infer(model, dataset=dataset, mask="val")
        if arch_optim is not None:
            arch_optim.zero_grad()
            loss_val.backward()
            arch_optim.step()
        return metric_val, loss_val

    def _build_subgraphs(self, dataset):

        x = dataset.x.detach().cpu()
        edge_index = dataset.edge_index.detach().cpu()
        y = dataset.y.detach().cpu()
        num_nodes = x.size(0)

        def _mask(name):
            m = getattr(dataset, name, None)
            if m is None:
                return torch.zeros(num_nodes, dtype=torch.bool)
            return m.detach().cpu().bool()

        base = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=_mask("train_mask"),
            val_mask=_mask("val_mask"),
            test_mask=_mask("test_mask"),
        )
        base.num_nodes = num_nodes

        num_parts = max(1, int(self.num_parts))
        parts_per_batch = max(1, int(self.parts_per_batch))

        try:
            from torch_geometric.loader import ClusterData, ClusterLoader

            cluster_data = ClusterData(base, num_parts=num_parts, log=False)
            loader = ClusterLoader(cluster_data, batch_size=parts_per_batch, shuffle=True)
            return list(loader)
        except Exception as e:
            _logger.warning(
                "ClusterData/ClusterLoader 不可用(%s)，回退到随机节点分块分批。", repr(e)
            )
            return self._random_partition_subgraphs(base, num_parts, parts_per_batch)

    @staticmethod
    def _random_partition_subgraphs(base, num_parts, parts_per_batch):

        num_nodes = base.x.size(0)
        perm = torch.randperm(num_nodes)
        part_splits = torch.chunk(perm, num_parts)
        sub_graphs = []
        for start in range(0, len(part_splits), parts_per_batch):
            grouped = part_splits[start:start + parts_per_batch]
            if len(grouped) == 0:
                continue
            node_idx = torch.cat(list(grouped))
            node_idx, _ = torch.sort(node_idx)
            sub_edge_index, _ = subgraph(
                node_idx, base.edge_index, relabel_nodes=True, num_nodes=num_nodes
            )
            sub = Data(
                x=base.x[node_idx],
                edge_index=sub_edge_index,
                y=base.y[node_idx],
                train_mask=base.train_mask[node_idx],
                val_mask=base.val_mask[node_idx],
                test_mask=base.test_mask[node_idx],
            )
            sub.num_nodes = node_idx.size(0)
            sub_graphs.append(sub)
        return sub_graphs

    def _train_one_epoch_batched(
        self, epoch, model, sub_graphs, estimator, model_optim, gen_optim, arch_optim
    ):

        model.train()
        last_metric, last_loss = {}, None
        for sub in sub_graphs:
            sub_data = sub.clone().to(self.device)
            val_idx = sub_data.val_mask.nonzero(as_tuple=False).flatten()
            train_idx = sub_data.train_mask.nonzero(as_tuple=False).flatten()
            sub_data.clean_idx = val_idx
            sub_data.trigger_idx = val_idx

            has_train = train_idx.numel() > 0
            has_val = val_idx.numel() > 0

            if has_train:
                model_optim.zero_grad()
                _, loss_train = estimator.infer(model, dataset=sub_data, mask="train")
                loss_train.backward()
                if self.gradient_clip and self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                model_optim.step()

            if has_val:
                gen_optim.zero_grad()
                _, loss_gen = estimator.infer(model, dataset=sub_data, mask="gen")
                loss_gen.backward()
                gen_optim.step()

                metric_val, loss_val = estimator.infer(model, dataset=sub_data, mask="val")
                if arch_optim is not None:
                    arch_optim.zero_grad()
                    loss_val.backward()
                    arch_optim.step()
                last_metric, last_loss = metric_val, loss_val
        return last_metric, last_loss

    @torch.no_grad()
    def export(self, nas_modules) -> dict:
        result = dict()
        for name, module in nas_modules:
            if name not in result:
                result[name] = module.export()
        return result
