import torch
from autogllight.utils import set_seed
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Actor, WikipediaNetwork, Flickr, Coauthor, Amazon, TUDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, erdos_renyi_graph
import numpy as np
from torch_geometric.utils import k_hop_subgraph

class GraphDataLoader:
    def __init__(
        self,
        device,
        dataset_name: str,
        root: str = '../../data',
        trigger_size: int = 3,
        trigger_prob: float = 0.5,
        vs_size: int = 40,
        attack_method: str = None,
        target_class: int = 0,
        split: bool = False
    ):
        self.device = device
        self.dataset_name = dataset_name
        self.root = root
        self.trigger_size = trigger_size
        self.trigger_prob = trigger_prob
        self.vs_size = vs_size
        self.attack_method = attack_method
        self.target_class = target_class
        self.split = split

    def load_data(self) -> Data:
        transform = T.Compose([T.NormalizeFeatures()])
        if self.dataset_name in ['Citeseer', 'Cora', 'Pubmed']:
            dataset = Planetoid(root=self.root, name=self.dataset_name, transform=transform)
        elif self.dataset_name == 'Actor':
            dataset = Actor(root='./data/', transform=transform)
        elif self.dataset_name in ['chameleon']:
            dataset = WikipediaNetwork(root=self.root, name=self.dataset_name, geom_gcn_preprocess=True, transform=transform)
        elif self.dataset_name == 'Flickr':
            dataset = Flickr(root='./data/Flickr/', transform=transform)
        elif self.dataset_name in ['CS', 'Physics']:
            dataset = Coauthor(root=self.root, name=self.dataset_name, transform=transform)

        elif self.dataset_name in ['Photo', 'Computers']:
            dataset = Amazon(root=self.root, name=self.dataset_name, transform=transform)

        elif self.dataset_name == 'proteins':
            dataset = TUDataset(root=self.root, name='PROTEINS', transform=transform)

        data = dataset[0].to(self.device)

        if self.split:
            data, _, _, _ = self._get_split(data)

        data = self._select_val_triggers(data)
        return data

    def _get_split(self, data: Data):
        perm = np.random.permutation(data.num_nodes)

        n_train = int(0.2 * data.num_nodes)
        n_val   = int(0.1 * data.num_nodes)
        n_test  = int(0.2 * data.num_nodes)

        idx_train = torch.tensor(sorted(perm[:n_train]), dtype=torch.long, device=self.device)
        idx_val   = torch.tensor(sorted(perm[n_train:n_train + n_val]), dtype=torch.long, device=self.device)
        idx_test  = torch.tensor(sorted(perm[n_train + n_val:n_train + n_val + n_test]), dtype=torch.long, device=self.device)

        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
        data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
        data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
        data.train_mask[idx_train] = True
        data.val_mask[idx_val]     = True
        data.test_mask[idx_test]   = True

        return data, idx_train, idx_val, idx_test

    def _get_split_class(
            self,
            data: Data,
            train_ratio: float = 0.2,
            val_ratio: float = 0.1,
            test_ratio: float = 0.2,
            device: str = "cpu"
    ) -> Data:

        y = data.y.cpu()
        num_nodes = data.num_nodes
        num_classes = int(y.max()) + 1

        n_train = int(train_ratio * num_nodes)
        n_val = int(val_ratio * num_nodes)

        per_cls_train = n_train // num_classes
        per_cls_val = n_val // num_classes

        min_count = min((y == c).sum().item() for c in range(num_classes))
        per_cls_train = min(per_cls_train, min_count // 2)
        per_cls_val = min(per_cls_val, min_count - per_cls_train)

        idx_train, idx_val = [], []

        for c in range(num_classes):
            cls_idx = torch.where(y == c)[0]
            cls_idx = cls_idx[torch.randperm(cls_idx.size(0))]
            idx_train.append(cls_idx[:per_cls_train])
            idx_val.append(cls_idx[per_cls_train:per_cls_train + per_cls_val])

        idx_train = torch.cat(idx_train).to(device)
        idx_val = torch.cat(idx_val).to(device)

        mask_all = torch.zeros(num_nodes, dtype=torch.bool)
        mask_all[idx_train] = True
        mask_all[idx_val] = True
        remaining = torch.arange(num_nodes)[~mask_all].to(device)
        n_test = int(test_ratio * num_nodes)
        idx_test = remaining[:n_test]

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        data.train_mask[idx_train] = True
        data.val_mask[idx_val] = True
        data.test_mask[idx_test] = True

        return data, data.train_mask, data.val_mask, data.test_mask

    def _select_val_triggers(self, data: Data) -> Data:

        clean_idx = data.val_mask.nonzero(as_tuple=False).flatten()

        data.clean_idx = clean_idx
        data.trigger_idx = clean_idx
        data.target_class = torch.tensor(self.target_class, device=self.device)
        return data

