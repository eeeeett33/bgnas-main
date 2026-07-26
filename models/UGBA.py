from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from models.GCN import GCN
from autogllight.nas.space.graph_nas_temp import GraphNasNodeClassificationSpace
from torch_geometric.data import Data

class GradWhere(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, thrd, device):

        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        self.mid_space = GraphNasNodeClassificationSpace(
            hidden_dim=nfeat,
            layer_number=2,
            dropout=0.0,
            input_dim=nfeat,
            output_dim=nfeat,
            con_ops=["concat"],
        )
        self.mid_space.instantiate()
        self.mid_space = self.mid_space.to(device)

        self.feat = nn.Linear(nfeat, nout * nfeat)
        self.edge = nn.Linear(nfeat, int(nout * (nout - 1) / 2))
        self.device = device

    def apply_selection(self, selection):

        self.mid_space = self.mid_space.parse_model(selection)
        self.mid_space = self.mid_space.to(self.device)
        return self

    def forward(self, input, thrd):

        GW = GradWhere.apply
        fake_data = Data(x=input.to(self.device), edge_index=torch.empty(2, 0, dtype=torch.long, device=self.device))
        h = self.mid_space(fake_data)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight

class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])

        loss = torch.relu(thrd - edge_sims).mean()
        return loss

import numpy as np
class Backdoor:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size)
        self.all_inject_edges = None
        self.first = True

    def get_trigger_index(self,trigger_size):
        edge_list = []
        edge_list.append([0,0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        return edge_index

    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index

    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd)

        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        if self.first == True:
            self.first = False
            self.all_inject_edges = deepcopy(trojan_edge)
        else:
            self.all_inject_edges =  torch.cat([self.all_inject_edges, trojan_edge],dim=1)
        return update_feat, update_edge_index, update_edge_weights

    def __getitem__(self, index):
        return self.all_inject_edges[index]

    def get_all_trojan_edge(self):
        return deepcopy(self.all_inject_edges)

    def fit_with_shadow(self, features, edge_index, edge_weight, labels, idx_train, idx_val, idx_attach, idx_unlabeled, target_class, shadow_model, train_iters=200, lr=0.01, weight_decay=5e-4, debug=False, fixed_trigger=None):
        args = self.args
        device = self.device
        self.target_class = target_class
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=device, dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight

        from torch_geometric.data import Data

        class _ShadowWrapper(nn.Module):
            def __init__(self, mdl):
                super().__init__()
                self.mdl = mdl.to(device)
            def forward(self, x, edge_index, edge_weight=None):
                data = Data(x=x, edge_index=edge_index)
                return self.mdl(data)

        self.shadow_model = _ShadowWrapper(shadow_model)

        if fixed_trigger is not None:
            self.trojan = fixed_trigger.to(device)
        else:
            self.trojan = GraphTrojanNet(device, features.shape[1], args.trigger_size, layernum=2).to(device)
        self.homo_loss = HomoLoss(self.args, self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=lr, weight_decay=weight_decay)
        for ep in range(train_iters):
            self.shadow_model.train()
            optimizer_shadow.zero_grad()
            out = self.shadow_model(features, edge_index, edge_weight)
            loss = F.nll_loss(out[idx_train], labels[idx_train])
            loss.backward()
            optimizer_shadow.step()
            if debug and ep % 20 == 0:
                with torch.no_grad():
                    self.shadow_model.eval()
                    pred = self.shadow_model(features, edge_index, edge_weight)
                    acc = utils.accuracy(pred[idx_train], labels[idx_train])
                print(f"[Shadow] epoch {ep}, loss {float(loss):.4f}, train acc {float(acc):.4f}")

        for p in self.shadow_model.parameters():
            p.requires_grad = False
        self.shadow_model.eval()

        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.retrain_lr, weight_decay=args.retrain_wd)

        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class
        self.poisoned_labels = self.labels

        loss_best = 1e9
        self.weights = deepcopy(self.trojan.state_dict())
        for i in range(args.trojan_epochs):
            self.trojan.train()
            optimizer_trigger.zero_grad()

            idx_outter = idx_val

            trojan_feat, trojan_weights = self.trojan(features[idx_outter], self.args.thrd)
            trojan_weights = torch.cat([torch.ones([len(idx_outter), 1], dtype=torch.float, device=device), trojan_weights], dim=1)
            trojan_weights = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1, features.shape[1]])

            trojan_edge = self.get_trojan_edge(len(features), idx_outter, self.args.trigger_size).to(device)
            update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
            update_feat = torch.cat([features, trojan_feat])
            update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

            with torch.no_grad():
                self.shadow_model.eval()
            output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)

            labels_outter = labels.clone()
            labels_outter[idx_outter] = args.target_class
            loss_target = self.args.target_loss_weight * F.nll_loss(output[torch.cat([idx_train, idx_outter])],
                                                                   labels_outter[torch.cat([idx_train, idx_outter])])

            loss_homo = 0.0
            if self.args.homo_loss_weight > 0:
                loss_homo = self.homo_loss(trojan_edge[:, :int(trojan_edge.shape[1] / 2)],
                                           trojan_weights,
                                           update_feat,
                                           self.args.homo_boost_thrd)
            loss_outter = loss_target + self.args.homo_loss_weight * loss_homo
            loss_outter.backward()
            nn.utils.clip_grad_norm_(self.trojan.parameters(), max_norm=1.0)
            optimizer_trigger.step()

            if float(loss_outter) < loss_best:
                loss_best = float(loss_outter)
                self.weights = deepcopy(self.trojan.state_dict())
            if debug and i % 1 == 0:
                print(f"[Trigger] epoch {i}, loss_target {float(loss_target):.5f}, homo {float(loss_homo) if isinstance(loss_homo, torch.Tensor) else 0.0:.5f}")

                tt = output.argmax(dim=1)
                print('ASR: ', sum(tt[idx_outter].to('cpu').numpy() == 0) / len(idx_outter), ' pred: ', set(tt[idx_outter].to('cpu').numpy()))

        self.trojan.load_state_dict(self.weights)
        self.trojan.eval()

    def get_poisoned(self):

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels
        poison_labels[self.idx_attach] = self.target_class
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

