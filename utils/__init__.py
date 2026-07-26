
from .data_atk import GraphDataLoader

import torch
import numpy as np

def tensor2onehot(labels):

    labels = labels.long()
    eye = torch.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx.to(labels.device)

def accuracy(output, labels):

    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def idx_to_mask(indices, n):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask
import scipy.sparse as sp
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
def subgraph(subset,edge_index, edge_attr = None, relabel_nodes: bool = False):

    device = edge_index.device

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    return edge_index, edge_attr, edge_mask

def get_split(args,data, device):
    rs = np.random.RandomState(10)
    perm = rs.permutation(data.num_nodes)
    train_number = int(0.2*len(perm))
    idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True

    val_number = int(0.1*len(perm))
    idx_val = torch.tensor(sorted(perm[train_number:train_number+val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True

    test_number = int(0.2*len(perm))
    idx_test = torch.tensor(sorted(perm[train_number+val_number:train_number+val_number+test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[:int(len(idx_test)/2)]
    idx_atk = idx_test[int(len(idx_test)/2):]

    return data, idx_train, idx_val, idx_clean_test, idx_atk
