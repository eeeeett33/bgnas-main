import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj,dense_to_sparse
import torch
import scipy.sparse as sp
from torch_geometric.utils import to_undirected

def edge_sim_analysis(edge_index, features):
    sims = []
    for (u,v) in edge_index:
        sims.append(float(F.cosine_similarity(features[u].unsqueeze(0),features[v].unsqueeze(0))))
    sims = np.array(sims)
    return sims

def prune_unrelated_edge(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    updated_edge_index = edge_index[:,edge_sims>args.prune_thr]
    updated_edge_weights = edge_weights[edge_sims>args.prune_thr]
    return updated_edge_index,updated_edge_weights

def edge_distribution(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])

    return edge_sims

def prune_unrelated_edge_isolated(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    dissim_edges_index = np.where(edge_sims.cpu()<=args.prune_thr)[0]
    edge_weights[dissim_edges_index] = 0
    dissim_edges = edge_index[:,dissim_edges_index]
    dissim_nodes = torch.cat([dissim_edges[0],dissim_edges[1]]).tolist()
    dissim_nodes = list(set(dissim_nodes))
    updated_edge_index = edge_index[:,edge_weights>0.0]
    updated_edge_weights = edge_weights[edge_weights>0.0]
    return updated_edge_index,updated_edge_weights,dissim_nodes

def is_undirected(edge_index):
    edge_index_2=edge_index.flip(0)
    sorted_indices = torch.argsort(edge_index_2[0])
    edge_index_2 = edge_index_2[:, sorted_indices]

    print(edge_index)
    print(edge_index_2)

def to_directed(edge_index):
    mask = edge_index[0]<=edge_index[1]
    unique_edges = edge_index[ : , mask]
    return unique_edges

def get_rand_edge_dropping(args,edge_index,edge_weights,x,device):
    edge_index_1 = edge_index[:,edge_weights>0.0].to(device)
    edge_weights_1 = edge_weights[edge_weights>0.0].to(device)
    unique_edges = to_directed(edge_index_1)
    E = unique_edges.shape[1]
    retain_edge_num = int(E * (1-args.dropping_rate))
    perm = np.random.permutation(E)
    perm = perm[:retain_edge_num]
    unique_edges = unique_edges[:, perm]

    edge_index_new = to_undirected(unique_edges)

    edge_weights_new = edge_weights_1[:edge_index_new.shape[1]]
    return edge_index_new, edge_weights_new

def select_target_nodes(args,seed,model,features,edge_index,edge_weights,labels,idx_val,idx_test):
    test_ca,test_correct_index = model.test_with_correct_nodes(features,edge_index,edge_weights,labels,idx_test)
    test_correct_index = test_correct_index.tolist()
    test_correct_nodes = idx_test[test_correct_index].tolist()
    target_class_nodes_test = [int(nid) for nid in idx_test
            if labels[nid]==args.target_class]
    idx_val,idx_test = idx_val.tolist(),idx_test.tolist()
    rs = np.random.RandomState(seed)
    cand_atk_test_nodes = list(set(test_correct_nodes) - set(target_class_nodes_test))
    atk_test_nodes = rs.choice(cand_atk_test_nodes, args.target_test_nodes_num)
    cand_clean_test_nodes = list(set(idx_test) - set(atk_test_nodes))
    clean_test_nodes = rs.choice(cand_clean_test_nodes, args.clean_test_nodes_num)
    N = features.shape[0]
    cand_poi_train_nodes = list(set(idx_val)-set(atk_test_nodes)-set(clean_test_nodes))
    poison_nodes_num = int(N * args.vs_ratio)
    poi_train_nodes = rs.choice(cand_poi_train_nodes, poison_nodes_num)

    return atk_test_nodes, clean_test_nodes,poi_train_nodes

def normalize(mx):

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()
