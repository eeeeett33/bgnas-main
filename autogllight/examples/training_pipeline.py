import torch
import numpy as np
from copy import deepcopy
from torch_geometric.data import Data
from models.UGBA import Backdoor
from models.construct import model_construct
from help_funcs import prune_unrelated_edge, prune_unrelated_edge_isolated
from utils import subgraph
from torch_geometric.utils import to_undirected

def run_pipeline_like_edgepruning(args, data, device, searched_model, fixed_trigger=None):

    data.edge_index = to_undirected(data.edge_index)
    train_edge_index, _, edge_mask = subgraph(torch.bitwise_not(data.test_mask), data.edge_index, relabel_nodes=False)
    mask_edge_index = data.edge_index[:, torch.bitwise_not(edge_mask)]

    idx_train = data.train_mask.nonzero(as_tuple=False).flatten()
    idx_val = data.val_mask.nonzero(as_tuple=False).flatten()
    idx_test = data.test_mask.nonzero(as_tuple=False).flatten()

    searched_model = searched_model.to(device)
    searched_model.train()

    half = int(len(idx_test) / 2)
    idx_clean_test = idx_test[:half]
    idx_atk = idx_test[half:]

    unlabeled_idx = (torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)).nonzero().flatten()
    if args.use_vs_number:
        size = args.vs_number
    else:
        size = int((len(data.test_mask) - data.test_mask.sum()) * args.vs_ratio)
    assert size > 0
    idx_attach = []
    features = data.x
    labels = data.y
    model = Backdoor(args, device)
    model.fit_with_shadow(
        features, train_edge_index, None, labels,
        idx_train, idx_val, idx_attach, unlabeled_idx, args.target_class,
        shadow_model=searched_model, train_iters=args.retrain_epochs,
        lr=args.train_lr, weight_decay=args.weight_decay, debug=args.debug,
        fixed_trigger=fixed_trigger,
    )

    searched_model = searched_model.to(device)
    searched_model.eval()

    induct_edge_index = torch.cat([train_edge_index, mask_edge_index], dim=1)
    induct_edge_weights = torch.ones([induct_edge_index.shape[1]], dtype=torch.float, device=device)

    with torch.no_grad():
        data_induct_clean = Data(x=data.x, edge_index=induct_edge_index, y=data.y)
        out_clean = searched_model(data_induct_clean)
        clean_acc = (out_clean[idx_clean_test].argmax(1) == data.y[idx_clean_test]).float().mean().item()

    eval_mode = getattr(args, 'evaluate_mode', 'overall')
    eval_mode = '1'
    if eval_mode == '1by1':
        from torch_geometric.utils import k_hop_subgraph
        overall_induct_edge_index = induct_edge_index.clone()
        overall_induct_edge_weights = induct_edge_weights.clone()
        _asr = 0.0
        flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
        _flip_asr = 0.0
        for idx in idx_atk:
            idx_int = int(idx)
            sub_nodeset, sub_edge_index, sub_mapping, _ = k_hop_subgraph(node_idx=[idx_int], num_hops=2, edge_index=overall_induct_edge_index, relabel_nodes=True)
            relabeled_node_idx = sub_mapping
            sub_induct_edge_weights = torch.ones([sub_edge_index.shape[1]], device=device)
            induct_x, induct_edge_index2, induct_edge_weights2 = model.inject_trigger(relabeled_node_idx, data.x[sub_nodeset], sub_edge_index, sub_induct_edge_weights, device)
            if args.defense_mode in ['prune', 'isolate']:
                induct_edge_index2, induct_edge_weights2 = prune_unrelated_edge(args, induct_edge_index2, induct_edge_weights2, induct_x, device, False)
            with torch.no_grad():
                data_sub = Data(x=induct_x, edge_index=induct_edge_index2)
                out_sub = searched_model(data_sub)
            rate = (out_sub.argmax(dim=1)[relabeled_node_idx] == args.target_class).float().mean()
            _asr += float(rate)
            if data.y[idx] != args.target_class:
                _flip_asr += float(rate)
        asr = _asr / max(1, idx_atk.shape[0])
    else:
        induct_x, induct_edge_index2, induct_edge_weights2 = model.inject_trigger(idx_atk, data.x, induct_edge_index, induct_edge_weights, device)
        if args.defense_mode in ['prune', 'isolate']:
            induct_edge_index2, induct_edge_weights2 = prune_unrelated_edge(args, induct_edge_index2, induct_edge_weights2, induct_x, device)
        with torch.no_grad():
            data_overall = Data(x=induct_x, edge_index=induct_edge_index2)
            out_overall = searched_model(data_overall)
        asr = (out_overall.argmax(dim=1)[idx_atk] == args.target_class).float().mean().item()

    return {
        'ASR': float(asr),
        'CA': float(clean_acc),
    }

def _selection_to_top_ops(source, gnn_ops=None):

    if isinstance(source, dict):
        selection = source
    else:
        selection = getattr(source, 'selection', None)
        if selection is None and hasattr(source, '_model'):
            selection = getattr(source._model, 'selection', None)
    if not isinstance(selection, dict):
        raise ValueError(
            "无法从传入对象解析出离散架构 selection；请传入 parse_model 得到的 "
            "BoxModel 或 selection 字典。"
        )

    if gnn_ops is None:
        model = getattr(source, '_model', source)
        gnn_ops = getattr(model, 'gnn_ops', None)

    def _layer_idx(key):
        try:
            return int(key.split('_')[1])
        except (IndexError, ValueError):
            return 0

    in_keys = sorted((k for k in selection if k.startswith('in_')), key=_layer_idx)
    op_keys = sorted((k for k in selection if k.startswith('op_')), key=_layer_idx)

    top = []
    for k in in_keys:
        v = selection[k]
        top.append(int(v[0]) if isinstance(v, (list, tuple)) else int(v))

    ops = []
    for k in op_keys:
        idx = int(selection[k])
        if gnn_ops is not None and 0 <= idx < len(gnn_ops):
            ops.append(gnn_ops[idx])
        else:
            ops.append(idx)

    return {'top': top, 'ops': ops}

def analyze_architecture(gnn_model, trigger_gen=None):

    result = _selection_to_top_ops(gnn_model)

    if trigger_gen is not None:
        mid_space = None
        if hasattr(trigger_gen, 'mid_space'):
            mid_space = trigger_gen.mid_space
        elif hasattr(trigger_gen, 'trojan') and hasattr(trigger_gen.trojan, 'mid_space'):
            mid_space = trigger_gen.trojan.mid_space
        if mid_space is not None:
            try:
                result['generator'] = _selection_to_top_ops(mid_space)
            except ValueError:
                pass

    return result

