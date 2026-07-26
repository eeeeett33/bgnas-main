
import torch
import torch.nn.functional as F

def _bce_loss_with_logits(output, labels, **kwargs):

    return F.binary_cross_entropy_with_logits(output, labels, **kwargs)

def _loss_with_logits(output, labels, **kwargs):

    return F.cross_entropy(output, labels, **kwargs)

def minimax_loss_gen(output_fake, real_label_val=1.0, **kwargs):

    real_labels = torch.full((output_fake.shape[0],1),
                            real_label_val,
                            device=output_fake.device)
    loss = _bce_loss_with_logits(output_fake, real_labels, **kwargs)

    return loss

def minimax_loss_dis(output_fake,
                     output_real,
                     real_label_val=1.0,
                     fake_label_val=0.0,
                     **kwargs):

    fake_labels = torch.full((output_fake.shape[0],1),
                            fake_label_val,
                            device=output_fake.device)
    real_labels = torch.full((output_real.shape[0],1),
                            real_label_val,
                            device=output_real.device)

    errD_fake = _bce_loss_with_logits(output=output_fake,
                                      labels=fake_labels,
                                      **kwargs)

    errD_real = _bce_loss_with_logits(output=output_real,
                                      labels=real_labels,
                                      **kwargs)

    loss = errD_real + errD_fake

    return loss

def ns_loss_gen(output_fake):

    output_fake = torch.sigmoid(output_fake)

    return -torch.mean(torch.log(output_fake + 1e-8))

def wasserstein_loss_dis(output_real, output_fake):

    loss = -1.0 * output_real.mean() + output_fake.mean()

    return loss

def wasserstein_loss_gen(output_fake):

    loss = -output_fake.mean()

    return loss

def hinge_loss_dis(output_fake, output_real):

    loss = F.relu(1.0 - output_real).mean() + \
           F.relu(1.0 + output_fake).mean()

    return loss

def hinge_loss_gen(output_fake):

    loss = -output_fake.mean()

    return loss

def compute_gan_loss(loss_type, output):

    if loss_type == "gan":
        lossG = minimax_loss_gen(output)

    elif loss_type == "ns":
        lossG = ns_loss_gen(output)

    elif loss_type == "hinge":
        lossG = hinge_loss_gen(output)

    elif loss_type == "wasserstein":
        lossG = wasserstein_loss_gen(output)

    else:
        raise ValueError("Invalid loss_type {} selected.".format(loss_type))

    return lossG

def pdist(a, b, p=2,eps=1e-16):
    return ((a-b).abs().pow(p).sum(-1) + eps).pow(1/p)

def percept_loss(all_emb_list, inj_idx):
    n_inj = inj_idx.shape[0]
    graphps_mx = torch.zeros(n_inj, n_inj, device=all_emb_list[0].device)
    for layer in range(len(all_emb_list)):
        emb = all_emb_list[layer][inj_idx]
        norm_emb = F.normalize(emb)
        layer_loss = pdist(norm_emb.unsqueeze(1), norm_emb.unsqueeze(0))
        graphps_mx += layer_loss
    graphps_triu = torch.triu(graphps_mx)
    indices = graphps_triu.nonzero().t()
    graphps = graphps_triu[indices[0],indices[1]].sum() / indices.shape[1]
    return -graphps

def tensor2sparse_coo_tensor(edge_index, n, device):
    num_edges = edge_index.size(1)
    indices = torch.stack([edge_index[0], edge_index[1]], dim=0)
    values = torch.ones(num_edges, dtype=torch.float).to(device)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (n, n))
    return sparse_tensor

def add_feature(f1, f2):
    pass

def add_edge_index(e1, e2):
    pass

def compute_D_loss(real_batch, fake_batch, netD, adj_tensor, feat, new_feat, new_adj_tensor,  n , new_n, device,clipfeat=False):
    real_batch_size = len(real_batch)
    fake_batch_size = len(fake_batch)

    new_feat = torch.cat((feat, new_feat), dim=0)
    new_adj_tensor = torch.cat((adj_tensor, new_adj_tensor), dim=1)

    adj_tensor = tensor2sparse_coo_tensor(adj_tensor, n, device)
    new_adj_tensor = tensor2sparse_coo_tensor(new_adj_tensor, new_n, device)

    real_rate = real_batch_size/(real_batch_size+fake_batch_size)
    fake_rate = fake_batch_size/(real_batch_size+fake_batch_size)
    pred_real = netD(feat, adj_tensor)[1]

    if clipfeat:
        clip_feat = torch.clamp(new_feat.detach(),feat.min(),feat.max())
        pred_fake_D = netD(clip_feat,new_adj_tensor.detach())[1]
    else:
        pred_fake_D = netD(new_feat.detach(),new_adj_tensor.detach())[1]
    loss_D = netD.compute_gan_loss(pred_real[real_batch], pred_fake_D[fake_batch])

    real_label = torch.full((real_batch_size,1), 1.0, device=pred_real.device)
    fake_label = torch.full((fake_batch_size,1), 0.0, device=pred_real.device)
    acc_real = netD.compute_acc(pred_real[real_batch], real_label)
    acc_fake = netD.compute_acc(pred_fake_D[fake_batch], fake_label)
    acc_D =  acc_real * real_rate + acc_fake * fake_rate
    return loss_D, acc_D, acc_real, acc_fake
