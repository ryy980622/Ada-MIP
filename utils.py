import os
import pyro
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import degree, to_undirected, softmax


def relabel(batch, node_mask=None):
    if node_mask is None:
        mask = torch.zeros(batch.x.size(0), dtype=torch.bool)
        mask[batch.edge_index.flatten()] = True
        row, col = batch.edge_index
        node_idx = row.new_full((batch.x.size(0),), -1)
        idx = torch.nonzero(mask).flatten()
        node_idx[idx] = torch.arange(idx.size(0), device=row.device)
        batch.edge_index = node_idx[batch.edge_index]
        batch.x = batch.x[mask]
        batch.batch = batch.batch[mask]
    else:
        mask = torch.zeros(batch.x.size(0), dtype=torch.bool)
        mask[node_mask] = True
        row, col = batch.edge_index
        node_idx = row.new_full((batch.x.size(0),), -1)
        idx = torch.nonzero(mask).flatten()
        node_idx[idx] = torch.arange(idx.size(0), device=row.device)
        batch.edge_index = node_idx[batch.edge_index]
        batch.x = batch.x[mask]
        batch.batch = batch.batch[mask]
    return batch


def sparse_to_dense(x, indices, pad='min'):
    num_nodes = torch.bincount(indices)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
    index = torch.arange(indices.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[indices]) + (indices * max_num_nodes)
    dense_x = x.new_full((batch_size * max_num_nodes, ), torch.finfo(x.dtype).min if pad == 'min' else .0)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)
    return dense_x, index


def topk(logits, ratio, indices): #indice:点到图的映射, logits:点被采样的概率值
    if indices is not None:
        num_nodes = torch.bincount(indices)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
        x, __ = sparse_to_dense(logits, indices)
        __, sorted_idx = x.sort(dim=-1, descending=True)
        k = (ratio * num_nodes).ceil().to(torch.long) if isinstance(ratio, float) else ratio
        c = torch.cat([indices.new_zeros(1), num_nodes.cumsum(dim=0)], -1)
        idx = torch.cat([sorted_idx[i, :k[i]] + c[i] for i in range(batch_size)], dim=0)
        mask = torch.zeros((logits.numel(), ), dtype=torch.bool, device=logits.device)
        mask[idx] = True
    else:
        __, sorted_idx = logits.sort(dim=-1, descending=True)
        k = (ratio * logits.size(0)).ceil().to(torch.long) if isinstance(ratio, float) else ratio
        mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        mask[sorted_idx[:k]] = True
    return mask


def gumble_topk(logits, ratio, indices):
    u = torch.rand_like(logits)
    z = -torch.log(-torch.log(u))
    if indices is not None:
        z = z + torch.log(softmax(logits, indices))
    else:
        z = z + torch.log(F.softmax(logits, dim=-1))
    return topk(z, ratio, indices)


def sample(high: int, size: int, device=None):
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)


def negative_sampling(edge_index, num_nodes=None, num_neg_samples=None,
                      method="sparse", force_undirected=False):

    if not num_nodes:
        num_nodes = torch.unique(edge_index[0]).size(0)

    size = num_nodes * num_nodes

    if not num_neg_samples:
        num_neg_samples = size - edge_index.size(1)

    row, col = edge_index

    if force_undirected:
        num_neg_samples = num_neg_samples // 2

        # Upper triangle indices: N + ... + 1 = N (N + 1) / 2
        size = (num_nodes * (num_nodes + 1)) // 2

        # Remove edges in the lower triangle matrix.
        mask = row <= col
        row, col = row[mask], col[mask]

        # idx = N * i + j - i * (i+1) / 2
        idx = row * num_nodes + col - row * (row + 1) // 2
    else:
        idx = row * num_nodes + col

    # Percentage of edges to oversample so that we are save to only sample once
    # (in most cases).

    alpha = abs(1 / (1 - 1.1 * (edge_index.size(1) / size))) if (1 - 1.1 * (edge_index.size(1) / size)) > 1 else 1.

    if method == 'dense':
        mask = edge_index.new_ones(size, dtype=torch.bool)
        mask[idx] = False
        mask = mask.view(-1)

        perm = sample(size, int(alpha * num_neg_samples), device=edge_index.device)
        perm = perm[mask[perm]][:num_neg_samples]

    else:
        perm = sample(size, int(alpha * num_neg_samples))
        mask = torch.from_numpy(np.isin(perm, idx.to('cpu'))).to(torch.bool)
        perm = perm[~mask][:num_neg_samples].to(edge_index.device)

    if perm.size(0) == 0:
        return None

    if force_undirected:
        row = torch.floor((-torch.sqrt((2. * num_nodes + 1.)**2 - 8. * perm) + 2 * num_nodes + 1) / 2)
        col = perm - row * (2 * num_nodes - row - 1) // 2
        neg_edge_index = torch.stack([row, col], dim=0).long()
        neg_edge_index = to_undirected(neg_edge_index)
    else:
        row = perm // num_nodes
        col = perm % num_nodes
        neg_edge_index = torch.stack([row, col], dim=0).long()

    return neg_edge_index


def batched_negative_sampling(batch, force_undirected=True):
    split = degree(batch.batch[batch.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(batch.edge_index, split, dim=1)
    num_nodes = degree(batch.batch, dtype=torch.long)
    cum_nodes = torch.cat([batch.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])

    neg_edge_indices = []
    for edge_index, N, C in zip(edge_indices, num_nodes.tolist(), cum_nodes.tolist()):
        num_neg_samples = edge_index.size(1)
        neg_edge_index = negative_sampling(edge_index - C, N, num_neg_samples, force_undirected=force_undirected)
        if neg_edge_index is not None:
            neg_edge_indices.append(neg_edge_index + C)
    return torch.cat(neg_edge_indices, dim=1)


def accuracy(yt, yp):
    correct = (yt == yp).sum().item()
    return (correct / yt.size(0)) * 100


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)


def mute_warning():
    import sys
    import warnings
    warnings.filterwarnings('ignore')
    if not sys.warnoptions:
        warnings.simplefilter('ignore')
        os.environ['PYTHONWARNINGS'] = 'ignore'


if __name__ == '__main__':
    topk(torch.tensor([-.8, 0.3, 1.2, .4, .2, .9, -.9, -.2, .1, -.05]), .5,
                torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long))