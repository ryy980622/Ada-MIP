import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_max, scatter
from torch_geometric.utils import softmax, k_hop_subgraph
from pyro.distributions import RelaxedBernoulliStraightThrough, RelaxedOneHotCategoricalStraightThrough
from utils import relabel, negative_sampling, batched_negative_sampling, topk, gumble_topk, sparse_to_dense
from torch.nn.parameter import Parameter
import numpy as np
from arguments import arg_parse



class Attention(nn.Module):
    def __init__(self, input_dim, dim, num_layer=3):
        super().__init__()
        input_dim, dim = int(input_dim), int(dim)
        self.W = Parameter(torch.empty(size=(input_dim, 2*dim)))
        self.norm = nn.LayerNorm(2 * dim)
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.q = Parameter(torch.empty(size=(2*dim, 1)))
        torch.nn.init.xavier_uniform_(self.q.data, gain=1.414)
        #self.l2 = nn.Linear(2*input_dim, dim)
        self.l3 = nn.Linear(dim, 1)
        self.b = nn.Linear(num_layer, dim)

    def forward(self, x): # x: batch * num_view * input_dim
        w = torch.matmul(x, self.W) # w:batch * num_view * dim
        w = self.norm(w)
        w = F.relu(w)
        w = torch.matmul(w, self.q)   # w: batch * num_view * 1
        w = F.softmax(w, dim=1) # w: batch * num_view * 1
        w = w.transpose(1, 2)   # w: batch * 1 * num_view
        # out = F.relu(self.l2(x + torch.matmul(w, x_aug).squeeze()))
        # out = F.relu(self.l2(torch.cat([x, torch.matmul(w, x_aug).squeeze()], dim=1)))
        out = self.l3(torch.matmul(w, x).squeeze())
        return out.view(-1)
class Node_sampler(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, num_layer=3, ratio=0.9, random_augment=False, random_drop=0.2):
        super(Node_sampler, self).__init__()
        self.ratio = ratio
        self.random_augment = random_augment
        self.num_layer = num_layer
        if not self.random_augment:
            self.att = Attention(hid_dim/num_layer, hid_dim/num_layer, num_layer=num_layer)
            self.head = nn.Sequential(
                nn.Linear(hid_dim, 2 * hid_dim),
                nn.LayerNorm(2 * hid_dim),
                nn.ReLU(),
                nn.Linear(2 * hid_dim, 1), )
        else:
            self.drop = nn.Dropout(random_drop)

    def forward(self, batch, h, g, temperature=0.8):
        if not self.random_augment:
            n_nodes = torch.bincount(batch.batch)
            #logits = h + torch.repeat_interleave(g, n_nodes, dim=0)
            #logits = logits.view(logits.shape[0], self.num_layer, -1)
            #logits = self.att(logits).squeeze()

            logits = self.head(h + torch.repeat_interleave(g, n_nodes, dim=0)).squeeze()
            mask = topk(logits, self.ratio, batch.batch)
            # mask = RelaxedBernoulliStraightThrough(temperature, logits=logits).rsample().to(torch.bool)
            #mask = gumble_topk(logits / temperature, self.ratio, batch.batch)
            mask = torch.gather(mask, 0, batch.edge_index[0]) * torch.gather(mask, 0, batch.edge_index[1])
            edge_weight = torch.gather(logits, 0, batch.edge_index[0]) + torch.gather(logits, 0, batch.edge_index[1])
            #edge_weight = torch.gather(torch.ones_like(logits), 0, batch.edge_index[0])
            #batch.x = batch.x*logits.unsqueeze(1)

            batch.edge_index = batch.edge_index[:, mask.bool()]
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr[mask.bool(), :]
            batch.edge_weight = edge_weight[mask.bool()]
            batch.node_weight = logits
        else:
            b = deepcopy(batch)
            mask = torch.ones(batch.x.size(0), device=batch.x.device)
            c = torch.cat([batch.batch.new_zeros(1), torch.bincount(batch.batch).cumsum(dim=0)], -1)
            mask = torch.cat([self.drop(mask[c[i]:c[i + 1]]) for i in range(len(c) - 1)]).bool()
            mask = torch.gather(mask, 0, batch.edge_index[0]) * torch.gather(mask, 0, batch.edge_index[1])
            batch.edge_index = batch.edge_index[:, mask]
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr[mask, :]
        mask = batch.x.new_zeros((batch.x.size(0),), dtype=torch.bool)
        mask[batch.edge_index.flatten()] = True
        return batch, mask

    @torch.no_grad()
    def inference(self, batch, h, g):
        if not self.random_augment:
            n_nodes = torch.bincount(batch.batch)
            logits = self.head(h + torch.repeat_interleave(g, n_nodes, dim=0)).squeeze()
            mask = topk(logits, 0.9, batch.batch)
            mask = torch.gather(mask, 0, batch.edge_index[0]) * torch.gather(mask, 0, batch.edge_index[1])
            edge_weight = torch.gather(logits, 0, batch.edge_index[0]) + torch.gather(logits, 0, batch.edge_index[1])
            batch.edge_index = batch.edge_index[:, mask.bool()]
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr[mask.bool(), :]
            batch.edge_weight = edge_weight[mask.bool()]
        else:
            mask = torch.ones(batch.x.size(0), device=batch.x.device)
            c = torch.cat([batch.batch.new_zeros(1), torch.bincount(batch.batch).cumsum(dim=0)], -1)
            mask = torch.cat([self.drop(mask[c[i]:c[i + 1]]) for i in range(len(c) - 1)]).bool()
            mask = torch.gather(mask, 0, batch.edge_index[0]) * torch.gather(mask, 0, batch.edge_index[1])
            batch.edge_index = batch.edge_index[:, mask]
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr[mask, :]
        mask = batch.x.new_zeros((batch.x.size(0),), dtype=torch.bool)
        mask[batch.edge_index.flatten()] = True
        return batch, mask
class Edge_sampler(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, random_augment=False, random_drop=0.2):
        super(Edge_sampler, self).__init__()
        self.random_augment = random_augment
        if not self.random_augment:
            self.head = nn.Sequential(
                nn.Linear(hid_dim + 1, 2 * hid_dim),
                nn.LayerNorm(2 * hid_dim),
                nn.ReLU(),
                nn.Linear(2 * hid_dim, 1), )
        else:
            self.drop = nn.Dropout(random_drop)

    def forward(self, batch, h, g, temperature=0.8):
        if not self.random_augment:
            pos_edge = batch.edge_index
            neg_edge = batched_negative_sampling(batch) if len(batch.batch.unique()) > 1 \
                else negative_sampling(pos_edge, num_neg_samples=batch.edge_index.size(1))
            __, idx = torch.sort(torch.gather(batch.batch, 0, torch.cat((pos_edge[0], neg_edge[0]), dim=-1)))
            edges = torch.cat((pos_edge, neg_edge), dim=-1)[:, idx]
            connectivity = torch.ones_like(edges[0])
            connectivity[idx >= pos_edge.size(1)] = 0.
            enc = torch.cat((h[edges[0]] + h[edges[1]], connectivity.view(-1, 1)), dim=-1)
            logits = self.head(enc).squeeze()
            # mask = RelaxedBernoulliStraightThrough(temperature, logits=logits).rsample().to(torch.bool)
            k = (torch.bincount(torch.gather(batch.batch, 0, pos_edge[0])) * 0.9).to(torch.long)
            #all_edges = torch.cat((pos_edge[0], neg_edge[0]), dim=0)
            #k = (torch.bincount(torch.gather(batch.batch, 0, all_edges))*0.7).to(torch.long)
            mask = topk(logits, k, torch.gather(batch.batch, 0, edges[0]))
            # mask = gumble_topk(logits / temperature, k, torch.gather(batch.batch, 0, edges[0]))
            batch.edge_index = edges[:, mask]
            if batch.edge_attr is not None:
                batch.edge_attr = torch.cat((batch.edge_attr, torch.zeros((neg_edge.shape[1], batch.edge_attr.shape[1]),
                                                                          dtype=batch.edge_attr.dtype,
                                                                          device=batch.edge_attr.device)), 0)
                batch.edge_attr = batch.edge_attr[mask, :]
            batch.edge_weight = logits[mask]
        else:
            mask = torch.ones(batch.edge_index.size(1), dtype=torch.float, device=batch.x.device)
            mask = self.drop(mask).bool()
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr[mask, :]
            batch.edge_index = batch.edge_index[:, mask]

        mask = batch.x.new_zeros((batch.x.size(0),), dtype=torch.bool)
        mask[batch.edge_index.flatten()] = True
        return batch, mask

    @torch.no_grad()
    def inference(self, batch, h, g):
        if not self.random_augment:
            pos_edge = batch.edge_index
            neg_edge = batched_negative_sampling(batch) if len(batch.batch.unique()) > 1 else negative_sampling(batch)
            __, idx = torch.sort(torch.gather(batch.batch, 0, torch.cat((pos_edge[0], neg_edge[0]), dim=-1)))
            edges = torch.cat((pos_edge, neg_edge), dim=-1)[:, idx]
            connectivity = torch.ones_like(edges[0], device=batch.x.device)
            connectivity[idx >= pos_edge.size(1)] = 0.
            enc = torch.cat((h[edges[0]] + h[edges[1]], connectivity.view(-1, 1)), dim=-1)
            logits = self.head(enc).squeeze()
            mask = topk(logits, 0.5, torch.gather(batch.batch, 0, edges[0]))
            batch.edge_index = edges[:, mask]
            batch.edge_weight = logits[mask]
        else:
            mask = torch.ones(batch.edge_index.size(1), dtype=torch.bool, device=batch.x.device)
            mask = self.drop(mask)
            batch.edge_index = batch.edge_index[:, mask]
        mask = batch.x.new_zeros((batch.x.size(0),), dtype=torch.bool)
        mask[batch.edge_index.flatten()] = True
        return batch, mask
class Subgraph_sampler(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, n_hops=2, random_augment=False, random_drop=0.2):
        super(Subgraph_sampler, self).__init__()
        self.n_hops = n_hops
        self.random_augment = random_augment
        if not self.random_augment:
            self.head = nn.Sequential(
                nn.Linear(hid_dim, 2 * hid_dim),
                nn.LayerNorm(2 * hid_dim),
                nn.ReLU(),
                nn.Linear(2 * hid_dim, 1), )

    def forward(self, batch, h, g, temperature=0.8):
        if not self.random_augment:
            n_nodes = torch.bincount(batch.batch)
            logits = self.head(h + torch.repeat_interleave(g, n_nodes, dim=0)).squeeze()
            node_prob = softmax(logits, batch.batch).squeeze()
            node_prob, index = sparse_to_dense(node_prob, batch.batch, pad=.0)
            num_nodes = torch.bincount(batch.batch)
            max_num_nodes = num_nodes.max().item()
            _, node_sample = torch.topk(node_prob, 1, dim=1)
            node_sample = F.one_hot(node_sample, max_num_nodes).squeeze()

            #node_sample = RelaxedOneHotCategoricalStraightThrough(temperature, probs=node_prob).rsample()
            node_sample = torch.gather(node_sample.flatten(), 0, index)
            __, __, __, mask = k_hop_subgraph(torch.nonzero(node_sample).flatten(), self.n_hops, batch.edge_index)
            edge_weight = torch.gather(logits, 0, batch.edge_index[0]) + torch.gather(logits, 0, batch.edge_index[1])
            #edge_weight = torch.gather(torch.ones_like(logits), 0, batch.edge_index[0])
            #batch.x = batch.x * logits.unsqueeze(1)

            batch.edge_index = batch.edge_index[:, mask]
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr[mask, :]
            batch.edge_weight = edge_weight[mask]
            batch.node_weight = node_prob.squeeze()
            #print(batch.node_weight)
        else:
            n_nodes = torch.bincount(batch.batch)
            c_nodes = torch.cat((torch.zeros(1, dtype=torch.long, device=batch.x.device), n_nodes.cumsum(dim=0)),
                                dim=-1)
            idx = [(c_nodes[i] + torch.randint(0, n, (1,), device=batch.x.device)).item() for i, n in
                   enumerate(n_nodes)]
            __, __, __, mask = k_hop_subgraph(idx, self.n_hops, batch.edge_index)
            batch.edge_index = batch.edge_index[:, mask]
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr[mask, :]
        mask = batch.x.new_zeros((batch.x.size(0),), dtype=torch.bool)
        mask[batch.edge_index.flatten()] = True
        return batch, mask

    @torch.no_grad()
    def inference(self, batch, h, g, n_hops=2):
        _, batch = relabel(batch)
        if not self.random_augment:
            n_nodes = torch.bincount(batch.batch)
            logits = self.head(h + torch.repeat_interleave(g, n_nodes, dim=0)).flatten()
            __, node_sample = scatter_max(logits, batch.batch)
            __, __, __, mask = k_hop_subgraph(node_sample, n_hops, batch.edge_index)
            edge_weight = torch.gather(logits, 0, batch.edge_index[0]) + torch.gather(logits, 0, batch.edge_index[1])
            batch.edge_index = batch.edge_index[:, mask]
            batch.edge_weight = edge_weight[mask]
        else:
            n_nodes = torch.bincount(batch.batch)
            c_nodes = torch.cat((torch.zeros(1, dtype=torch.long, device=n_nodes.device), n_nodes.cumsum(dim=0)),
                                dim=-1)
            idx = [(c_nodes[i] + torch.randint(0, n, (1,), device=batch.x.device)).item() for i, n in
                   enumerate(n_nodes)]
            __, __, __, mask = k_hop_subgraph(idx, n_hops, batch.edge_index)
            batch.edge_index = batch.edge_index[:, mask]
        mask = batch.x.new_zeros((batch.x.size(0),), dtype=torch.bool)
        mask[batch.edge_index.flatten()] = True
        return batch, mask
class Attr_sampler(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, random_augment=False, random_drop=0.2):
        super(Attr_sampler, self).__init__()
        self.random_augment = random_augment
        self.in_dim = in_dim
        self.device = arg_parse().device
        if not self.random_augment:
            self.linear = nn.Linear(in_dim, in_dim)
            self.head = nn.Sequential(
                nn.Linear(hid_dim, 2 * hid_dim),
                nn.LayerNorm(2 * hid_dim),
                nn.ReLU(),
                #nn.Linear(2 * hid_dim, in_dim), )
                nn.Linear(2 * hid_dim, 1), )
        else:
            self.drop = nn.Dropout(random_drop)

    def forward(self, batch, h, g, temperature=0.8):
        if not self.random_augment:
            #if not batch.x.dtype == torch.int64:
                #batch.x = self.linear(batch.x)
            logits = self.head(h)
            logits = F.sigmoid(logits).squeeze()
            mask = topk(logits, 0.9, batch.batch)
            res_idx = (mask == 1).nonzero().squeeze()
            drop_idx = (mask == 0).nonzero().squeeze()
            try:
                mask_num = drop_idx.shape[0]
            except:
                mask_num = 1
            batch.x[res_idx] = batch.x[res_idx]*logits[res_idx].unsqueeze(1)
            batch.x[drop_idx] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, self.in_dim)), dtype=torch.float32).to(self.device)
            '''
            #mask = RelaxedBernoulliStraightThrough(temperature=temperature, logits=logits).rsample()
            if batch.x.dtype == torch.int64:
                batch.x = batch.x * torch.squeeze(mask).bool()
            else:
                batch.x = batch.x * mask
            '''
            batch.node_weight = logits
        else:
            batch.x = self.drop(batch.x)
        mask = batch.x.new_zeros((batch.x.size(0),), dtype=torch.bool)
        mask[batch.edge_index.flatten()] = True
        return batch, mask

    @torch.no_grad()
    def inference(self, batch, h, g):
        if not self.random_augment:
            batch.x = self.linear(batch.x)
            logits = self.head(h)
            mask = torch.sigmoid(logits).round()
            batch.x = batch.x * mask
        mask = batch.x.new_zeros((batch.x.size(0),), dtype=torch.bool)
        mask[batch.edge_index.flatten()] = True
        return batch, mask
