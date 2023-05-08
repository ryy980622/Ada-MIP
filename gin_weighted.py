import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import sys
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from arguments import arg_parse
from torch_scatter import scatter_max, scatter
from torch_geometric.utils import softmax, k_hop_subgraph
# from pyro.distributions import RelaxedBernoulliStraightThrough, RelaxedOneHotCategoricalStraightThrough
from utils import relabel, negative_sampling, batched_negative_sampling, topk, gumble_topk, sparse_to_dense

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch, edge_weight=None):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):
            if edge_weight is None:
                x = F.relu(self.convs[i](x, edge_index))    # 5层GIN
            else:
                edge_weight = edge_weight.view(-1, 1)
                x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = self.bns[i](x)
            xs.append(x)
            # if i == 2:
                # feature_map = x2

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):
        device = arg_parse().device
        #device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)    # 所有图的embedding：图数*5*32
        y = np.concatenate(y, 0)
        return ret, y

    def get_embeddings_v(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                   break

        return x_g, ret, y



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        try:
            num_features = dataset.num_features
        except:
            num_features = 1
        dim = 32

        self.encoder = Encoder(num_features, dim)

        self.fc1 = Linear(dim*5, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class FFN(nn.Module):
    def __init__(self, hid_dim):
        super(FFN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hid_dim, 2 * hid_dim),
            nn.ReLU(),
            nn.Linear(2 * hid_dim, 2 * hid_dim),
            nn.ReLU(),
            nn.Linear(2 * hid_dim, hid_dim),
            nn.ReLU()
        )
        self.skip = nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        return self.fc(x) + self.skip(x)


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class GIN(MessagePassing):
    def __init__(self, emb_dim):
        super(GIN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim),
            nn.LayerNorm(2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim), )

    def forward(self, x, edge_index, edge_weight=None):
        return self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return self.ffn(aggr_out) + aggr_out


class GNN(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layer, dropout, readout='sum'):
        super(GNN, self).__init__()
        self.readout = readout
        self.n_layer = n_layer
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, hid_dim)
        self.layers = nn.ModuleList([GIN(hid_dim) for __ in range(n_layer)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hid_dim) for __ in range(n_layer)])
        self.norm = nn.ModuleList([nn.LayerNorm(hid_dim) for __ in range(n_layer)])
        self.node_head = FFN(hid_dim)
        # self.graph_head = FFN(hid_dim)
        self.graph_head = FFN(hid_dim*n_layer)
    def forward(self, x, edge_index, edge_weight, batch, mask):
        h = self.linear(x)
        x = h
        hs = []
        for layer in range(self.n_layer):
            h = self.layers[layer](x=h, edge_index=edge_index, edge_weight=edge_weight)
            h = self.norm[layer](h)
            #h = self.bns[layer](h)
            h = self.drop(F.relu(h))
            hs.append(h)
        #h = x + h
        hs = [self.node_head(h) for h in hs]
        if mask is not None:
            # gs = [scatter(h[mask], batch[mask], dim=0, dim_size=batch.max() + 1, reduce=self.readout) for h in hs]
            gs = [global_add_pool(h[mask], batch[mask]) for h in hs]
        else:
            # gs = [scatter(h, batch, dim=0, dim_size=batch.max() + 1, reduce=self.readout) for h in hs]
            gs = [global_add_pool(h, batch) for h in hs]
        g = torch.cat(gs, 1)
        g = self.graph_head(g)
        #return h, g
        return torch.cat(hs, dim=1), g
class Base_Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Base_Encoder, self).__init__()

        # num_features = dataset.num_features
        self.num_gc_layers = num_gc_layers

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.linear = nn.Linear(num_features, dim)
        for i in range(num_gc_layers):
            conv = GIN(dim)
            #conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch, edge_weight=None, mask=None):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        x = self.linear(x)
        xs = []
        for i in range(self.num_gc_layers):
            if edge_weight is None:
                x = F.relu(self.convs[i](x, edge_index))    # 5层GIN
            else:
                # edge_weight = edge_weight.view(-1, 1)
                x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = self.bns[i](x)
            xs.append(x)
            # if i == 2:
                # feature_map = x2
        if mask is not None:
            xpool = [global_add_pool(x[mask], batch[mask]) for x in xs]
        else:
            xpool = [global_add_pool(x, batch) for x in xs]
        # xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)
