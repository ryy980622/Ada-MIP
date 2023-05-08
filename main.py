import copy
import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
# from core.encoders import *

# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset

from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin_weighted import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse
from torch_geometric.transforms import Constant
from weisfeiler_lehman_subtree import cal_sim_matrix, svc_classify
#from auto_aug import Node_sampler, Edge_sampler, Subgraph_sampler, Attr_sampler
from auto_aug_new import Node_sampler, Edge_sampler, Subgraph_sampler, Attr_sampler
from copy import deepcopy
from gin_weighted import GNN
from graph_visualization import visualize
import time

class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim, output_dim=3)
        self.global_d = FF(self.embedding_dim, output_dim=3)
        self.com_d = FF(self.embedding_dim, output_dim=3)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode = 'fd'
        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1, num_view=4):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.node_sampler = Node_sampler(num_gc_layers*hidden_dim, num_gc_layers*hidden_dim)
        self.edge_sampler = Edge_sampler(num_gc_layers*hidden_dim, num_gc_layers*hidden_dim)
        self.subgraph_sampler = Subgraph_sampler(num_gc_layers*hidden_dim, num_gc_layers*hidden_dim)
        self.attr_sampler = Attr_sampler(dataset_num_features, num_gc_layers*hidden_dim)
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.aug_encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        #self.base_encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.base_encoder = GNN(dataset_num_features, hidden_dim, num_gc_layers, dropout=0)
        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        #self.com_d = FF(self.embedding_dim)

        self.proj_heads = []
        for _ in range(num_view):
            self.proj_heads.append(
                nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                # nn.Sequential(nn.Linear(hidden_dim, self.embedding_dim), nn.ReLU(inplace=True),
                              nn.Linear(self.embedding_dim, self.embedding_dim)))
        for i, proj_head in enumerate(self.proj_heads):
            self.add_module('proj_head_{}'.format(i), proj_head)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, batch_merge):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y_aug, M_aug = self.aug_encoder(x, edge_index, batch)

        batch_aug0, mask_aug0 = self.node_sampler(deepcopy(batch_merge), M_aug, y_aug)
        batch_aug1, mask_aug1 = self.edge_sampler(deepcopy(batch_merge), M_aug, y_aug)
        batch_aug2, mask_aug2 = self.subgraph_sampler(deepcopy(batch_merge), M_aug, y_aug)
        batch_aug3, mask_aug3 = self.attr_sampler(deepcopy(batch_merge), M_aug, y_aug)


        M, y = self.base_encoder(x, edge_index, None, batch, None)
        #y, M = self.base_encoder(x, edge_index, batch)
        '''
        M_aug0, y_aug0 = M, y
        M_aug1, y_aug1 = M, y
        M_aug2, y_aug2 = M, y
        M_aug3, y_aug3 = M, y
        g_cons, g_aug0, g_aug1, g_aug2, g_aug3 = y, y, y, y, y
        '''
        M_aug0, y_aug0  = self.base_encoder(batch_aug0.x, batch_aug0.edge_index, batch_aug0.edge_weight, batch_aug0.batch, mask_aug0)
        M_aug1, y_aug1 = self.base_encoder(batch_aug1.x, batch_aug1.edge_index, batch_aug1.edge_weight, batch_aug1.batch, mask_aug1)
        M_aug2, y_aug2 = self.base_encoder(batch_aug2.x, batch_aug2.edge_index, batch_aug2.edge_weight, batch_aug2.batch, mask_aug2)
        M_aug3, y_aug3 = self.base_encoder(batch_aug3.x, batch_aug3.edge_index, None, batch_aug3.batch, mask_aug3)

        #print(y.shape,y_aug0.shape)
        g_sim = self.proj_heads[1](y)
        g_cons = self.proj_heads[0](y)

        g_aug0 = self.proj_heads[0](y_aug0)
        g_aug1 = self.proj_heads[0](y_aug1)
        g_aug2 = self.proj_heads[0](y_aug2)
        g_aug3 = self.proj_heads[0](y_aug3)


        return g_cons, g_sim, y, g_aug0, y_aug0, g_aug1, y_aug1, g_aug2, y_aug2, g_aug3, y_aug3
        #return g_sim

    def get_embedding(self, loader):
        #device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        device = args.device
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                _, x = self.base_encoder(x, edge_index, None, batch, None)
                #x, _ = self.base_encoder(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)  # 所有图的embedding：图数*5*32
        y = np.concatenate(y, 0)
        return ret, y

    def loss_cal(self, x, x_aug):

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def loss_all(self, x_aug0, x_aug1, x_aug2, x_aug3):
        return model.loss_cal(x_aug0, x_aug1) + model.loss_cal(x_aug0, x_aug2) + model.loss_cal(x_aug0, x_aug3) + model.loss_cal(x_aug1,
                                                                                                  x_aug2) + model.loss_cal(x_aug1, x_aug3) + model.loss_cal(x_aug2, x_aug3)
        #return model.loss_cal(x_aug1, x_aug2) + model.loss_cal(x_aug1, x_aug3) + model.loss_cal(x_aug2, x_aug3)
    def loss_GED(self, z, dis):
        num_graph = z.size()[0]
        z1, z2 = torch.chunk(z, 2, dim=0)
        loss = (torch.norm(z1 - z2, 2, dim=1) ** 2 - dis) ** 2
        loss = torch.mean(loss)
        return loss

    def loss_kernel_sim(self, z,  GED_W):
        batch_size, _ = z.size()
        z_abs = z.norm(dim=1)
        GED_W = GED_W * 2 - 1
        sim_matrix = torch.einsum('ik,jk->ij', z, z) / torch.einsum('i,j->ij', z_abs, z_abs)
        loss_func = nn.MSELoss(reduction='mean')
        return loss_func(sim_matrix, GED_W)
    def visualize_all_graph(self, loader):
        device = args.device
        with torch.no_grad():
            for data in loader:
                data = data[0]
                data.to(device)
                y_aug, M_aug = self.aug_encoder(data.x, data.edge_index, data.batch)
                batch_aug0, mask_aug0 = self.node_sampler(deepcopy(data), M_aug, y_aug)
                #batch_aug1, mask_aug1 = self.edge_sampler(deepcopy(data), M_aug, y_aug)
                batch_aug2, mask_aug2 = self.subgraph_sampler(deepcopy(data), M_aug, y_aug)
                batch_aug3, mask_aug3 = self.attr_sampler(deepcopy(data), M_aug, y_aug)

                visualize(batch_aug0, data, 0)
                #visualize(batch_aug1, data, 1)
                #visualize(batch_aug2, data, 2)
                #visualize(batch_aug3, data, 3)
        return

import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def transform(node_num, data_aug):
    edge_idx = data_aug.edge_index.numpy()
    _, edge_num = edge_idx.shape
    idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

    node_num_aug = len(idx_not_missing)
    data_aug.x = data_aug.x[idx_not_missing]

    data_aug.batch = data_aug.batch[idx_not_missing]
    idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
    edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                not edge_idx[0, n] == edge_idx[1, n]]
    data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
    return data_aug


def store_graph(edge_index, x, y, batch):
    G_s = []
    num_graph = int(batch[-1]) + 1
    node2graph = {i: int(batch[i]) for i in range(len(batch))}
    edge = {int(batch[i]): set() for i in range(len(batch))}
    node_attr = {int(batch[i]): {} for i in range(len(batch))}
    for i in range(len(x)):
        g = node2graph[i]

        node_attr[g][i] = int(x[i].argmax(dim=0))
    for i in range(len(edge_index[0])):
        g = node2graph[int(edge_index[0][i])]
        edge[g].add((int(edge_index[0][i]), int(edge_index[1][i])))
    for i in range(num_graph):
        tem_list = [edge[i], node_attr[i], {}]
        G_s.append(tem_list)
    return G_s


def store_eval_graph(dataloader_eval):
    G_s = []
    for data in dataloader_eval:
        data = data[0]
        G_s += store_graph(data.edge_index, data.x, data.y, data.batch)
    return G_s


def init_feature(dataset):
    deg = []
    num_graph = int(dataset.data.y.shape[0])
    mx_degree = 0
    for i in range(num_graph):
        data = dataset.get(i)[0]
        l = [0 for _ in range(len(data.x))]
        for j in range(len(data.edge_index[0])):
            v1, v2 = data.edge_index[0][j], data.edge_index[1][j]
            l[v1] += 1
            l[v2] += 1
            mx_degree = max(mx_degree, max(l[v1], l[v2]))
        deg += l
    '''
    pre_node =0
    d = 0

    for i in range(len(dataset.data.edge_index[0])):
        cur_node = int(dataset.data.edge_index[0][i])
        if cur_node == pre_node:
            d += 1
        else:
            mx_degree = max(d, mx_degree)
            deg.append(d)
            d = 1
        pre_node = cur_node
    deg.append(d)
    '''
    mx_dim = 96
    num_node, _ = dataset.data.x.shape
    assert (num_node == len(deg))
    if mx_degree > mx_dim:
        mx_degree = mx_dim
    new_x = torch.zeros(num_node, mx_degree)
    for i, d in enumerate(deg):

        if d >= mx_dim:
            new_x[i][-1] = 1
        else:
            new_x[i][d-1] = 1

        #new_x[i][d - 1] = 1
    dataset.data.x = new_x
    return dataset


if __name__ == '__main__':
    sim_losses = []
    cons_losses = []
    mx_acc = 0
    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val': [], 'test': []}
    epochs = 500
    log_interval = 1
    batch_size = 128
    #batch_size = 512
    lr = args.lr
    DS = args.DS
    kernel = args.kernel
    num_view = args.num_view
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS, aug=args.aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    if '-' in DS or 'COLLAB' in DS:
        dataset = init_feature(dataset)
        dataset_eval = init_feature(dataset_eval)
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=len(dataset))
    dataloader_vis = DataLoader(dataset_eval, batch_size=1)

    #device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    device = args.device
    model = simclr(args.hidden_dim, args.num_gc_layers, num_view=num_view).to(device)
    # model.eval()
    # emb, y = model.encoder.get_embeddings(dataloader_eval)
    # acc_val, acc = evaluate_embedding(emb, y)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    GED_dic = {}
    loss_min = 1e9
    for epoch in range(1, epochs + 1):
        loss_epoch, loss_sim, loss_cons = 0, 0, 0
        num_batch = 0
        model.train()
        for data in dataloader:

            # print('start')
            # from SE_augmentation import batch_com

            # data, data_aug, data_com = data
            data, data_aug0, data_aug1, data_aug2, data_aug3 = data

            if epoch == 1:
                G_batch = store_graph(data.edge_index, data.x, data.y, data.batch)
                start = time.time()
                GED_W = torch.tensor(cal_sim_matrix(G_batch, kernel=kernel), dtype=torch.float32)
                end = time.time()
                # print("preprocess time:", end-start)

                # GED_W = cal_GED(data.x, data.edge_index, data.batch)
                GED_W = GED_W.to(device)
                GED_dic[num_batch] = GED_W
            else:
                GED_W = GED_dic[num_batch]

            node_num, _ = data.x.size()
            data = data.to(device)
            x, x_sim, z_ori, x_aug0, z_aug0, x_aug1, z_aug1, x_aug2, z_aug2, x_aug3, z_aug3 = \
                model(data.x, data.edge_index, data.batch, data)
            #x_sim = model(data.x, data.edge_index, data.batch, data)

            start = time.time()

            optimizer.zero_grad()
            #loss = model.loss_all(x_aug0, x_aug1, x_aug2, x_aug3)
            sim_loss = model.loss_kernel_sim(x_sim, GED_W)
            cons_loss = model.loss_all(x_aug0, x_aug1, x_aug2, x_aug3)
            loss = sim_loss + 0.005 * cons_loss

            # print(loss)
            loss_epoch += loss.item() * data.num_graphs
            loss_sim += sim_loss.item() * data.num_graphs
            loss_cons += 0.005 *cons_loss.item() * data.num_graphs
            loss.backward()

            optimizer.step()
            end = time.time()
            # print("epoch time:", end - start)
            num_batch += 1
        sim_losses.append(float(loss_sim/1173))
        cons_losses.append(float(loss_cons/1173))
        print('Epoch {}, Loss {}'.format(epoch, loss_epoch / len(dataloader)), kernel)
        print(loss_sim/1173, loss_cons/1173)
        if epoch % log_interval == 0:
            model.eval()
            #emb, y = model.base_encoder.get_embeddings(dataloader_eval)
            emb, y = model.get_embedding(dataloader_eval)
            # if epoch == 1:
            #     model.visualize_all_graph(dataloader_vis)
            # svc_classify(G, y)

            torch.save(emb, path + '/x.pt')
            torch.save(y, path + '/y.pt')
            acc_val, acc, std = evaluate_embedding(emb, y, search=False)

            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            mx_acc = max(mx_acc, acc)
            print(mx_acc)
            # print(accuracies['val'][-1], accuracies['test'][-1])

    tpe = ('local' if args.local else '') + ('prior' if args.prior else '')
    np.save('adv_loss/sim_loss.npy', np.array(sim_losses))
    np.save('adv_loss/cons_loss.npy', np.array(cons_losses))
    with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
        f.write('\n')
