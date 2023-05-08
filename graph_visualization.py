import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random


def visualize(batch, batch_ori, type):
    G = nx.Graph()
    if type != 1:
        mn = float(torch.min(batch.node_weight))
        mx = float(torch.max(batch.node_weight))
        batch.node_weight = -1+ random.random()*0.02 + 2*(batch.node_weight-mn)/(mx-mn)
        batch.node_weight = F.sigmoid(batch.node_weight).squeeze()
        node_weight = batch.node_weight.cpu().tolist()
        node_weight = [round(a, 2) for a in node_weight]
        nodes = []
        edges = batch_ori.edge_index.T.cpu().tolist()
        node_label = torch.nonzero(batch_ori.x).T[1].cpu().tolist()
        labels = {i: w for i, w in enumerate(node_weight)}
        for i, edge in enumerate(edges):
            if edge[0] < edge[1]:
                # G.add_edge(edge[0], edge[1], weight=edge_weight[i])
                G.add_edge(edge[0], edge[1])
                # edge_labels[(edge[0], edge[1])] = edge_weight[i]
                if (edge[0], {"size": node_weight[edge[0]]}) not in nodes:
                    nodes.append((edge[0], {"size": node_weight[edge[0]]}))
                if (edge[1], {"size": node_weight[edge[1]]}) not in nodes:
                    nodes.append((edge[1], {"size": node_weight[edge[1]]}))
        G.add_nodes_from(nodes)
        pos = nx.spring_layout(G)
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)  # 绘制图中边的权重
        # print(edge_labels)
        # nx.draw_networkx(G, pos)
        cmap = plt.cm.get_cmap('YlGn')
        nx.draw(G, with_labels=True, labels=labels, node_color=node_label, node_size=400, cmap=cmap)
        # nx.draw_networkx_labels(G, pos, labels=labels)
        plt.show()
    else:
        batch.edge_weight = F.sigmoid(batch.edge_weight).squeeze()
        edge_weight = batch.edge_weight.cpu().tolist()
        edge_weight = [round(a, 2) for a in edge_weight]
        nodes = []
        edges = batch.edge_index.T.cpu().tolist()
        node_label = torch.nonzero(batch.x).T[1].cpu().tolist()

        edge_labels = {}
        for i, edge in enumerate(edges):
            if edge[0] < edge[1]:
                # G.add_edge(edge[0], edge[1], weight=edge_weight[i])
                G.add_edge(edge[0], edge[1])
                edge_labels[(edge[0], edge[1])] = edge_weight[i]
                if edge[0] not in nodes:
                    nodes.append(edge[0])
                if edge[1] not in nodes:
                    nodes.append(edge[1])
        G.add_nodes_from(nodes)
        node_label = [x for i, x in enumerate(node_label) if i in G.nodes]
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)  # 绘制图中边的权重
        #nx.draw_networkx_edges(G, pos, style='dotted')
        num_node = G.number_of_nodes()
        '''
        for i in range(num_node):
            for j in range(num_node):
                if i<j and (i, j) not in list(G.edges):
                    G.add_edge(i, j)
        '''
        # print(edge_labels)
        # nx.draw_networkx(G, pos)
        cmap = plt.cm.get_cmap('YlGn')
        nx.draw(G, pos, with_labels=False, node_color=node_label, cmap=cmap)
        # nx.draw_networkx_labels(G, pos, labels=labels)
        plt.show()

    return








