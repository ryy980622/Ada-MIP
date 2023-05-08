# Ada-MIP
Ada-MIP:  Adaptive Self-supervised Graph Representation Learning via Mutual Information and Proximity Optimization

This is our Pytorch implementation for the TKDD 2023 paper:

>Yuyang Ren, Haonan Zhang, Peng Yu, Luoyi Fu, Xinbing Wang, Xinde Cao, Guihai Chen, Fei Long and Chenghu Zhou. Ada-MIP: Adaptive Self-supervised Graph Representation Learning via Mutual Information and Proximity Optimization

Author: Yuyang Ren (renyuyang@sjtu.edu.cn)

## Introduction
Ada-MIP is an adaptive self-supervised graph representation learning framework considering both mutual information between views (unique features) and inter-graph proximity information (common features). Ada-MIP learns graphs' unique information through a learnable and probably injective augmenter, which can acquire more adaptive views compared to the augmentation strategies applied by existing GCL methods; to learn graphs' common information, Ada-MIP employs graph kernels to calculate graphs' proximity and learn graph representations among which the precomputed proximity is preserved. By sharing a global encoder, graphs' unique and common information can be well integrated into the graph representations learned by Ada-MIP.


## Requirement
The code has been tested running under Python 3.7.0. The required packages are as follows:
* torch == 1.7.1
* torch-geometric == 2.0.2
* torch-cluster == 1.5.8
* torch-scatter == 2.0.5
* torch-spline-conv  == 1.2.0
* scikit-learn == 1.0.1
* numpy == 1.21.2

* Train

```
python main.py 
```

```

Some important hyperparameters:
* `lr`
  * It indicates the learning rates. 
  * The learning rate is searched in {1e-5, 1e-4, 3e-4,1e-3}.

* `batch`
  * It indicates the batch size. 
  * We search the batch size within {16, 32}.

* `h_dim`
  * It indicates the latent dimension of node embeddings. 
  * We search the latent dimension within {16, 32, 128}.

* `K`
  * It indicates the height of HPT. 
  * We search the height of HPT within {3, 4, 5}.

* `L_o`
  * It indicates the GNN layer number of original graphs. 
  * We search L_o within {1, 2, 3}.

* `L_t`
  * It indicates the GNN layer number of HPTs. 
  * We search L_o within {2, 3, 4}.
```

## Dataset
We provide the processed datasets in the experiments. Take MUTAG as an example.
* `MUTAG.txt`
  * Train file.
  * For each graph, the first line denotes the graph size and label.
  * Then each line denotes the edge set of each node.

* `MUTAG_aug.txt`
  * The preprocessed HPT file.
  * The format of this file is the same with train file.

