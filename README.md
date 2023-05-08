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

## Train

```
python main.py 
```

