"""
=========================================================================
Graph classification on MUTAG using the Weisfeiler-Lehman subtree kernel.
=========================================================================

Script makes use of :class:`grakel.WeisfeilerLehman`, :class:`grakel.VertexHistogram`
"""
from __future__ import print_function
print(__doc__)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath, PyramidMatch, WeisfeilerLehmanOptimalAssignment, SubgraphMatching, RandomWalk,\
    GraphletSampling, MultiscaleLaplacian
from evaluate_embedding import evaluate_embedding
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import copy

def cal_sim_matrix(G, kernel):
    if kernel == 'WL':
        gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    elif kernel == 'SP':
        gk = ShortestPath(normalize=True)
    elif kernel == 'PM':
        gk = PyramidMatch(normalize=True)
    elif kernel == 'WLOA':
        gk = WeisfeilerLehmanOptimalAssignment()
    elif kernel == 'SM':
        gk = SubgraphMatching(normalize=True, ke=None)
    elif kernel == 'RW':
        gk = RandomWalk(normalize=True)
    elif kernel == 'GL':
        gk = GraphletSampling(normalize=True)
    elif kernel == 'MLG':
        gk = MultiscaleLaplacian(normalize=True)
    K_train = gk.fit_transform(G)
    return np.array(K_train)

def svc_classify(G_list, labels, seed=0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    idx_list = list(skf.split(np.zeros(len(labels)), labels))

    accuracies = []
    for fold_id in range(0, 10):
        G_train, G_test, y_train, y_test = use_fold_data(G_list, idx_list, fold_id, labels)
        gk_gl = GraphletSampling(normalize=True)
        gk_wl = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
        gk_sp = ShortestPath(with_labels=False, normalize=True)
        gk_pm = PyramidMatch(normalize=True)
        gk_wloa = WeisfeilerLehmanOptimalAssignment(normalize=True)
        gk_sm = SubgraphMatching(normalize=True)
        gk_rw = RandomWalk(normalize=True)
        gk_mlg = MultiscaleLaplacian(normalize=True)
        # print(gk)
        # K_train_wl = gk_wl.fit_transform(G_train)
        # K_test_wl = gk_wl.transform(G_test)
        # K_train_sp = gk_sp.fit_transform(G_train)
        # K_test_sp = gk_sp.transform(G_test)
        K_train_pm = gk_mlg.fit_transform(G_train)
        K_test_pm = gk_mlg.transform(G_test)
        # K_train_wloa = gk_wloa.fit_transform(G_train)
        # K_test_wloa = gk_wloa.transform(G_test)
        sim_matrix = K_train_pm
        y = y_train

        sum_true = 0
        sum_false = 0
        sum = 0
        for i in range(len(sim_matrix)):
            for j in range(len(sim_matrix[0])):
                # print(y[i] == y[j], sim_matrix[i][j])
                if y[i] == y[j]:
                    sum_true += sim_matrix[i][j]
                    sum += 1
                else:
                    sum_false += sim_matrix[i][j]
        print('ave_true:', sum_true / sum)
        print('ave_false:', sum_false / (len(sim_matrix) * len(sim_matrix) - sum))

        # Uses the SVM classifier to perform classification
        # clf = SVC(kernel="precomputed")
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        # clf = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        clf = SVC(C=10)
        clf.fit(K_train_pm, y_train)
        y_pred = clf.predict(K_test_pm)

        # Computes and prints the classification accuracy
        acc = accuracy_score(y_test, y_pred)
        print("Kernel_Accuracy:", str(round(acc * 100, 2)) + "%")
        accuracies.append(acc)
        #print(fold_id, acc)
    print("ave:", np.mean(accuracies))



def use_fold_data(g_list, idx_list, fold_idx, labels):
    train_idx, test_idx = idx_list[fold_idx]
    G_train = [copy.deepcopy(g_list[i]) for i in train_idx]
    G_test = [copy.deepcopy(g_list[i]) for i in test_idx]
    y_train = [copy.deepcopy(labels[i]) for i in train_idx]
    y_test = [copy.deepcopy(labels[i]) for i in test_idx]
    return G_train, G_test, y_train, y_test

if __name__ == '__main__':
    # Loads the MUTAG dataset
    MUTAG = fetch_dataset("MUTAG", verbose=False)
    G, y = MUTAG.data, MUTAG.target
    # print(G)
    svc_classify(G, y, seed=1)
    '''
    # Splits the dataset into a training and a test set
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=42)
    
    
    
    # Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
    gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    #print(gk)
    K_train = gk.fit_transform(G_train)
    K_test = gk.transform(G_test)
    
    # Uses the SVM classifier to perform classification
    # clf = SVC(kernel="precomputed")
    clf = SVC(C=10)
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    
    # Computes and prints the classification accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", str(round(acc*100, 2)) + "%")
    acc_val, acc = evaluate_embedding(K_test, y_test)
    print(acc_val, acc)
    '''