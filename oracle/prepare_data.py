import scipy.sparse as sp
import numpy as np
import networkx as nx
import sys
import pickle as pkl
import argparse

"""
Adapted from https://github.com/PetarV-/GAT/blob/5af87e7fce2b90ae1cbd621cd58059036a3c7436/execute_cora_sparse.py
"""


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.tocsr().astype(np.float32)
    # indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    # return indices, adj.data, adj.shape
    return adj


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(f"data/{dataset_str}/ind.{dataset_str}.{names[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"data/{dataset_str}/ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    print(adj.shape)
    print(features.shape)

    return adj, features


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-ds", type=str, choices=["citeseer", "pubmed", "cora"], default="cora")
args = parser.parse_args()
dataset = args.dataset
# cora: nnz: 13264, indices: 13264, indptr: 2709
# pubmed: nnz: 108365, indices: 108365, indptr: 19718

# citeseer: nnz: 12431, indices: 12431, indptr: 3328


adj, features = load_data(dataset)
features, spars = preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]

adj = preprocess_adj_bias(adj)
print(f"nnz: {adj.nnz}, indices: {len(adj.indices)}, indptr: {len(adj.indptr)}")
print(np.unique(adj.data))

np.savetxt(f"./data/{dataset}/{dataset}_features.txt", features, delimiter=' ')
with open(f"./data/{dataset}/{dataset}_adj.txt", "w") as fout:
    fout.write(f"{len(adj.indptr) - 1} {adj.nnz}\n")
    for idx, indice in enumerate(adj.indices):
        if idx == len(adj.indices) - 1:
            fout.write(f"{indice}\n")
        else:
            fout.write(f"{indice} ")
    for idx, ptr in enumerate(adj.indptr):
        if idx == len(adj.indptr) - 1:
            fout.write(f"{ptr}\n")
        else:
            fout.write(f"{ptr} ")
