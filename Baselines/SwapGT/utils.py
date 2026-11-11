import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
# from torch_sparse import spspmm
import os
import re
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
import random
import numba
from scipy.sparse import csr_matrix, coo_matrix
from torch_geometric.utils import coalesce
from typing import List, Union, Tuple


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return lap_pos_enc


def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix

def calculate_cosine_similarity_matrix(a, b, eps=1e-8):

    # h_emb (N, M)
    # normalize
    a_n = a.norm(dim=1).unsqueeze(1)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))

    b_n = b.norm(dim=1).unsqueeze(1)
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))


    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, b_norm.transpose(0,1))
    diag = torch.diag(sim_matrix)
    a_diag = torch.diag_embed(diag)
    sim_matrix = sim_matrix - a_diag

    #对称矩阵
    sim_matrix_t = sim_matrix.t()
    sim_out = torch.where(sim_matrix>0,sim_matrix,sim_matrix_t)

    return sim_out


def topk_ppr_matrix(features,
                    edge_index: torch.Tensor,
                    num_nodes: int,
                    alpha: float,
                    eps: float,
                    output_node_indices: Union[np.ndarray, torch.LongTensor],
                    topk: int,
                    max_itr: int,
                    normalization='row') -> Tuple[csr_matrix, List[np.ndarray]]:
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""
    if isinstance(output_node_indices, torch.Tensor):
        output_node_indices = output_node_indices.numpy()

    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index_np = edge_index.cpu().numpy()

    _, indptr, out_degree = np.unique(edge_index_np[0],
                                      return_index=True,
                                      return_counts=True)
    indptr = np.append(indptr, len(edge_index_np[0]))

    neighbors, weights = calc_ppr_topk_parallel(indptr, edge_index_np[1], out_degree,
                                                alpha, eps, output_node_indices, max_itr)
    ppr_matrix = construct_sparse(neighbors, weights, (len(output_node_indices), num_nodes))
    ppr_matrix = ppr_matrix.tocsr()
    
    weights_of_all_neighbors = weights

    neighbors = sparsify(neighbors, weights, topk) #转化为[n,k]
    
    print("neighbors[10]", neighbors[10])
    
    idx_token = []
    topo_features = torch.empty(features.shape[0], 1+topk, features.shape[1])
    
    for ind in range(num_nodes):
        topo_features[ind, 0, :] = features[ind]
        topk_indices = list(neighbors[ind])
        if len(topk_indices) < topk:
            random_neighbors = list(random.sample(range(0, num_nodes), topk-len(topk_indices)))
            topk_indices = topk_indices + random_neighbors
        idx_token.append(topk_indices)
        topo_features[ind, 1:, :] = features[topk_indices]

    idx_token = torch.tensor(idx_token)

    return ppr_matrix, idx_token, topo_features


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int_))
    j = np.concatenate(neighbors)
    return coo_matrix((np.concatenate(weights), (i, j)), shape)


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/gdc.html#GDC.diffusion_matrix_approx
@numba.njit(cache=True, parallel=True)
def calc_ppr_topk_parallel(indptr, indices, deg, alpha, epsilon, nodes, max_itr) \
        -> Tuple[List[np.ndarray], List[np.ndarray]]:
    js = [np.zeros(0, dtype=np.int64)] * len(nodes)
    vals = [np.zeros(0, dtype=np.float32)] * len(nodes)
    for i in numba.prange(len(nodes)):
        # print(i)
        j, val = _calc_ppr_node(nodes[i], indptr, indices, deg, alpha, epsilon, max_itr)
        js[i] = np.array(j)
        vals[i] = np.array(val)
    return js, vals   # 节点集、权重集


@numba.njit(cache=True, locals={'_val': numba.float32, 'res': numba.float32, 'res_vnode': numba.float32})
def _calc_ppr_node(inode, indptr, indices, deg, alpha, epsilon, max_itr):
    alpha_eps = alpha * epsilon
    f32_0 = numba.float32(0)
    p = {inode: f32_0}
    r = {}
    r[inode] = alpha
    q = [inode]
    
    time_to_stop = 0

    while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else f32_0
        if unode in p:
            p[unode] += res
        else:
            p[unode] = res
        r[unode] = f32_0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
            _val = (1 - alpha) * res / deg[unode]
            if vnode in r:
                r[vnode] += _val
            else:
                r[vnode] = _val

            res_vnode = r[vnode] if vnode in r else f32_0
            if res_vnode >= alpha_eps * deg[vnode]:
                if vnode not in q:
                    q.append(vnode)
        
        time_to_stop += 1
        if time_to_stop >= max_itr:
            # print("Maximum time iteration reached of push operation for node " + str(inode))
            break
    # print('PPR stoped at ' + str(time_to_stop) + ' iteration for node ' + str(inode))
    return list(p.keys()), list(p.values())

def sparsify(neighbors: List[np.ndarray], weights: List[np.ndarray], topk: int):
    new_neighbors = []
    for n, w in zip(neighbors, weights):
        idx_topk = np.argsort(w)[-topk:]
        new_neighbor = n[idx_topk]
        new_neighbors.append(new_neighbor)

    return new_neighbors



def node_neighborhood_feature(adj, features, k, alpha):

    x_0 = features
    for i in range(k):
        features = (1-alpha) * torch.mm(adj, features) + alpha * x_0

    return features



# random obtain k from 2k (ablation) 
def mix_token_ab(token_sequence, token_idx, num_aug, depth, features):
    # 1 + 2k

    lenth = int((token_sequence.shape[1]-1)/2)
    
    #token_sequence, size:[n, k+1, d]
    split_list = [1, token_sequence.shape[1]-1]
    split_tensor = torch.split(token_sequence, split_list, dim=1)
    
    # node itself
    node_self = split_tensor[0] #[n, 1, d]
    # nodes' tokens
    sampling_tokens = split_tensor[1] #[n, 2k, d]
    temp_sequence = sampling_tokens[:,:lenth,:]
    # output token
    final_aug_token = torch.cat((node_self, temp_sequence), dim=1)
    
    #numpy_array = sampling_tokens.numpy()

    for i in range(num_aug):
        
        temp_index = np.random.choice(np.arange(0, 2 * lenth), size=lenth, replace=False)
        #temp_index = np.random.choice(np.arange(0, 2 * lenth), size=lenth)
        print(temp_index)
        temp_sequence = sampling_tokens[:,temp_index,:]
        temp_token = torch.cat((node_self, temp_sequence), dim=1)

        final_aug_token = torch.cat((final_aug_token, temp_token), dim=1)

    return final_aug_token # [n, (k+1)*(num_aug+1), d]  



def mix_token(token_sequence, token_idx, num_aug, depth, features):

    #token_sequence, size:[n, k+1, d]
    split_list = [1, token_sequence.shape[1]-1]
    split_tensor = torch.split(token_sequence, split_list, dim=1)
    
    # node itself
    node_self = split_tensor[0] #[n, 1, d]
    # nodes' tokens
    sampling_tokens = split_tensor[1] #[n, k, d]

    # print("token_idx:", token_idx)
    # output token
    final_aug_token = token_sequence  
    map_idx = token_idx

    for i in range(num_aug):
        new_idx = token_idx
        for j in range(depth):
            # random_token_idx (n, k)
            random_indices = torch.randint(0, token_sequence.shape[1]-1, (token_sequence.shape[0],token_sequence.shape[1]-1))

            # mix_token index (n,k)
            # update swap strategy
            new_indices = map_idx[new_idx.to(torch.long), random_indices.to(torch.long)]
            
            k = new_idx.shape[1]
            kk = int(1*k/2)
            selected_cols = torch.randperm(k)[:kk]
            for col in selected_cols:
                new_idx[:,col] = new_indices[:,col]

        # print("new_idx:",new_idx)

        aug_token = features[new_idx.to(torch.long)]

        temp_token = torch.cat((node_self, aug_token), dim=1)

        final_aug_token = torch.cat((final_aug_token, temp_token), dim=1)

    return final_aug_token # [n, (k+1)*(num_aug+1), d] 


def mix_token_a(token_sequence, token_idx, num_aug, depth, features):

    #token_sequence, size:[n, k+1, d]
    split_list = [1, token_sequence.shape[1]-1]
    split_tensor = torch.split(token_sequence, split_list, dim=1)
    
    # node itself
    node_self = split_tensor[0] #[n, 1, d]
    # nodes' tokens
    sampling_tokens = split_tensor[1] #[n, k, d]


    print("token_idx:", token_idx)
    # output token
    final_aug_token = token_sequence  
    #new_idx = torch.empty(token_idx.shape[0],token_idx.shape[1])

    for i in range(num_aug):
        new_idx = token_idx
        for j in range(depth):
            # random_token_idx (n, k)
            random_indices = torch.randint(0, token_sequence.shape[1]-1, (token_sequence.shape[0],token_sequence.shape[1]-1))

            # mix_token index (n,k)
            # update swap strategy
            new_indices = new_idx[new_idx.to(torch.long), random_indices.to(torch.long)]
            
            k = new_idx.shape[1]
            kk = int(1*k/2)
            selected_cols = torch.randperm(k)[:kk]
            for col in selected_cols:
                new_idx[:,col] = new_indices[:,col]

        print("new_idx:",new_idx)

        aug_token = features[new_idx.to(torch.long)]

        temp_token = torch.cat((node_self, aug_token), dim=1)

        final_aug_token = torch.cat((final_aug_token, temp_token), dim=1)

    return final_aug_token # [n, (k+1)*(num_aug+1), d] 



def node_seq_feature(features, pk, sample_batch, labels):

    nodes_features_p = torch.empty(features.shape[0], pk+1, features.shape[1])  # size= (N, K+1, d )

    # sampling_token_index
    idx_tokens = torch.empty(features.shape[0], pk)

    # label_index
    idx_labels = torch.empty(features.shape[0], pk+1)

    x = features + torch.zeros_like(features)

    # sample in the batch 
    total_batch = int(features.shape[0]/sample_batch)

    rest_batch = int(features.shape[0]%sample_batch)

    print("------------begin sampling------------")

    for index_batch in tqdm(range(total_batch)):

        # x_batch, [b,d]
        # x1_batch, [b,d]
        x_batch = x[(index_batch)*sample_batch:(index_batch+1)*sample_batch,:]

        s = torch.matmul(x_batch, x.transpose(1, 0))

        print(s.shape)
        for i in range(sample_batch):
            s[i][(index_batch)*(sample_batch) + i] = -1000      

        topk_values, topk_indices = torch.topk(s, pk, dim=1)

        for index in range(sample_batch):

            nodes_features_p[(index_batch)*sample_batch + index, 0, :] = features[(index_batch)*sample_batch + index]

            idx_labels[(index_batch)*sample_batch + index, 0] = labels[(index_batch)*sample_batch + index]

            idx_tokens[(index_batch)*sample_batch + index] = topk_indices[index]


            for i in range(pk):
                nodes_features_p[(index_batch)*sample_batch + index, i+1, :] = features[topk_indices[index][i]]

                idx_labels[(index_batch)*sample_batch + index, i+1] = labels[topk_indices[index][i]]


    if rest_batch > 0:

        x_batch = x[(total_batch)*sample_batch:(total_batch)*sample_batch + rest_batch,:]

        s = torch.matmul(x_batch, x.transpose(1, 0))

        for i in range(rest_batch):
            s[i][(total_batch)*sample_batch + i] = -1000    

        topk_values, topk_indices = torch.topk(s, pk, dim=1)

        for index in range(rest_batch):
            nodes_features_p[(total_batch)*sample_batch + index, 0, :] = features[(total_batch)*sample_batch + index]

            idx_labels[(total_batch)*sample_batch + index, 0] = labels[(total_batch)*sample_batch + index]

            idx_tokens[(total_batch)*sample_batch + index] = topk_indices[index]

            for i in range(pk):
                nodes_features_p[(total_batch)*sample_batch + index, i+1, :] = features[topk_indices[index][i]]

                idx_labels[(total_batch)*sample_batch + index, i+1] = labels[topk_indices[index][i]]


    return nodes_features_p, idx_tokens, idx_labels

