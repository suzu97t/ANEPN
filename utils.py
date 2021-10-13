import time
import itertools
import torch as t
import torch.nn.functional as F
import math
import scipy.io as sio
import sys
import scipy
import numpy as np
import scipy.sparse as sp
from keras.utils import np_utils
from distutils.util import strtobool
from sklearn import preprocessing
import random
from scipy.sparse import csr_matrix, coo_matrix, tril
from sklearn.model_selection import train_test_split



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = t.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = t.from_numpy(sparse_mx.data)
    shape = t.Size(sparse_mx.shape)
    return t.sparse.FloatTensor(indices, values, shape)



def split_data(labels, labels_per_class):
    num_nodes = len(labels)
    all_data, all_class = np.arange(num_nodes).astype(int), np.unique(labels)
    idx_train, idx_test = [], []
    
    for c in all_class:
      idx_train = np.hstack([idx_train,random.sample(list(np.where(labels==c)[0].astype(int)), labels_per_class)])
      
    others = np.delete(all_data.astype(int), idx_train.astype(int))
    idx_test = np.random.choice(others, 1000, replace= False)
   
    return idx_train, idx_test


def negative_edge_sampling(A):
    A = A.tocsr()
    N = A.shape[0]
    E = A.nnz/2
    S =  A + A.dot(A) 
    t_mat = tril(np.tri(N),format='csr')
    S += t_mat
    del t_mat
    neg_indices = np.where(S.toarray()==0)
    del S
    sample_idx = np.random.choice(np.arange(len(neg_indices[0])), size = int(2*E), replace=False)
    neg_idx_row = neg_indices[0][sample_idx]
    neg_idx_col = neg_indices[1][sample_idx]

    A_neg = csr_matrix((np.ones(2*int(2*E)), (np.append(neg_idx_row,neg_idx_col), np.append(neg_idx_col,neg_idx_row))), shape=(N, N))
    del neg_indices,  neg_idx_row, neg_idx_col, sample_idx
    return A_neg 

def graph_conv(G, X, k):
  G_k = G
  for i in range(k-1):
    G_k = G_k.dot(G)
  return G_k.dot(X)  


def train_test(N, y, label_rate):
  dum = np.arange(N)
  indices = np.arange(N)
  X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(dum, y, indices, test_size=1-label_rate)
  idx_test =np.random.choice(idx_test, 1000, replace=False)
  return idx_train, idx_test

def pre_process(dataname, label_rate, device):
  
    A = sio.mmread("./network_datasets/network_datasets/"+dataname+"_A.mtx")
    X = sio.mmread("./network_datasets/network_datasets/"+dataname+"_X.mtx")
    y = np.load("./network_datasets/network_datasets/"+dataname+"_y.npy")

    N, M = X.shape[0], X.shape[1]
    C = len(np.unique(y))
    
    adj = A.tocsr()
    I = sp.identity(adj.shape[0], dtype='int8', format='csr')
    
    rowsum = np.array(adj.sum(1),dtype='float16')
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    d = sp.diags(r_inv).tocsr()
    A_norm = d.dot(adj)
    A_norm = A_norm.dot(d)

    adj = adj + I
    rowsum = np.array(adj.sum(1),dtype='float16')
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    d = sp.diags(r_inv).tocsr()
    A_til = d.dot(adj)
    A_til = A_til.dot(d)
    
    A_neg = negative_edge_sampling(adj)
   
    L_pos = I - A_norm
    
    G = I - 0.5*L_pos
    
    X_bar = graph_conv(G,X,2)
    
    
    rowsum = np.array(A_neg.sum(1),dtype='float16')
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    d = sp.diags(r_inv).tocsr()
    A_neg_norm = d.dot(A_neg).dot(d)

    L_neg = I - A_neg_norm
    
    features = t.FloatTensor(X_bar.toarray()).to(device)
    label = t.LongTensor(y).to(device)
    L_pos = sparse_mx_to_torch_sparse_tensor(L_pos).to(device)
    L_neg = sparse_mx_to_torch_sparse_tensor(L_neg).to(device)
   
    labels_per_class = math.ceil(math.ceil(N*label_rate)/C)
    idx_train, idx_test = split_data(y, labels_per_class)
    # idx_train, idx_test = train_test(N, y, label_rate)
    del A_norm, A_neg, A_til, A_neg_norm, X_bar, X
    return features, label, L_pos, L_neg, idx_train, idx_test, M , C
