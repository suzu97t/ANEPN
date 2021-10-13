
import itertools
import time
import torch as t
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import math
import scipy.io as sio
import sys
import scipy
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import calinski_harabasz_score
import torch.nn as nn
#from clustering_metric import *
from utils import *
from layer import *
#from linkpred import *
import argparse
#from sklearn.metrics import calinski_harabasz_score
import warnings
import random

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1234,  
                    help='random seed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--h', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--n_epo', type=int, default=500,  
                    help='max epoch')
parser.add_argument('--pre_epo', type=int, default=50,  
                    help='pre-train epoch')                    
parser.add_argument('--patience', type=int, default=10,  
                    help='patience for training stop')                    
parser.add_argument('--dataname', type=str, default="cora", choices=['cora','citeseer','pubmed','Blog','flickr','wiki'],  
                    help='name of dataset')                    
parser.add_argument('--label_rate', type=float, default=0.01,
                    help='label rate of training data')
parser.add_argument('--mu', type=float, default=1.0,
                    help='Margin parameter')

args = parser.parse_args()
if t.cuda.is_available(): 
  device = 'cuda'
else:
  device = 'cpu'

if args.seed == 777:
  random.seed()
  args.seed = random.randint(0,100000)


random.seed(args.seed)
np.random.seed(args.seed)
t.manual_seed(args.seed)
t.cuda.manual_seed_all(args.seed)

# datasets= ['cora','citeseer','pubmed','Blog','flickr','wiki','cora_full','dblp','cs','physics','amazon_photo','amazon_computers']
# dataname = datasets[args.idx_data]
features, label, L_pos, L_neg, idx_train, idx_test, M, C = pre_process(args.dataname, args.label_rate, device)


net = ANEPN(M, args.h, C)
net.to(device)
optimizer=optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True) 
# optimizer=optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd) 

net.train()
CELoss = nn.NLLLoss()

score = 0.0
patience_count = 0
patience = 10
pre_score = 0.0
score_list = np.zeros(args.n_epo)
alpha = 0.0
start_time=time.time()
for epoch in range(args.n_epo):

    optimizer.zero_grad()
    pred, Z =net(features)
    L_ce = CELoss(pred[idx_train], label[idx_train])
 
  
    if epoch == 0 :
      pre_score = score
    if epoch >= args.pre_epo-1:
      pred_labels = t.argmax(pred,axis=1).to('cpu').detach().numpy().copy()
      if len(np.unique(pred_labels)) != C:
        score = pre_score 
      else:
        score = calinski_harabasz_score(Z.to('cpu').detach().numpy().copy(), pred_labels)#.to('cpu').detach().numpy().copy())
        #print(score)
    if epoch %10 == 0 and epoch >= args.pre_epo:
      alpha += 0.05 

    if  epoch >= args.pre_epo :
      if score_list[epoch-1] > score:
        patience_count += 1   
     
 
    if epoch >= args.pre_epo:
      L_sm =  t.trace(t.matmul(t.sparse.mm(L_pos, Z).T ,Z))/args.h
      L_asm =  t.trace(t.matmul(t.sparse.mm(L_neg, Z).T ,Z))/args.h
      loss = L_ce + alpha*L_sm + alpha*F.relu(args.mu - L_asm)
    else:
      loss = L_ce
  
    score_list[epoch] = score
   
    loss.backward()
    
    optimizer.step()

    if patience == patience_count:
      break
      #print(epoch,"#####################")
# print("training time: ",time.time()-start_time)
net.eval()
# np.save("score_list.npy",np.array(score_list))
pred, Z =net(features)
print(accuracy(pred[idx_test], label[idx_test]).item())
del net, features, label, L_pos, L_neg, pred, Z