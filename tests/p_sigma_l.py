import argparse
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import pickle
import time
import os

import networkx as nx
import sys
sys.path.append('/home/ubuntu/virgo')
from utils import *
from gcn import GCN

def p_sigma_l(g:dgl.DGLGraph, gcn, h_act):
    g = g.cpu()
    khop_adj = dgl.khop_adj(g, len(gcn.layers)).nonzero()
    total = min(2 ** 24, khop_adj.shape[0])
    sampled_k_paths = th.multinomial(th.ones(total), num_samples=100, replacement=False)
    sampled_k_paths = khop_adj[sampled_k_paths].cpu().numpy()
    
    # to obtain observations of sigma
    g_nx = g.to_networkx()
    k_paths = []
    for u, v in sampled_k_paths:
        for _k_path in nx.all_simple_paths(g_nx, u, v, cutoff=len(gcn.layers)):
            if len(_k_path) == len(gcn.layers)+1:
                k_paths.append(_k_path)
        if len(k_paths) > 2000:
            break
    k_paths = np.array(k_paths)
    res = 1
    sigma_list = []
    for i, _h_act in enumerate(h_act[:-1]):
        sel_nodes = _h_act[k_paths[:, i+1]]
        res = res * sel_nodes
        sigma = (res > 0).int() 
        sigma_list.append(sigma)
    sigma = th.stack(sigma_list, dim=0).to(g.device) # observations of random variables sigma at each layer
    p_sigma = (sigma > 0).sum(dim=-1) / sigma.shape[-1]
    num_paths = len(k_paths)
    return list(p_sigma.mean(dim=-1).numpy()), \
            list(p_sigma.std(dim=-1).numpy()), \
            num_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--hid-dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--l', type=int, default=4)
    args = parser.parse_args()

    g, data = load_dataset(args.dataset, rand_feat=False, drop_feat=False, device=args.device)
    if args.dataset == 'proteins':
        g.update_all(fn.copy_e('feat', 'm'), fn.mean('m', 'feat'))
        g.edata.pop('feat')
        g.ndata.pop('species')
    feat = g.ndata['feat']

    if args.dataset == 'proteins':
        m_list = [feat.shape[1], args.hid_dim, 112]
    else:
        m_list = [feat.shape[1], args.hid_dim, data.num_classes]

    gcn = GCN(args.l, m_list[0], m_list[1], m_list[-1]).to(args.device)
    _, h_list, h_act_list = gcn(g, feat)
    p_sigma_mean, p_sigma_std, num_paths = p_sigma_l(g, gcn, h_act_list)

    print(f"Dataset {args.dataset} | Layers {args.l} | Paths {num_paths}")
    print(f"p_sigma_mean {p_sigma_mean} | p_sigma_std {p_sigma_std}")
    print(1)
    