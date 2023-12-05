import argparse
import torch as th
import dgl
import dgl.function as fn

import networkx as nx
import sys
sys.path.append('/home/ubuntu/virgo')

from dgl.data import *
from utils import *
from gcn import GCN

def corr_paths(g:dgl.DGLGraph, gcn, h_act):
    sampled_nodes = th.multinomial(th.ones(g.number_of_nodes()), 
            num_samples=args.nodes * 2, replacement=False).cpu()
    khop_adj = dgl.khop_adj(g, len(gcn.layers)).nonzero()
    
    # to obtain paths
    g_nx = g.cpu().to_networkx()
    k_paths = []
    for u in sampled_nodes:
        k_path_u = []
        for v in khop_adj[khop_adj[:, 0] == u, 1]:
            for _k_path in nx.all_simple_paths(g_nx, u.item(), v.item(), cutoff=len(gcn.layers)):
                if len(_k_path) == len(gcn.layers)+1:
                    k_path_u.append(_k_path)
            if len(k_path_u) > args.paths:
                break
        if len(k_path_u) < args.paths:
            continue
        k_paths.append(th.tensor(k_path_u)[:args.paths])
        if len(k_paths) >= args.nodes:
            break
    k_paths = th.stack(k_paths, dim=0)

    # to obtain observations of sigma
    res = 1
    sigma_list = []
    for i, _h_act in enumerate(h_act[:-1]):
        sel_nodes = _h_act[k_paths[:, :, i+1]]
        res = res * sel_nodes
        sigma = (res > 0).int() 
        sigma_list.append(sigma)
    sigma = th.stack(sigma_list, dim=0).to(g.device) # observations of random variables sigma at each layer

    # to obtain observations of paths of different nodes
    res = 1
    path_list = []
    for i, (layer, _sigma) in enumerate(zip(gcn.layers[:-1], sigma)):
        # _sigma.shape = [i, j], where i is the number of paths, j is the number of observations
        index = th.multinomial(th.ones(layer.weight.shape[0] * layer.weight.shape[1]), 
                num_samples=_sigma.shape[-1], replacement=False)
        choices = th.flatten(layer.weight.detach())[index] # to sample observations of w
        res = res * choices # observations of product of w, for covariance of w and sigma
        _res = res.unsqueeze(0).unsqueeze(0)
        path_list.append(_res * _sigma) # observations of different paths, for covariance between paths
    path = th.stack(path_list, dim=0).to(g.device)

    # to obtain observations of paths
    path_list = []
    for _ in range(100):
        index = th.multinomial(th.ones(args.hid_dim), num_samples=int(args.hid_dim / 2), 
                replacement=False)
        path_list.append(path[:, :, :, index].sum(dim=-1))
    path = th.stack(path_list, dim=3).cpu()

    # to obtain corr between different paths
    corr_list = []
    for l in range(path.shape[0]):
        _corr_list = []
        for n in range(path.shape[1]):
            _corr_list.append(th.corrcoef(path[l, n]))
        corr_list.append(th.stack(_corr_list, dim=0))
    corr = th.stack(corr_list, dim=0)
    num_nodes = corr.shape[1]
    num_paths = corr.shape[2]
    return corr, num_nodes, num_paths, 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--hid-dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--nodes', type=int, default=10)
    parser.add_argument('--paths', type=int, default=3)
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
    corr, num_nodes, num_paths, times = corr_paths(g, gcn, h_act_list)

    _mean = list(corr.mean(dim=(1, 2, 3)).numpy())
    _mean.reverse()
    _std = list(corr.std(dim=(2, 3)).mean(dim=-1).numpy())
    _std.reverse()

    print(f"Dataset {args.dataset} | Layers {args.l} | Nodes {num_nodes} | Paths {num_paths} | Times {times}")
    print(f"Mean {_mean} | "
          f"Std {_std}")
    print(1)
    