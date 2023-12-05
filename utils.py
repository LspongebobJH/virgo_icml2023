from math import floor
from statistics import harmonic_mean
from wsgiref import handlers
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import dgl
import dgl.function as fn
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
from copy import deepcopy
import random

from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.graphproppred import DglGraphPropPredDataset
from dgl.utils import expand_as_pair
from dgl.base import DGLError
from dgl.data import *
from dgl.transforms import AddSelfLoop
from dgl.nn.pytorch import GraphConv
from torch.nn.init import uniform_, normal_

from copy import deepcopy
from operator import itemgetter

class Aggr(nn.Module):
    def __init__(self, l):
        super(Aggr, self).__init__()
        self.l = l
        self.conv_layers = nn.ModuleList()
        for _ in range(self.l):
            self.conv_layers.append(GraphConv(1, 1, weight=False, bias=False))

    def forward(self, g: dgl.DGLGraph, h: th.Tensor):
        c5_list = []
        l = self.l
        for i in range(l):
            h = self.conv_layers[i](g, h)
            c5_list.append((h ** 2).sum().item())
        return c5_list

def intersection(tensor_list, dim=0):
    tensor1 = tensor_list[0]
    for i in range(1, len(tensor_list)):
        tensor2 = tensor_list[i]
        combined = th.cat([tensor1, tensor2])
        uniques, counts = combined.unique(dim=dim, return_counts=True)
        tensor1 = uniques[counts > 1]
    return tensor1

def difference(tensor_list, dim=0):
    tensor1 = tensor_list[0]
    for i in range(1, len(tensor_list)):
        tensor2 = tensor_list[i]
        combined = th.cat([tensor1, tensor2])
        uniques, counts = combined.unique(dim=dim, return_counts=True)
        tensor1 = uniques[counts == 1]
    return tensor1

def to_uni_a(var_list):
    a_list = []
    for var in var_list:
        a_list.append(np.sqrt(3 * var))
    return a_list

def to_var(a_list):
    var_list = []
    for a in a_list:
        var_list.append(a**2 / 3)
    return var_list

def alpha_divergence(hL_1:th.FloatTensor, hL:th.FloatTensor, var=False):
    '''
    hL_1 is the true distribution, hL is the estimated distribution
    '''
    if var is True:
        hL_1Var, hLVar = hL_1, hL
    else:
        hL_1Var, hLVar = hL_1.var(dim=1), hL.var(dim=1)
    res = hLVar - hL_1Var - hL_1Var * th.log(hLVar / hL_1Var)
    res = res[~res.isnan()]
    return res.mean().item()

def entropy(h:th.FloatTensor, var=False):
    if var:
        hVar = h
    else:
        hVar = h.var(dim=1)
    
    return (-1 * hVar * th.log(hVar)).mean().item()

def gen_c2_for(g:dgl.DGLGraph, m_list, l):
    '''
    generate c2_for for all layers
    m_list == [in_dim, hid_dim, out_dim]
    '''
    device = g.device
    c2_list = []
    c2_list.append(2 / m_list[0])
    aggr = Aggr(l)
    aggr = aggr.to(device)
    aggr_list = aggr(g, g.ndata['feat'].mean(dim=1))
    for i in range(1, l):
        c5L_1, c5L = aggr_list[i-1], aggr_list[i]
        c2 = 2 / m_list[1] * c5L_1 / c5L
        c2_list.append(c2) # the c2 of the 1, 2, 3, ... layers
    return c2_list

def gen_c2_back(g:dgl.DGLGraph, m_list, l, num_classes):
    device = g.device
    c2_list = []
    aggr = Aggr(l)
    aggr = aggr.to(device)
    aggr_list = aggr(g, th.ones(g.num_nodes(), 1).to(g.device))
    c2_list.append(g.num_nodes() / ((num_classes-1) * aggr_list[0]))
    for i in range(1, l):
        c5L_1, c5L = aggr_list[i-1], aggr_list[i]
        c2 = 2 / m_list[1] * c5L_1 / c5L
        c2_list.append(c2)
    c2_list.reverse()
    return c2_list

def gen_c2(g:dgl.DGLGraph, m_list, l, num_classes, mean_method):
    c2For_list = np.array(gen_c2_for(g, m_list, l))
    c2Back_list = np.array(gen_c2_back(g, m_list, l, num_classes))
    if mean_method == 'harmonic':
        return 2 / (1 / c2For_list + 1 / c2Back_list)
    elif mean_method == 'normal':
        return (c2For_list + c2Back_list) / 2

def init_layers(g:dgl.DGLGraph, model_name, layers, init_name, mean_method='harmonic', num_classes=None, pyg=False):
    if pyg:
        if model_name == 'gcn':
            m_list = [layers[0].lin.weight.shape[1], layers[0].lin.weight.shape[0], layers[-1].lin.weight.shape[0]]    
        elif model_name == 'sage':
            m_list = [layers[0].lin_l.weight.shape[1], layers[0].lin_l.weight.shape[0], layers[-1].lin_l.weight.shape[0]]   
        elif model_name == 'gin':
            m_list = [layers[0].mlp[0].weight.shape[1], layers[0].mlp[0].weight.shape[0], layers[-1].mlp[0].weight.shape[0]]
    else:
        m_list = [layers[0].weight.shape[0], layers[0].weight.shape[1], layers[-1].weight.shape[1]]
    numLayers = len(layers)

    if init_name == 'virgofor':
        a_list = to_uni_a(gen_c2_for(g, m_list, len(layers)))
    elif init_name == 'virgoback':
        a_list = to_uni_a(gen_c2_back(g, m_list, len(layers), num_classes))
    elif init_name == 'virgo':
        a_list = to_uni_a(gen_c2(g, m_list, len(layers), num_classes, mean_method))
    elif init_name == 'kaifor':
        for layer in layers:
            if pyg:
                init.kaiming_uniform_(layer.lin.weight, mode='fan_in', nonlinearity='relu')
            else:
                init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        return
    elif init_name == 'kaiback':
        for layer in layers:
            if pyg:
                init.kaiming_uniform_(layer.lin.weight, mode='fan_out', nonlinearity='relu')
            else:
                init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
        return
    else:
        for layer in layers:
            if pyg:
                init.xavier_uniform_(layer.lin.weight)
            else:
                init.xavier_uniform_(layer.weight)
        return
    for i, layer in enumerate(layers):
        if pyg:
            if model_name == "gcn":
                uniform_(layer.lin.weight, a=-a_list[i], b=a_list[i])    
            elif model_name == 'sage':
                uniform_(layer.lin_l.weight, a=-a_list[i], b=a_list[i])
                uniform_(layer.lin_r.weight, a=-a_list[i], b=a_list[i])    
            elif model_name == 'gin':
                uniform_(layer.mlp[0].weight, a=-a_list[i], b=a_list[i])
                uniform_(layer.mlp[-1].weight, a=-a_list[i], b=a_list[i])
        else:
            uniform_(layer.weight, a=-a_list[i], b=a_list[i])
    return a_list

def load_dataset(gName, rand_feat=False, drop_feat=False, num_samples=-1, device='cuda'):
    if gName in ['arxiv', 'proteins', 'products']:
        data = DglNodePropPredDataset('ogbn-'+gName, root="/mnt/jiahanli/datasets")
        g, _ = data[0]
        transform = AddSelfLoop()
        g = transform(g)
        if gName == 'proteins':
            g.update_all(fn.copy_e('feat', 'm'), fn.mean('m', 'feat'))
            g.edata.pop('feat')
    elif gName in ['collab', 'ppa', 'ddi', 'citation2']:
        data = DglLinkPropPredDataset('ogbl-'+gName, root="/mnt/jiahanli/datasets")
        g = data[0]
        transform = AddSelfLoop()
        g = transform(g)
        if gName == 'ppa':
            g.ndata['feat'] = g.ndata['feat'].float()
        if gName == 'ddi':
            emb = th.nn.Embedding(g.num_nodes(),
                             256).to(device)
            th.nn.init.xavier_uniform_(emb.weight)
            g.ndata['feat'] = emb.weight.to(g.device)
    elif gName in ['molhiv', 'molpcba', 'ogbg-ppa']:
        if gName == 'ogbg-ppa':
            data = DglGraphPropPredDataset(gName, root="/mnt/jiahanli/datasets")
        else:
            data = DglGraphPropPredDataset('ogbg-'+gName, root="/mnt/jiahanli/datasets")
        print("...start batching graphs...")
        if num_samples == -1:
            num_samples = len(data)
            sub_dataset = data.graphs
        else:
            rng = np.random.default_rng()
            sub_dataset = list(rng.choice(len(data), size=num_samples, replace=False))
            sub_dataset = itemgetter(*sub_dataset)(data.graphs)
        g = dgl.batch(sub_dataset)
        if gName == 'ogbg-ppa':
            g.ndata['feat'] = th.empty(size=(g.num_nodes(), 1))
            normal_(g.ndata['feat'])
        g.ndata['feat'] = g.ndata['feat'].float()
        print("...end batching graphs...")
        transform = AddSelfLoop()
        g = transform(g)
    else:
        if gName == 'cora':
            data = CoraGraphDataset(transform=AddSelfLoop(), verbose=False, raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'citeseer':
            data = CiteseerGraphDataset(transform=AddSelfLoop(), verbose=False, raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'pubmed':
            data = PubmedGraphDataset(transform=AddSelfLoop(), verbose=False, raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'coraFull':
            data = CoraFullDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'aifb':
            data = AIFBDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'mutag':
            data = MUTAGDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'bgs':
            data = BGSDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'am':
            data = AMDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'computer':
            data = AmazonCoBuyComputerDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'photo':
            data = AmazonCoBuyPhotoDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'cs':
            data = CoauthorCSDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'physics':
            data = CoauthorPhysicsDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'wikics':
            data = WikiCSDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'flickr':
            data = FlickrDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'yelp':
            data = YelpDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        elif gName == 'reddit':
            data = RedditDataset(transform=AddSelfLoop(), raw_dir="/mnt/jiahanli/datasets")
        g = data[0]

    if rand_feat and drop_feat:
        raise Exception(f"rand_feat and drop_feat cannot be True at the same time")

    if rand_feat:
        if gName == 'proteins':
            uniform_(g.edata['feat'], -1, 1)
            g.update_all(fn.copy_e('feat', 'm'), fn.mean('m', 'feat'))
            g.edata.pop('feat')
        else:
            uniform_(g.ndata['feat'], -1, 1)

    if drop_feat:
        key_list = list(g.ndata.keys())
        for key in key_list:
            g.ndata.pop(key)
        key_list = list(g.edata.keys())
        for key in key_list:
            g.edata.pop(key)

    g = g.to(device)

    return g, data

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def step_score(self, score, model, save=True):  # test score
        if self.best_score is None:
            self.best_score = score
            if save:
                self.save_model(model)
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if save:
                self.save_model(model)
            self.best_score = np.max((score, self.best_score))
            self.counter = 0
            self.early_stop = False

    def save_model(self, model):
        model.eval()
        self.best_model = deepcopy(model.state_dict())

    def load_model(self, model):
        model.load_state_dict(self.best_model)

def conf_int(value, nSamples, CI=0.95):
    if CI == 0.95:
        z = 1.96
    return z * np.sqrt(1 / nSamples * value * (1 - value))

if __name__ == '__main__':
    g, data = load_dataset('arxiv')
    m_list = [g.ndata['feat'].shape[1], 256, 10]
    c2for_list = gen_c2_for(g, m_list, 5)
    afor_list = to_uni_a(c2for_list)
    print(1)