import torch.nn as nn
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.init import uniform_
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import numpy as np

class GCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

class Aggr(nn.Module):
    def __init__(self, l):
        super(Aggr, self).__init__()
        self.l = l
        self.conv_layers = nn.ModuleList()
        for _ in range(self.l):
            self.conv_layers.append(GCNConv(1, 1, bias=False))

    def forward(self, data, h):
        c5_list = []        
        edge_index = data.edge_index
        for i in range(self.l):
            h = self.conv_layers[i](h, edge_index)
            c5_list.append((h ** 2).sum().item())
        return c5_list

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

def gen_c2_for(data, m_list, l):
    '''
    generate c2_for for all layers
    m_list == [in_dim, hid_dim, out_dim]
    '''
    device = data.x.device
    c2_list = []
    c2_list.append(2 / m_list[0])
    aggr = Aggr(l)
    aggr = aggr.to(device)
    aggr_list = aggr(data, h=data.x.mean(dim=1).unsqueeze(dim=1))
    for i in range(1, l):
        c5L_1, c5L = aggr_list[i-1], aggr_list[i]
        c2 = 2 / m_list[1] * c5L_1 / c5L
        c2_list.append(c2) # the c2 of the 1, 2, 3, ... layers
    return c2_list

def gen_c2_back(data, m_list, l, num_classes):
    device = data.x.device
    c2_list = []
    aggr = Aggr(l)
    aggr = aggr.to(device)
    aggr_list = aggr(data, torch.ones(data.x.shape[0], 1).to(device))
    c2_list.append(data.x.shape[0] / ((num_classes-1) * aggr_list[0]))
    for i in range(1, l):
        c5L_1, c5L = aggr_list[i-1], aggr_list[i]
        c2 = 2 / m_list[1] * c5L_1 / c5L
        c2_list.append(c2)
    c2_list.reverse()
    return c2_list

def init_layers(data, model_name, layers, init_name, num_classes=None):
    if model_name == 'gcn':
        m_list = [layers[0].lin.weight.shape[1], layers[0].lin.weight.shape[0], layers[-1].lin.weight.shape[0]]    
    elif model_name == 'graphsage':
        m_list = [layers[0].lin_l.weight.shape[1], layers[0].lin_l.weight.shape[0], layers[-1].lin_l.weight.shape[0]]   
    elif model_name == 'gin':
        m_list = [layers[0].nn[0].weight.shape[1], layers[0].nn[0].weight.shape[0], layers[-1].nn[0].weight.shape[0]]
    elif model_name == 'gat':
        m_list = [layers[0].lin_src.weight.shape[1], layers[0].lin_src.weight.shape[0], layers[-1].lin_src.weight.shape[0]]

    if init_name == 'virgofor':
        a_list = to_uni_a(gen_c2_for(data, m_list, len(layers)))
    elif init_name == 'virgoback':
        a_list = to_uni_a(gen_c2_back(data, m_list, len(layers), num_classes))
    
    for i, layer in enumerate(layers):
        if model_name == "gcn":
            uniform_(layer.lin.weight, a=-a_list[i], b=a_list[i])    
        elif model_name == 'sage':
            uniform_(layer.lin_l.weight, a=-a_list[i], b=a_list[i])
            uniform_(layer.lin_r.weight, a=-a_list[i], b=a_list[i])    
        elif model_name == 'gin':
            uniform_(layer.nn[0].weight, a=-a_list[i], b=a_list[i])
            uniform_(layer.nn[-1].weight, a=-a_list[i], b=a_list[i])
        elif model_name == 'gat':
            uniform_(layer.lin_src.weight, a=-a_list[i], b=a_list[i])
            uniform_(layer.lin_dst.weight, a=-a_list[i], b=a_list[i])