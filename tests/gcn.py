import torch.nn as nn

from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, n_layers, in_dim, hid_dim, out_dim):
        super().__init__()
        assert n_layers >= 2
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=None))
        
        for _ in range(n_layers-2):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=None))
        self.layers.append(GraphConv(hid_dim, out_dim, activation=None))

        self.act = F.relu

    def forward(self, g, feat):
        h_list, h_act_list = [], []
        h = feat
        h_list.append(h.detach().cpu())
        h_act_list.append(h.detach().cpu())
        for layer in self.layers:
            h = layer(g, h)
            h_list.append(h.detach().cpu())
            h = self.act(h)
            h_act_list.append(h.detach().cpu())
        return h, h_list, h_act_list