import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


from utils import *

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, hid_dim):
        '''
            hid_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(hid_dim, 2*hid_dim), torch.nn.BatchNorm1d(2*hid_dim), torch.nn.ReLU(), torch.nn.Linear(2*hid_dim, hid_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = torch.nn.Linear(7, hid_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, hid_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.lin = torch.nn.Linear(hid_dim, hid_dim)
        self.root_emb = torch.nn.Embedding(1, hid_dim)
        self.edge_encoder = torch.nn.Linear(7, hid_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, n_layers, hid_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            hid_dim (int): node embedding dimensionality
            n_layers (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.n_layers = n_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.n_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, hid_dim) # uniform input node embedding

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(n_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(hid_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hid_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(hid_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding

        h_list = [self.node_encoder(x)]
        for layer in range(self.n_layers):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.n_layers - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.n_layers + 1):
                node_representation += h_list[layer]

        return node_representation

class GNN(torch.nn.Module):

    def __init__(self, num_class, n_layers = 5, hid_dim = 300, 
                    gnn_type = 'gin', residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(GNN, self).__init__()

        self.n_layers = n_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.hid_dim = hid_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.n_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNN_node(n_layers, hid_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(hid_dim, 2*hid_dim), torch.nn.BatchNorm1d(2*hid_dim), torch.nn.ReLU(), torch.nn.Linear(2*hid_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(hid_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.hid_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.hid_dim, self.num_class)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)

multicls_criterion = torch.nn.CrossEntropyLoss()

def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def pipe(gnn, epochs, lr, init, num_g_samples,
         num_workers, batch_size, runs, device='cuda'):

    dataset = PygGraphPropPredDataset(name='ogbg-ppa', transform=add_zeros, root='/mnt/jiahanli/datasets')
    if init in ['virgo', 'virgofor', 'virgoback']:
        g, _ = load_dataset('ogbg-ppa', num_samples=num_g_samples, rand_feat=False, drop_feat=False, device=device) # g here is only used for calculating C2
        # g, _ = load_dataset('molhiv', num_samples=num_g_samples, rand_feat=False, drop_feat=False, device=device)

    split_idx = dataset.get_idx_split()
    evaluator = Evaluator('ogbg-ppa')

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, num_workers = num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False, num_workers = num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False, num_workers = num_workers)

    train_curve_all, valid_curve_all, test_curve_all = [], [], []
    a_list = None

    for run in range(runs):
        model = GNN(gnn_type = gnn, num_class = dataset.num_classes, n_layers = 5, hid_dim = 300, drop_ratio = 0.5).to(device)
        if init in ['virgo', 'virgofor', 'virgoback']:
            if a_list is not None:
                for i, layer in enumerate(model.gnn_node.convs):
                    if gnn == "gcn":
                        uniform_(layer.lin.weight, a=-a_list[i], b=a_list[i])
                    elif gnn == 'gin':
                        uniform_(layer.mlp[0].weight, a=-a_list[i], b=a_list[i])
                        uniform_(layer.mlp[-1].weight, a=-a_list[i], b=a_list[i])
            else: 
                a_list = init_layers(g, gnn, model.gnn_node.convs, init, pyg=True)
                del g

        optimizer = optim.Adam(model.parameters(), lr=lr)

        valid_curve = []
        test_curve = []
        train_curve = []

        for epoch in range(epochs):
            print("=====Epoch {}, Run {}".format(epoch, run))
            print('Training...')
            train(model, device, train_loader, optimizer)

            print('Evaluating...')
            train_perf = eval(model, device, train_loader, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

            print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

        best_val_epoch = np.argmax(np.array(valid_curve))
        
        print('------')
        print(f'Final Train:{100 * train_curve[best_val_epoch]:.2f}% | '
            f'Valid:{100 * valid_curve[best_val_epoch]:.2f}% | '
            f'Test:{100 * test_curve[best_val_epoch]:.2f}%')
        train_curve.append(train_curve[best_val_epoch])
        valid_curve.append(valid_curve[best_val_epoch])
        test_curve.append(test_curve[best_val_epoch])

        train_curve_all.append(train_curve)
        valid_curve_all.append(valid_curve)
        test_curve_all.append(test_curve)
    return train_curve_all, valid_curve_all, test_curve_all

def run_test():
    searchSpace = {
            "gnn": 'gcn',
            "epochs":1,
            "batch_size":32,
            "num_g_samples":10,
            "num_workers":0,
            "lr":0.001,
            "init":'virgofor',
            "runs":2
    }

    print(searchSpace)
    pipe(**searchSpace)

if __name__ == "__main__":
    run_test()
    print(1)