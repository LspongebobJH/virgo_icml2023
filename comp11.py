import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.loader import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from utils_pyg import init_layers

from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from torch_geometric.data import Batch
from sklearn.model_selection import StratifiedKFold

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, in_dim, hid_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            hid_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        if gnn_type == 'gin':
                self.convs.append(GINConv(
                    nn = torch.nn.Sequential(torch.nn.Linear(in_dim, hid_dim), 
                            torch.nn.BatchNorm1d(hid_dim), 
                            torch.nn.ReLU(), 
                            torch.nn.Linear(hid_dim, hid_dim))
                ))
        elif gnn_type == 'gcn':
            self.convs.append(GCNConv(in_dim, hid_dim))
        
        self.batch_norms.append(torch.nn.BatchNorm1d(hid_dim))

        for _ in range(num_layer-1):
            if gnn_type == 'gin':
                self.convs.append(GINConv(
                    nn = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim), 
                            torch.nn.BatchNorm1d(hid_dim), 
                            torch.nn.ReLU(), 
                            torch.nn.Linear(hid_dim, hid_dim))
                ))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hid_dim, hid_dim))

            self.batch_norms.append(torch.nn.BatchNorm1d(hid_dim))

    def forward(self, batched_data):
        x, edge_index = batched_data.x, batched_data.edge_index

        ### computing input node embedding

        h_list = [x]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
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
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

class GNN(torch.nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layer = 5,
                    gnn_type = 'gin', residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            out_dim (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNN_node(num_layer, in_dim, hid_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)

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
            self.graph_pred_linear = torch.nn.Linear(2*self.hid_dim, self.out_dim)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)

def train(model, device, loader, optimizer):
    model.train()
    if model.out_dim == 1:
        cls_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        cls_criterion = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            if model.out_dim == 1:
                loss = cls_criterion(pred.squeeze().to(torch.float32), batch.y.to(torch.float32))
            else:
                loss = cls_criterion(pred.to(torch.float32), batch.y)
            loss.backward()
            optimizer.step()

@torch.no_grad()
def eval(model, device, loader):
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
            
            if model.out_dim == 1:
                pred = pred.squeeze()
            y_true.append(batch.y.detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    return evaluate(y_pred, y_true)

def evaluate(logits:torch.tensor, labels:torch.tensor):
    if labels.max() == 1:
        acc = (logits >= 0.5).int().eq(labels).sum() / len(labels)
    else:
        acc = torch.argmax(logits, dim=-1).eq(labels).sum() / len(labels)
    return acc

def separate_data(dataset, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 0)

    labels = dataset.data.y
    idx_list = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, test_idx = idx_list[fold_idx]

    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]

    return train_dataset, test_dataset

def sample_g_est(dataset, num_g_samples):
    '''
    sample graphs for estimation
    '''
    if num_g_samples == -1:
        num_g_samples = len(dataset)
        sub_dataset = dataset
    else:
        rng = np.random.default_rng()
        sub_dataset = list(rng.choice(len(dataset), size=num_g_samples, replace=False))
        sub_dataset = dataset[sub_dataset]
    sub_dataset = Batch.from_data_list(sub_dataset)
    return sub_dataset
    
def pipe(gName, gnn, epochs, lr, init, num_g_samples, hid_dim, drop_ratio,
         num_workers, batch_size, fold_idx, device='cuda'):

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    dataset = TUDataset(root='/mnt/jiahanli/datasets', name=gName)
    max_degree = torch.max(torch.tensor([torch.max(degree(data.edge_index[0], data.num_nodes)) \
                    for data in dataset]))
    transform = OneHotDegree(max_degree.int().item())
    dataset = TUDataset(root='/mnt/jiahanli/datasets', name=gName, transform=transform)
    in_dim = dataset[0].x.shape[1]
    out_dim = dataset.data.y.max()+1
    if out_dim == 2:
        out_dim = 1
    train_dataset, test_dataset = separate_data(dataset, fold_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = num_workers)

    # in_dim, hid_dim, out_dim, num_layer = 5,
    # gnn_type = 'gin', residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"
    model = GNN(in_dim, hid_dim, out_dim, gnn_type=gnn, num_layer=5, drop_ratio=drop_ratio)
    model = model.to(device)
    batch_est = sample_g_est(train_dataset, num_g_samples)
    init_layers(batch_est, gnn, model.gnn_node.convs, init, hid_dim)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    test_curve = []
    train_curve = []

    for epoch in range(epochs):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader)
        test_perf = eval(model, device, test_loader)

        print(f"Train: {100 * train_perf:.2f}% | Test: {100 * test_perf:.2f}%")

        train_curve.append(train_perf)
        test_curve.append(test_perf)

        scheduler.step()

    print('------')
    print(f'The Last Train:{100 * train_curve[-1]:.2f}% | '
        f'Test:{100 * test_curve[-1]:.2f}%')

    return train_curve, test_curve

def run_test():
    searchSpace = {
            "gName": 'IMDB-BINARY',
            "epochs":350,
            "lr":1e-2,
            "gnn":'gin',
            "num_workers":4,
            "batch_size":32,
            "num_g_samples":-1,
            "fold_idx":0,
            "hid_dim":64,
            "drop_ratio":0.5,

    }

    print(searchSpace)
    pipe(**searchSpace)

if __name__ == "__main__":
    run_test()
    

    print(1)