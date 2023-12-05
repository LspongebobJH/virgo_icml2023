'''
proteins
'''
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from utils import EarlyStopping, init_layers, load_dataset

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(GCN, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(
            GCNConv(in_dim, hid_dim, normalize=False))
        for _ in range(n_layers - 2):
            self.layers.append(
                GCNConv(hid_dim, hid_dim, normalize=False))
        self.layers.append(
            GCNConv(hid_dim, out_dim, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        for layer in self.layers[:-1]:
            x = layer(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj_t)
        return x

def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

def pipe(mean_method, model_name, hid_dim, l, lr, epochs, dropout, wd, device='cuda', init='virgo'):
    dataset = PygNodePropPredDataset(
        name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'),
        root="/mnt/jiahanli/datasets")

    data = dataset[0]
    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    # Pre-compute GCN normalization.
    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model = GCN(data.num_features, hid_dim,
                    112, l, dropout=dropout)
    # initialize model using dgl-implemented method and datasets
    g, _ = load_dataset('proteins', device=device) # g here is only used for calculating C2
    if init == 'virgoback':
        init_layers(g, model_name, model.layers, init, num_classes=_.num_classes, pyg=True)
    elif init == 'virgo':
        init_layers(g, model_name, model.layers, init, mean_method=mean_method, num_classes=_.num_classes, pyg=True)
    else:
        init_layers(g, model_name, model.layers, init, pyg=True)

    del g
    torch.cuda.empty_cache()

    data, model = data.to(device), model.to(device)
    evaluator = Evaluator(name='ogbn-proteins')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    train_rocauc_list, valid_rocauc_list, test_rocauc_list = [], [], []
    best_valid_rocauc, final_test_rocauc = 0.0, 0.0
    for epoch in range(epochs):
        loss = train(model, data, train_idx, optimizer)
        result = test(model, data, split_idx, evaluator)

        train_rocauc, valid_rocauc, test_rocauc = result
        if valid_rocauc > best_valid_rocauc:
            best_valid_rocauc = valid_rocauc
            final_test_rocauc = test_rocauc
        print(
            f'Epoch: {epoch:02d} | '
            f'Loss: {loss:.4f} | '
            f'Train: {100 * train_rocauc:.2f}% | '
            f'Valid: {100 * valid_rocauc:.2f}% | '
            f'Test: {100 * test_rocauc:.2f}% | ')
        train_rocauc_list.append(train_rocauc)
        valid_rocauc_list.append(valid_rocauc)
        test_rocauc_list.append(test_rocauc)
    print('============================')
    print(f'Final Test: {final_test_rocauc:4f}')
    test_rocauc_list.append(final_test_rocauc)
    return train_rocauc_list, valid_rocauc_list, test_rocauc_list

def run_test():
    searchSpace = {
                "mean_method": 'harmonic',
                "hid_dim": 256,
                "model_name": "gcn",
                "l": 3,
                "lr": 1e-2,
                "epochs": 2,
                "dropout": 0.5,
                "wd": 0.0,
            }
    print(searchSpace)
    pipe(**searchSpace)

if __name__ == '__main__':
    run_test()

    print(1)