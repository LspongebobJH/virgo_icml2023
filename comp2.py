'''
arxiv
'''

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from utils import EarlyStopping, init_layers, load_dataset

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(GCN, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hid_dim, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hid_dim))
        for _ in range(n_layers - 2):
            self.layers.append(
                GCNConv(hid_dim, hid_dim, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hid_dim))
        self.layers.append(GCNConv(hid_dim, out_dim, cached=True))

        self.dropout = dropout

    def forward(self, x, adj_t):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(SAGE, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(in_dim, hid_dim))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hid_dim))
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(hid_dim, hid_dim))
            self.bns.append(torch.nn.BatchNorm1d(hid_dim))
        self.layers.append(SAGEConv(hid_dim, out_dim))

        self.dropout = dropout

    def forward(self, x, adj_t):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj_t)
        return x.log_softmax(dim=-1)

def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def pipe(model_name, hid_dim, l, lr, wd, epochs, patience, dropout, mean_method=None, device='cuda', init_name='virgo'):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor(),
                                     root="/mnt/jiahanli/datasets")

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    if model_name == 'gcn':
        model = GCN(data.num_features, hid_dim,
                    dataset.num_classes, l, dropout).to(device)
    else:
        model = SAGE(data.num_features, hid_dim,
                    dataset.num_classes, l, dropout).to(device)
    # initialize model using dgl-implemented method and datasets
    g, _ = load_dataset('arxiv', device=device) # g here is only used for calculating C2
    if init_name == 'virgoback':
        init_layers(g, model_name, model.layers, init_name, num_classes=_.num_classes, pyg=True)    
    elif init_name == 'virgo':
        init_layers(g, model_name, model.layers, init_name, mean_method=mean_method, num_classes=_.num_classes, pyg=True)    
    else:
        init_layers(g, model_name, model.layers, init_name, pyg=True)

    evaluator = Evaluator(name='ogbn-arxiv')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_acc_list, valid_acc_list, test_acc_list = [], [], []
    best_valid_acc, final_test_acc, cnt = 0.0, 0.0, 0
    for epoch in range(epochs):
        loss = train(model, data, train_idx, optimizer)
        result = test(model, data, split_idx, evaluator)

        train_acc, valid_acc, test_acc = result
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_test_acc = test_acc
            cnt = 0
        else:
            cnt += 1
        print(
            f'Epoch: {epoch:02d} | '
            f'Loss: {loss:.4f} | '
            f'Train: {100 * train_acc:.2f}% | '
            f'Valid: {100 * valid_acc:.2f}% | '
            f'Test: {100 * test_acc:.2f}% | '
            f'Patience: {cnt}/{patience}')
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        test_acc_list.append(test_acc)
        if cnt >= patience:
            print("Early Stopping!")
            break
    print('============================')
    print(f'Final Test: {final_test_acc:4f}')
    test_acc_list.append(final_test_acc)
    return train_acc_list, valid_acc_list, test_acc_list

def run_test():
    searchSpace = {
                "model_name": 'gcn',
                "hid_dim": 256,
                "l": 5,
                "lr": 1e-3,
                "epochs": 10,
                "patience": 50,
                "init_name": 'virgo',
                "mean_method": 'harmonic',
                "wd": 5e-6,
                "dropout": 0.0
            }

    pipe(**searchSpace)

if __name__ == '__main__':
    
    run_test()
    

    print(1)