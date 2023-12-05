'''
link pred, ppa + gcn
'''

import argparse

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from utils import EarlyStopping, init_layers, load_dataset

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers):
        super(GCN, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(
            GCNConv(in_dim, hid_dim, normalize=False))
        for _ in range(num_layers - 2):
            self.layers.append(
                GCNConv(hid_dim, hid_dim, normalize=False))
        self.layers.append(
            GCNConv(hid_dim, out_dim, normalize=False))

        self.dropout = 0.0

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj_t):
        for layer in self.layers[:-1]:
            x = layer(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, adj_t)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_dim, hid_dim))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hid_dim, hid_dim))
        self.lins.append(torch.nn.Linear(hid_dim, out_dim))

        self.dropout = 0.0

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(num_neg, epoch, model, predictor, data, split_edge, optimizer, batch_size, num_workers):
    if num_neg == -1:
        num_neg = edge.size()[-1]
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0

    pbar = tqdm(total=pos_train_edge.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True, num_workers=num_workers):
        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, [2, num_neg] if num_neg < edge.size()[-1] else edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        pbar.update(batch_size)
    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, num_workers):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

def pipe(num_neg=-1, lr=1e-2, init='virgo', num_workers=1, device='cuda'):
    dataset = PygLinkPropPredDataset(name='ogbl-ppa',
                                     root='/mnt/jiahanli/datasets',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    data.x = data.x.to(torch.float)
    split_edge = dataset.get_edge_split()
    model = GCN(data.num_features, 256,
                256, 3).to(device)
    # Pre-compute GCN normalization.
    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t
    # initialize model using dgl-implemented method and datasets
    g, _ = load_dataset('ppa', device=device) # g here is only used for calculating C2
    init_layers(g, 'gcn', model.layers, init, pyg=True)
    del g
    torch.cuda.empty_cache()
    data = data.to(device)

    predictor = LinkPredictor(256, 256, 1, 3).to(device)
    evaluator = Evaluator(name='ogbl-ppa')

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=lr)

    res_list = []
    for epoch in range(20):
        loss = train(num_neg, epoch, model, predictor, data, split_edge, optimizer, 64*1024, num_workers)
        results = test(model, predictor, data, split_edge, evaluator, 64*1024, num_workers)
        res_list.append(results)
        for key, result in results.items():
            train_hits, valid_hits, test_hits = result
            print(key)
            print(
                f'Epoch: {epoch:02d} | '
                f'Loss: {loss:.4f} | '
                f'Train: {100 * train_hits:.2f}% | '
                f'Valid: {100 * valid_hits:.2f}% | '
                f'Test: {100 * test_hits:.2f}%'
            )
        print('---')
    return res_list

def run_test():
    searchSpace = {
            "num_neg": 64*1024,
    }

    print(searchSpace)
    pipe(**searchSpace)

if __name__ == "__main__":
    run_test()
    print(1)