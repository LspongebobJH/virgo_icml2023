'''
link pred, citation2 + clusterGCN
'''
import torch

from tqdm import tqdm
from utils import init_layers, load_dataset
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor
from torch_geometric.data import ClusterData, ClusterLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers):
        super(GCN, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hid_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hid_dim, hid_dim))
        self.layers.append(GCNConv(hid_dim, out_dim))

        self.dropout = 0.0

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

class GCNInference(torch.nn.Module):
    def __init__(self, weights):
        super(GCNInference, self).__init__()
        self.weights = weights

    def forward(self, x, adj):
        for i, (weight, bias) in enumerate(self.weights):
            x = adj @ x @ weight + bias
            x = np.clip(x, 0, None) if i < len(self.weights) - 1 else x
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

def train(epoch, model, predictor, loader, optimizer, device):
    model.train()
    pbar = tqdm(total=len(loader))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        h = model(data.x, data.edge_index)

        src, dst = data.edge_index
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.x.size(0), src.size(),
                                dtype=torch.long, device=device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = src.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        pbar.update(1)
    pbar.close()

    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, device):
    predictor.eval()
    print('Evaluating full-batch GNN on CPU...')

    weights = [(layer.lin.weight.t().cpu().detach().numpy(),
                layer.bias.cpu().detach().numpy()) for layer in model.layers]
    model = GCNInference(weights)

    x = data.x.numpy()
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
    adj = adj.set_diag()
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    adj = adj.to_scipy(layout='csr')

    h = torch.from_numpy(model(x, adj)).to(device)

    def test_split(split):
        source = split_edge[split]['source_node'].to(device)
        target = split_edge[split]['target_node'].to(device)
        target_neg = split_edge[split]['target_node_neg'].to(device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr

def pipe(epochs=100, lr=1e-3, init='virgo', num_workers=4, device='cuda'):
    device = torch.device(device)
    dataset = PygLinkPropPredDataset(name='ogbl-citation2', root='/mnt/jiahanli/datasets')
    split_edge = dataset.get_edge_split()
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    cluster_data = ClusterData(data, num_parts=15000,
                               recursive=False, save_dir=dataset.processed_dir)

    loader = ClusterLoader(cluster_data, batch_size=256,
                           shuffle=True, num_workers=num_workers)

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

    model = GCN(data.x.size(-1), 256, 256, 3).to(device)

    # initialize model using dgl-implemented method and datasets
    g, _ = load_dataset('citation2', device=device) # g here is only used for calculating C2
    init_layers(g, 'gcn', model.layers, init, pyg=True)

    del g
    torch.cuda.empty_cache()

    predictor = LinkPredictor(256, 256, 1, 3).to(device)
    evaluator = Evaluator(name='ogbl-citation2')

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=lr)
    train_mrr_list, valid_mrr_list, test_mrr_list = [], [], []
    best_valid_mrr, final_test_mrr = 0.0, 0.0
    for epoch in range(epochs):
        loss = train(epoch, model, predictor, loader, optimizer, device)
        result = test(model, predictor, data, split_edge, evaluator,
                        batch_size=64 * 1024, device=device)
        train_mrr, valid_mrr, test_mrr = result
        train_mrr_list.append(train_mrr)
        valid_mrr_list.append(valid_mrr)
        test_mrr_list.append(test_mrr)
        if valid_mrr > best_valid_mrr:
            best_valid_mrr = valid_mrr
            final_test_mrr = test_mrr
        print(
            f'Epoch: {epoch:02d} | '
            f'Loss: {loss:.4f} | '
            f'Train: {100 * train_mrr:.2f}% | '
            f'Valid: {100 * valid_mrr:.2f}% | '
            f'Test: {100 * test_mrr:.2f}%')
        print('---')
    print('============================')
    print(f'Final Test: {100 * final_test_mrr:2f}%')
    test_mrr_list.append(final_test_mrr)
    return train_mrr_list, valid_mrr_list, test_mrr_list

def run_test():
    searchSpace = {
            "lr": 1e-3,
    }

    print(searchSpace)
    pipe(**searchSpace)

if __name__ == "__main__":
    run_test()
    print(1)