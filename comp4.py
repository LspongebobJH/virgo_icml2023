'''
products + sage, mini batch training
'''
# Reaches around 0.7870 ± 0.0036 test accuracy.

import torch
import torch.nn.functional as F

from tqdm import tqdm

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from utils import EarlyStopping, init_layers, load_dataset
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()

        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.layers.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.layers[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.layers[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

def train(model, optimizer, train_idx, train_loader, epoch, x, y, num_batches, device):
    model.train()

    pbar = tqdm(total=train_loader.batch_size * num_batches if train_loader.batch_size * num_batches < train_idx.size(0) else train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    cnt_batches, cnt_samples = 0, 0
    batch_acc_list = []
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        batch_correct = int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        total_correct += batch_correct
        batch_acc = batch_correct / batch_size
        batch_acc_list.append(batch_acc)
        pbar.update(batch_size)

        cnt_batches += 1
        cnt_samples += batch_size
        if cnt_batches >= num_batches:
            break

    pbar.close()

    # loss = total_loss / len(train_loader)
    loss = total_loss / num_batches
    # approx_acc = total_correct / train_idx.size(0)
    approx_acc = total_correct / cnt_samples

    return loss, approx_acc, batch_acc_list

@torch.no_grad()
def test(model, split_idx, evaluator, subgraph_loader, x, y, device):
    model.eval()

    out = model.inference(x, subgraph_loader, device)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

def pipe(mean_method, epochs=20, l=2, hid_dim=256, dropout=0.5, lr=1e-3, weight_decay=0.0, num_batches=-1, num_workers=8, init='virgo', device='cuda'):
    dataset = PygNodePropPredDataset('ogbn-products', root="/mnt/jiahanli/datasets")
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-products')
    data = dataset[0]

    if l == 2:
        sizes = [10, 5]
    elif l == 3:
        sizes = [15, 10, 5]
    elif l == 4:
        sizes = [20, 15, 10, 5]
    train_idx = split_idx['train']
    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                sizes=sizes, batch_size=1024,
                                shuffle=True, num_workers=num_workers)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                    batch_size=4096, shuffle=False,
                                    num_workers=num_workers)
    if num_batches == -1:
        num_batches = len(train_loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAGE(dataset.num_features, hid_dim, dataset.num_classes, num_layers=l, dropout=dropout)

     # initialize model using dgl-implemented method and datasets
    g, _ = load_dataset('products', device=device) # g here is only used for calculating C2
    if init == 'virgoback':
        init_layers(g, 'sage', model.layers, init, num_classes=_.num_classes, pyg=True)
    elif init == 'virgo':
        init_layers(g, 'sage', model.layers, init, mean_method=mean_method, num_classes=_.num_classes, pyg=True)
    else:
        init_layers(g, 'sage', model.layers, init, pyg=True)

    del g
    torch.cuda.empty_cache()

    model = model.to(device)
    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    all_batch_acc_list = []
    for epoch in range(epochs):
        loss, acc, batch_acc_list = train(model, optimizer, train_idx, train_loader, epoch, x, y, num_batches, device)
        print(f'Epoch {epoch:02d} | Loss: {loss:.4f} | Approx. Train: {acc:.4f}')

        train_acc, val_acc, test_acc = test(model, split_idx, evaluator, subgraph_loader, x, y, device)
        print(f'Evaluation, Train: {train_acc:.4f}| Val: {val_acc:.4f}| Test: {test_acc:.4f}')
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        all_batch_acc_list.append(batch_acc_list)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc

    print('============================')
    print(f'Final Test: {final_test_acc:4f}')
    test_acc_list.append(final_test_acc)
    
    return train_acc_list, val_acc_list, test_acc_list, final_test_acc, all_batch_acc_list

def run_test():
    searchSpace = {
            "num_batches": 100,
            "num_workers": 8,
            "l": 4,
            "hid_dim": 256,
            "epochs": 1,
            "mean_method": 'harmonic'
    }

    print(searchSpace)
    pipe(**searchSpace)

if __name__ == '__main__':
    run_test()

    print(1)

