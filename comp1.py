'''
cora, pubmed, citeseer, reddit
'''
import argparse
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np


from dgl.utils import expand_as_pair
from dgl.base import DGLError
from dgl.data import *
from dgl.transforms import AddSelfLoop
from dgl.nn.pytorch import GraphConv
from torch.nn.init import uniform_
from utils import *

class GCN(nn.Module):
    def __init__(self, n_layers, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        assert n_layers >= 2
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=F.relu))
        for _ in range(n_layers-2):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=F.relu))
        self.layers.append(GraphConv(hid_dim, out_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g, feat):
        h = feat
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

def evaluate(g, feat, labels, mask, model):
    model.eval()
    with th.no_grad():
        logits = model(g, feat)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def pipe(gName, model_name, hid_dim, l, lr, wd, epochs, patience, dropout, mean_method=None, device='cuda', init='virgo'):
    g, data = load_dataset(gName, rand_feat=False, device=device)
    if gName in ['arxiv', 'proteins']:
        feat = g.ndata['feat']
        _, labels = data[0]
        masks = data.get_idx_split()
        train_mask, val_mask, test_mask = masks['train'], masks['valid'], masks['test']
        labels, train_mask, val_mask, test_mask = \
            labels.squeeze().to(device), train_mask.to(device), val_mask.to(device), test_mask.to(device)
    else:   
        feat = g.ndata['feat']
        labels = g.ndata['label']
        masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
        train_mask = masks[0]
        val_mask = masks[1]
        test_mask = masks[2]
        
    # create GCN model    
    in_dim = feat.shape[1]
    out_dim = data.num_classes
    model = GCN(l, in_dim, hid_dim, out_dim, dropout).to(device)

    # initialize model
    init_layers(g, model_name, model.layers, init, mean_method=mean_method, num_classes=out_dim)

    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    earlystop = EarlyStopping(patience)

    # training loop
    train_acc_list, valid_acc_list, test_acc_list = [], [], []
    for epoch in range(epochs):
        model.train()
        logits = model(g, feat)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = evaluate(g, feat, labels, train_mask, model)
        val_acc = evaluate(g, feat, labels, val_mask, model)
        test_acc = evaluate(g, feat, labels, test_mask, model)
        train_acc_list.append(train_acc)
        valid_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        earlystop.step_score(val_acc, model)
        print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
              f"Test acc: {test_acc:.4f} | Patience: {earlystop.counter}/{patience}")
        if earlystop.early_stop:
            print("Early Stopping!")
            break
    earlystop.load_model(model)
    acc = evaluate(g, feat, labels, test_mask, model)
    print("Test accuracy {:.4f}".format(acc))
    test_acc_list.append(acc)
    return train_acc_list, valid_acc_list, test_acc_list, acc

def run_test():
    searchSpace = {
                "gName": 'cora',
                "model_name": "gcn",
                "hid_dim": 64,
                "l": 3,
                "lr": 1e-3,
                "epochs": 1000,
                "patience": 20,
                "dropout": 0.0,
                "wd": 5e-6,
                "mean_method": "normal",
                "init": 'virgo'
            }

    pipe(**searchSpace)

if __name__ == '__main__':
    run_test()

    print(1)
