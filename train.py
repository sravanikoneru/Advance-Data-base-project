import argparse
import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import models
import utils

from matplotlib import pyplot as plt
import models
import utils
import os


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')

    parser.set_defaults(model_type='GCN',
                        dataset='cora',
                        num_layers=2,
                        batch_size=32,
                        hidden_dim=32,
                        dropout=0.0,
                        epochs=200,
                        opt='adam',   # opt_parser
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()

def train(dataset, task, args):
    if task == 'graph':
        # graph classification: separate dataloader for test set
        data_size = len(dataset)
        loader = DataLoader(
                dataset[:int(data_size * 0.8)], batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
                dataset[int(data_size * 0.8):], batch_size=args.batch_size, shuffle=True)
        # print('There are %s graphs in the test set of ENZYMES' % (data_size - int(data_size * 0.8)))
    elif task == 'node':
        # use mask to split train/validation/test
        test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        # print('There are %s nodes in the test set of Cora' % (dataset[0]['test_mask'].sum().item()))
    else:
        raise RuntimeError('Unknown task')

    # build model
    model = models.GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, 
                            args, task=task)
    scheduler, opt = utils.build_optimizer(args, model.parameters())
    
    vals = []
    tests = []
    best_val_acc = 0
    test_acc = 0
    early_stop = 1e9
    stop_cnt = 0

    # train
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        print('train_loss\t%f' % total_loss)
        
        
        val_acc, tmp_test_acc = test(loader, model, is_validation=True), test(loader, model)
        vals.append(val_acc)
        tests.append(tmp_test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            stop_cnt = 0
        else:
            stop_cnt += 1
        print("Loss in Epoch {:03d}: {:.4f}. ".format(epoch, total_loss), end="")
        print("Current Best Val Acc {:.4f}, with Test Acc {:.4f}".format(best_val_acc, test_acc))

        if stop_cnt >= early_stop:
            break

    print('Final Val Acc {0}, Test Acc {1}'.format(best_val_acc, test_acc))
    return list(range(1, args.epochs + 1)), vals

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total

def main():
    args = arg_parse()
    # print(args)

    if args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
    elif args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        task = 'node'
    #train(dataset, task, args) 
    
    gcn_epoch, gcn_vals = train(dataset, task, args)
    plt.plot(gcn_epoch, gcn_vals, label="GCN")

    args.model_type = "GraphSage"
    args.hidden_dim = 256
    gcn_epoch, gcn_vals = train(dataset, task, args)
    plt.plot(gcn_epoch, gcn_vals, label="GraphSage")

    args.model_type = "GAT"
    args.hidden_dim = 16
    gcn_epoch, gcn_vals = train(dataset, task, args)
    plt.plot(gcn_epoch, gcn_vals, label="GAT")

    plt.title("Validation Accuracy Changes over Epochs on Enzymes Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()