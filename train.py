import os
import argparse
import torch
import numpy as np
import scipy.sparse as sp
from models.TGN import TGN
from models.idea_model import idea_model
from DataHelper import Dataset
from models.idea_model import TGSL
from torch.autograd import Variable

# from dataset import Dataset, get_PtbAdj

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="wikipedia", help='dataset')  # check
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--ptb_rate', type=float, default=0.15, help="noise ptb_rate")

parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--np_file', type=str, default='./data/Wikipedia/ml_wikipedia.npy', help='edge feature file')

parser.add_argument('--neighbor_size', type=int, default=2, help='the number of sampled neighbors.')
parser.add_argument('--emb_size', type=int, default=128, help='the dimension of node embedding.')
parser.add_argument('--neg_size', type=int, default=1, help='the number of negative samples.')

parser.add_argument('--n_layers', type=int, default=1, help='the number of TGN aggregation layers.')
parser.add_argument('--n_neighbors', type=int, default=5, help='the number of TGN sampled neighbors.')
parser.add_argument('--batch_size', type=int, default=500, help='batch size of TGN.')
parser.add_argument('--hid_size', type=int, default=80, help='hidden size of MLP in TGN.')
parser.add_argument('--mem_size', type=int, default=128, help='memory dimension of each node in TGN.')
parser.add_argument('--time_encoding_size', type=int, default=128,
                    help='The dimension of time encoding of each node in TGN.')
parser.add_argument('--edge_features_size', type=int, default=172, help='The dimension of edge features in TGN.')

parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--tgsl_loss_weight', type=float, default=0.8, help='hp for tgsl loss weight')
parser.add_argument('--sigma', type=float, default=100,
                    help='the parameter to control the variance of sample weights in rec loss')
parser.add_argument('--inner_steps', type=int, default=0, help='steps for inner optimization TGN')
parser.add_argument('--outer_steps', type=int, default=3, help='steps for outer optimization TGSL')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(args.device if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

print(args)

FType = torch.FloatTensor
LType = torch.LongTensor

data = Dataset(root='./data/Wikipedia/wikipedia.txt', name=args.dataset, args=args)

train_idxs, val_idxs, test_idxs = data.get_data_node_classification()
idea = idea_model(data, args, device=args.device)
idea.fit(train_idxs, val_idxs, test_idxs)

print("===Train over===")

