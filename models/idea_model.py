import math
import random
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DataHelper import accuracy
from torch.autograd import Variable
from torch.nn.functional import softmax
from models.TGN import TGN
import scipy.sparse as sp
import numpy as np
from models.TGSL import TGSL
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

FType = torch.FloatTensor
LType = torch.LongTensor


class idea_model:
    def __init__(self, data, args, device):
        self.device = device
        self.args = args

        self.data = data
        self.n_nodes = data.node_dim
        self.labels = data.labels.to(self.device)
        self.edge_features = data.edge_features
        self.node_features = data.node_features

        self.sources = data.sources
        self.destinations = data.destinations
        self.timestamps = data.timestamps
        self.edge_idxs = data.edge_idxs
        self.edge_weights = torch.zeros(data.index)
        self.temporal_features = []
        self.best_val_auc = 0.0
        self.best_test_auc = 0.0
        self.best_epoch = 0

        # 初始化模型
        self.tgsl = TGSL(self.data, self.args, self.device)
        self.tgn = TGN(self.n_nodes, self.args, self.device)

        self.criterion = torch.nn.BCELoss()

        self.optimizer_tgsl = optim.Adam(self.tgsl.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_tgn = optim.Adam(self.tgn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def fit(self, idx_train, idx_val, idx_test):
        args = self.args
        device = self.device
        batch_size = self.args.batch_size
        model_tgsl = self.tgsl
        model_tgn = self.tgn
        optimizer_tgsl = self.optimizer_tgsl
        optimizer_tgn = self.optimizer_tgn

        num_instance = len(idx_train)
        num_batch = math.ceil(num_instance / batch_size)

        # Train model
        print("\n==== train_tgsl ====")
        t_total = time.time()

        model_tgsl.train()
        model_tgn.train()
        model_tgn.to(device)
        model_tgsl.to(device)
        for epoch in range(args.epoch):
            start_epoch = time.time()
            loss_t = 0.0
            loss_tgn_t = 0.0
            loss_tgsl_t = 0.0

            model_tgn.init_model()

            max_node_idx = max(max(self.sources), max(self.destinations))
            adj_list = [[] for _ in range(max_node_idx + 1)]

            for batch_idx in range(0, num_batch):
                t = time.time()
                optimizer_tgsl.zero_grad()
                optimizer_tgn.zero_grad()

                start_idx = batch_idx * batch_size
                end_idx = min(num_instance, start_idx + batch_size)
                sources_batch, destinations_batch = self.sources[start_idx:end_idx], self.destinations[start_idx:end_idx]
                edge_idxs_batch, timestamps_batch = self.edge_idxs[start_idx:end_idx], self.timestamps[start_idx:end_idx]
                labels_batch = self.labels[start_idx:end_idx]

                loss_tgsl, edge_weights_batch, self.temporal_features \
                    = model_tgsl(torch.tensor(sources_batch, dtype=torch.long).to(self.device),
                                 torch.tensor(destinations_batch, dtype=torch.long).to(self.device),
                                 torch.tensor(self.node_features, dtype=torch.float).to(self.device),
                                 start_idx, end_idx)

                pre_prob_batch = model_tgn(np.array(sources_batch),
                                           np.array(destinations_batch),
                                           np.array(timestamps_batch),
                                           np.array(edge_idxs_batch),
                                           np.array(edge_weights_batch.detach().cpu()),
                                           self.edge_features,
                                           self.temporal_features, adj_list)

                loss_tgn = self.criterion(pre_prob_batch, labels_batch)

                total_loss = args.tgsl_loss_weight * loss_tgn + loss_tgsl

                total_loss.backward()
                optimizer_tgsl.step()
                optimizer_tgn.step()
                model_tgn.node_mem.detach_()

                loss_t += total_loss
                loss_tgn_t += loss_tgn
                loss_tgsl_t += loss_tgsl

            # Evaluate validation set performance separately
            model_tgsl.eval()
            model_tgn.eval()
            loss_tgsl, edge_weights_batch, self.temporal_features \
                = model_tgsl(torch.tensor(np.array(self.sources)[idx_val], dtype=torch.long).to(self.device),
                             torch.tensor(np.array(self.destinations)[idx_val], dtype=torch.long).to(self.device),
                             torch.tensor(self.node_features, dtype=torch.float).to(self.device),
                             idx_val[0], idx_val[-1] + 1)
            pre_prob_val = model_tgn(np.array(self.sources)[idx_val],
                                     np.array(self.destinations)[idx_val],
                                     np.array(self.timestamps)[idx_val],
                                     np.array(self.edge_idxs)[idx_val],
                                     np.array(edge_weights_batch.detach().cpu()),
                                     self.edge_features,
                                     self.temporal_features, adj_list)
            pre_prob_val = pre_prob_val.detach().cpu().numpy()
            auc_val = roc_auc_score(self.labels[idx_val].cpu(), pre_prob_val)

            if auc_val > self.best_val_auc:
                self.best_val_auc = auc_val
                self.best_epoch = epoch + 1

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_tgn: {:.4f}'.format(loss_tgn_t.item()),
                  'loss_tgsl: {:.4f}'.format(loss_tgsl_t.item()),
                  'loss_total: {:.4f}'.format(loss_t.item()))
            print('Epoch: {:04d}'.format(epoch + 1),
                  'auc_val: {:.4f}'.format(auc_val.item()),
                  'time: {:.4f}s'.format(time.time() - start_epoch))


        print("Optimization Finished!")
        print("best_epoch:{:4f}, best validation auc:{}".format(self.best_epoch, self.best_val_auc))
        print("Total time elapsed:{:4f}s".format(time.time() - t_total))





class evaluation:
    def __init__(self, args, device):
        super().__init__()
        self.batch_size = 500
        self.device = device
        self.criterion = torch.nn.BCELoss()
        self.lr = 0.0001

        self.affinity_score = MergeLayer(128, 128, 128, 1).to(self.device)


    def evaluate_link_prediction(self, sources_embedding_path, destinations_embedding_path):
        sources_embedding, destinations_embedding = [], []
        with open(sources_embedding_path, 'r') as reader:
            reader.readline()
            node_id = 0
            for line in reader:
                embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
                sources_embedding[node_id] = embeds
                node_id += 1
        with open(destinations_embedding_path, 'r') as reader:
            reader.readline()
            node_id = 0
            for line in reader:
                embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
                destinations_embedding[node_id] = embeds
                node_id += 1
        counts = len(sources_embedding)
        counts_train = int(counts * 0.8)
        train_idx = [i for i in range(counts_train)]
        val_idx = [i for i in range(counts_train, len(sources_embedding))]
        inter_counts = len(train_idx)
        num_batch = math.ceil(inter_counts / self.batch_size)
        negtive_destinations = np.random.randint(0, len(destinations_embedding), len(destinations_embedding))
        negtive_embedding = destinations_embedding[negtive_destinations]
        affinity_score = self.affinity_score
        optimizer = torch.optim.Adam(affinity_score.parameters(), lr=self.lr)
        affinity_score.train()
        for batch_idx in range(num_batch):
            start_idx = batch_idx * self.batch_size
            end_idx = min(inter_counts, start_idx + self.batch_size)
            size = end_idx - start_idx
            sources_embedding_batch = sources_embedding[start_idx:end_idx]
            destinations_embedding_batch, negtive_embedding_batch = destinations_embedding[start_idx:end_idx], negtive_embedding[start_idx:end_idx]

            optimizer.zero_grad()
            score = self.affinity_score(torch.cat([sources_embedding_batch, sources_embedding_batch], dim=0),
                                        torch.cat([destinations_embedding_batch,negtive_embedding_batch])).squeeze(dim=0)
            pos_score = score[:size].sigmoid()
            neg_score = score[size:].sigmoid()
            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=self.device)
                neg_label = torch.zeros(size, dtype=torch.float, device=self.device)
            loss = self.criterion(pos_score.squeeze(),pos_label) + self.criterion(neg_score.squeeze(), neg_label)
            loss.backward()
            optimizer.step()
        affinity_score.eval()
        score = self.affinity_score(torch.cat([sources_embedding[val_idx], sources_embedding[val_idx]], dim=0),
                                    torch.cat([destinations_embedding[val_idx], negtive_embedding[val_idx]])).squeeze(dim=0)
        pos_score = score[:size].sigmoid()
        neg_score = score[size:].sigmoid()
        pred_score = np.concatenate([(pos_score).cpu().numpy(), (neg_score).cpu().numpy()])
        true_label = np.concatenate([np.ones(size), np.zeros(size)])
        val_ap = average_precision_score(true_label, pred_score)
        val_auc = roc_auc_score(true_label, pred_score)
        print('link val_ap: {04d}, val_auc:{04d}'.format(val_ap, val_auc))


class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)
