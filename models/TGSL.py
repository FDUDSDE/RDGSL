import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.functional import softmax

FType = torch.FloatTensor
LType = torch.LongTensor


class TGSL(nn.Module):

    def __init__(self, data, args, device='cuda'):
        super(TGSL, self).__init__()
        self.device = args.device
        self.args = args
        self.index = data.index
        self.emb_size = args.emb_size
        self.max_timestamps = data.max_d_time
        self.batch_size = args.batch_size

        self.sources = torch.tensor(data.sources, dtype=torch.long).to(self.device)
        self.destinations = torch.tensor(data.destinations, dtype=torch.long).to(self.device)
        self.timestamps = torch.tensor(data.timestamps, dtype=torch.float).to(self.device)

        self.s_h_nodes = torch.tensor(data.s_his_nodes, dtype=torch.long).to(self.device)
        self.t_h_nodes = torch.tensor(data.t_his_nodes, dtype=torch.long).to(self.device)
        self.s_h_times = torch.tensor(data.s_his_times, dtype=torch.float).to(self.device)
        self.t_h_times = torch.tensor(data.t_his_times, dtype=torch.float).to(self.device)
        self.s_his_mask = torch.tensor(data.s_his_masks, dtype=torch.float).to(self.device)
        self.t_his_mask = torch.tensor(data.t_his_masks, dtype=torch.float).to(self.device)
        self.hist_len = args.neighbor_size
        self.neg_size = args.neg_size

        self.node_dim = data.node_dim
        self.gat_hidden_size = args.emb_size

        self.delta_s = torch.nn.Parameter((torch.zeros(self.node_dim) + 1.).type(FType).to(self.device),
                                          requires_grad=True)
        self.delta_t = torch.nn.Parameter((torch.zeros(self.node_dim) + 1.).type(FType).to(self.device),
                                          requires_grad=True)

        self.W = torch.nn.Parameter(torch.zeros(size=(self.emb_size, self.gat_hidden_size)).to(self.device))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = torch.nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1)).to(self.device))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.global_att_linear_layer = torch.nn.Linear(self.gat_hidden_size, 1).to(self.device)
        self.leakyrelu = torch.nn.LeakyReLU(0.2).to(self.device)

        self.MLP = nn.Sequential(nn.Linear(self.gat_hidden_size, 60),
                                 nn.ReLU(),
                                 nn.Linear(60, self.emb_size)).to(self.device)

        self.estimated_weights = None
        self.node_emb = None

    def forward(self, sources, destinations, node_features, start_idx, end_idx):

        self.node_emb = self.MLP(node_features)

        tgsl_loss = self.compute_tgsl_loss(sources, destinations, start_idx, end_idx)

        x0 = self.node_emb[sources]
        x1 = self.node_emb[destinations]
        output = torch.sum(torch.mul(x0, x1), dim=1)
        self.estimated_weights = F.relu(output)
        node_emb = copy.deepcopy(self.node_emb.detach())

        return tgsl_loss,  self.estimated_weights.float(), node_emb

    def compute_tgsl_loss(self, sources, destinations, start_idx, end_idx):

        batch_size = len(sources)
        pos0 = self.node_emb[sources]
        pos1 = self.node_emb[destinations]
        pos = F.relu(torch.sum(torch.mul(pos0, pos1), dim=1))

        neg_sources_idx = np.random.randint(0, self.node_dim, batch_size * self.neg_size)
        neg_destinations_idx = np.random.randint(0, self.node_dim, batch_size * self.neg_size)
        neg0 = self.node_emb[neg_sources_idx].view(batch_size, self.neg_size, -1)
        neg1 = self.node_emb[neg_destinations_idx].view(batch_size, self.neg_size, -1)
        neg_s = F.relu(torch.sum(torch.mul(pos0.unsqueeze(1), neg1), dim=2))
        neg_t = F.relu(torch.sum(torch.mul(pos1.unsqueeze(1), neg0), dim=2))

        pos_sim, s_neg_sim, t_neg_sim = self.computer_similarity(sources, destinations,
                                                                     self.timestamps[start_idx:end_idx],
                                                                     self.s_h_nodes[start_idx:end_idx], self.t_h_nodes[start_idx:end_idx],
                                                                     self.s_h_times[start_idx:end_idx], self.s_his_mask[start_idx:end_idx],
                                                                     self.t_h_times[start_idx:end_idx], self.t_his_mask[start_idx:end_idx],
                                                                     pos0, pos1,
                                                                     neg0, neg1)

        pos_loss = (-torch.log(torch.sigmoid(pos_sim / self.args.sigma) + 1e-6)) * F.mse_loss(pos, torch.ones_like(pos),
                                                                                              reduction='none')
        s_neg_loss = (-torch.log(torch.sigmoid(s_neg_sim.neg() / self.args.sigma) + 1e-6)) * F.mse_loss(neg_s,
                                                                                                        torch.zeros_like(
                                                                                                            neg_s),
                                                                                                        reduction='none')
        t_neg_loss = (-torch.log(torch.sigmoid(t_neg_sim.neg() / self.args.sigma) + 1e-6)) * F.mse_loss(neg_t,
                                                                                                        torch.zeros_like(
                                                                                                            neg_t),
                                                                                                        reduction='none')

        tgsl_loss = pos_loss.sum() / self.index + (s_neg_loss.sum() + t_neg_loss.sum()) / (self.index * self.neg_size)

        return tgsl_loss

    def computer_similarity(self, s_nodes, t_nodes, e_times,
                            s_h_nodes, t_h_nodes,
                            s_h_times, s_h_time_mask,
                            t_h_times, t_h_time_mask,
                            s_node_emb, t_node_emb,
                            s_n_node_emb, t_n_node_emb):

        e_times = e_times * 1.0 / self.max_timestamps
        s_h_times = s_h_times * 1.0 / self.max_timestamps
        t_h_times = t_h_times * 1.0 / self.max_timestamps
        batch = s_nodes.size()[0]

        s_h_node_emb = self.node_emb.index_select(0, Variable(s_h_nodes.view(-1))).view(batch, self.hist_len, -1)
        t_h_node_emb = self.node_emb.index_select(0, Variable(t_h_nodes.view(-1))).view(batch, self.hist_len, -1)
        delta_s = self.delta_s.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        delta_t = self.delta_t.index_select(0, Variable(t_nodes.view(-1))).unsqueeze(1)
        d_time_s = torch.abs(e_times.unsqueeze(1) - s_h_times)
        d_time_t = torch.abs(e_times.unsqueeze(1) - t_h_times)

        # GAT attention_rewrite
        for i in range(self.hist_len):
            s_h_node_emb_i = torch.transpose(s_h_node_emb[:, i:(i + 1), :], dim0=1, dim1=2).squeeze()
            s_node_emb_i = s_node_emb
            d_time_s_i = Variable(d_time_s)[:, i:(i + 1)]
            if i == 0:
                a_input = torch.cat([torch.mm(s_node_emb_i, self.W), torch.mm(s_h_node_emb_i, self.W)],dim=1)
                sim_s_s_his = self.leakyrelu(torch.exp(-delta_s * d_time_s_i) * torch.mm(a_input, self.a))
            else:
                a_input = torch.cat([torch.mm(s_node_emb_i, self.W), torch.mm(s_h_node_emb_i, self.W)], dim=1)
                sim_s_s_his = torch.cat([sim_s_s_his,
                                         self.leakyrelu(torch.exp(-delta_s * d_time_s_i) * torch.mm(a_input, self.a))],
                                        dim=1)

        for i in range(self.hist_len):
            t_h_node_emb_i = torch.transpose(t_h_node_emb[:, i:(i + 1), :], dim0=1, dim1=2).squeeze()
            t_node_emb_i = t_node_emb
            d_time_t_i = Variable(d_time_t)[:, i:(i + 1)]
            if i == 0:
                a_input = torch.cat([torch.mm(t_node_emb_i, self.W), torch.mm(t_h_node_emb_i, self.W)],
                                    dim=1)
                sim_t_t_his = self.leakyrelu(torch.exp(-delta_s * d_time_t_i) * torch.mm(a_input, self.a))
            else:
                a_input = torch.cat([torch.mm(t_node_emb_i, self.W), torch.mm(t_h_node_emb_i, self.W)], dim=1)
                sim_t_t_his = torch.cat([sim_t_t_his,
                                         self.leakyrelu(torch.exp(-delta_s * d_time_t_i) * torch.mm(a_input, self.a))],
                                        dim=1)

        att_s_his_s = softmax(sim_s_s_his, dim=1)
        att_t_his_t = softmax(sim_t_t_his, dim=1)

        s_his_hat_emb_inter = ((att_s_his_s * Variable(s_h_time_mask)).unsqueeze(2) *
                               torch.mm(s_h_node_emb.view(s_h_node_emb.size()[0] * self.hist_len, -1), self.W).
                               view(s_h_node_emb.size()[0], self.hist_len, -1)).sum(dim=1)
        t_his_hat_emb_inter = ((att_t_his_t * Variable(t_h_time_mask)).unsqueeze(2) *
                               torch.mm(t_h_node_emb.view(t_h_node_emb.size()[0] * self.hist_len, -1), self.W).
                               view(t_h_node_emb.size()[0], self.hist_len, -1)).sum(dim=1)

        # temporal-self-attention
        global_att = softmax(torch.tanh(self.global_att_linear_layer(torch.transpose(
            torch.cat(
                [(s_his_hat_emb_inter * torch.exp(-delta_s * Variable(d_time_s.mean(dim=1)).unsqueeze(1))).unsqueeze(2),
                 (t_his_hat_emb_inter * torch.exp(-delta_t * Variable(d_time_t.mean(dim=1)).unsqueeze(1))).unsqueeze(
                     2)],
                dim=2), dim0=1, dim1=2))), dim=1).squeeze(2)
        global_att_s = global_att[:, 0]
        global_att_t = global_att[:, 1]
        self.global_attention = global_att

        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1)
        p_alpha_s = ((s_h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2)
        p_alpha_t = ((t_h_node_emb - s_node_emb.unsqueeze(1)) ** 2).sum(dim=2)

        p_lambda = p_mu \
                   + global_att_s * (att_s_his_s * p_alpha_s * torch.exp(delta_s * Variable(d_time_s)) * Variable(
            s_h_time_mask)).sum(
            dim=1) \
                   + global_att_t * (att_t_his_t * p_alpha_t * torch.exp(delta_t * Variable(d_time_t)) * Variable(
            t_h_time_mask)).sum(
            dim=1)

        # computing negtive similarity
        n_mu_s = ((s_node_emb.unsqueeze(1) - t_n_node_emb) ** 2).sum(dim=2)
        n_mu_t = ((t_node_emb.unsqueeze(1) - s_n_node_emb) ** 2).sum(dim=2)
        n_alpha_s = ((s_h_node_emb.unsqueeze(2) - t_n_node_emb.unsqueeze(1)) ** 2).sum(dim=3)
        n_alpha_t = ((t_h_node_emb.unsqueeze(2) - s_n_node_emb.unsqueeze(1)) ** 2).sum(dim=3)

        n_lambda_s = n_mu_s \
                     + global_att_s.unsqueeze(1) * (att_s_his_s.unsqueeze(2) * n_alpha_s
                                                    * (torch.exp(delta_s * Variable(d_time_s)).unsqueeze(2))
                                                    * (Variable(s_h_time_mask).unsqueeze(2))).sum(dim=1)

        n_lambda_t = n_mu_t \
                     + global_att_t.unsqueeze(1) * (att_t_his_t.unsqueeze(2) * n_alpha_t
                                                    * (torch.exp(delta_t * Variable(d_time_t)).unsqueeze(2))
                                                    * (Variable(t_h_time_mask).unsqueeze(2))).sum(dim=1)

        return p_lambda, n_lambda_s, n_lambda_t
