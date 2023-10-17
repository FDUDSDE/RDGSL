import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DataHelper import accuracy, auc
import numpy as np
from torch.autograd import Variable
from collections import defaultdict
from DataHelper import get_neighbor_finder

FType = torch.FloatTensor
LType = torch.LongTensor


class TGN(nn.Module):
    def __init__(self, n_nodes, args, device='cuda'):
        super(TGN, self).__init__()
        self.args = args
        self.device = args.device
        self.n_layers = args.n_layers
        self.n_neighbors = args.n_neighbors
        self.BATCH_SIZE = args.batch_size
        self.emb_size = args.emb_size
        self.hid_size = args.hid_size
        self.men_size = args.mem_size
        self.time_encoding_size = args.time_encoding_size
        self.edge_features_size = args.edge_features_size
        self.n_nodes = n_nodes
        self.message_size = self.men_size * 2 + self.time_encoding_size \
                            + self.edge_features_size

        self.time_encoder = TimeEncode(dimension=self.time_encoding_size, device=self.device)

        self.memory_updater = nn.RNNCell(input_size=self.message_size,
                                         hidden_size=self.men_size).to(self.device)

        self.neighbor_finder = None
        self.node_mem = None
        self.last_updated = None
        self.temporal_node_features = None
        self.edge_number = None
        self.criterion = torch.nn.BCELoss()

        self.MLP = nn.Sequential(nn.Linear(self.emb_size, self.hid_size),
                                 nn.ReLU(),
                                 nn.Dropout(p=args.dropout, inplace=False),
                                 nn.Linear(self.hid_size, 10),
                                 nn.ReLU(),
                                 nn.Dropout(p=args.dropout, inplace=False),
                                 nn.Linear(10, 1)).to(self.device)

        self.linear_1 = torch.nn.ModuleList([
            torch.nn.Linear(self.emb_size + self.time_encoding_size + self.edge_features_size, self.emb_size) \
            for _ in range(self.n_layers)
        ]).to(self.device)
        self.linear_2 = torch.nn.ModuleList([
            torch.nn.Linear(self.men_size + self.time_encoding_size + self.emb_size, self.emb_size) \
            for _ in range(self.n_layers)
        ]).to(self.device)

    def init_model(self, ):
        self.node_mem = Variable(torch.zeros((self.n_nodes, self.men_size)). \
                                 to(self.device), requires_grad=False)

        self.last_updated = Variable(torch.zeros(size=(self.n_nodes,)). \
                                     to(self.device), requires_grad=False)

    def forward(self, sources, destinations, timestamps, edge_idxs, edge_weights, edge_features, \
                temporal_node_features, adj_list):
        """
        :param sources: source id
        :param destinations: destination id
        :param edge_idxs: index of interaction
        :param timestamps: timestamps of interaction
        :param edge_weights: weights of edge

        :return: probobility of the class of each node. shape:(node_number, class_number)
        """
        n_instance = len(sources)
        n_batch = math.ceil(n_instance / self.BATCH_SIZE)
        temporal_embedding = torch.tensor(np.zeros((n_instance, self.emb_size)), dtype=torch.float).to(self.device)

        self.temporal_node_features = temporal_node_features.detach_()
        self.edge_features = torch.FloatTensor(edge_features).to(self.device)
        self.edge_number = len(self.edge_features)

        self.neighbor_finder = get_neighbor_finder(sources, destinations, timestamps, edge_idxs, edge_weights, adj_list,
                                                   uniform=True)

        for batch_idx in range(n_batch):
            start_idx = batch_idx * self.BATCH_SIZE
            end_dix = min(n_instance, start_idx + self.BATCH_SIZE)
            sources_batch, destinations_batch = sources[start_idx:end_dix], destinations[start_idx:end_dix]
            edge_idxs_batch, timestamps_batch = edge_idxs[start_idx:end_dix], timestamps[start_idx:end_dix]

            temporal_embedding_batch = self.compute_temporal_embeddings(sources_batch,
                                                                        destinations_batch,
                                                                        timestamps_batch,
                                                                        edge_idxs_batch)

            temporal_embedding[start_idx:end_dix] = temporal_embedding_batch

        pre_class = self.MLP(temporal_embedding).squeeze(dim=1).sigmoid()

        return pre_class

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times, edge_idxs):  # 计算embedding

        """
        compute temporal embeddings for sources and destinations

        :param source_nodes: [batch_size] source id
        :param destination_nodes: [batch size] destination id
        :param edge_times: [batch size] timestamps of interaction
        :param edge_idxs: [batch size] index of interaction
        :param n_neighbors: number of temporal neighbor to consider in each convolutional layer

        :return: Temporal embeddings for sources and destinations
        """

        unique_source, source_id_to_message = self.get_message(source_nodes,
                                                               destination_nodes,
                                                               edge_times,
                                                               edge_idxs)

        self.update_memory(unique_source, source_id_to_message)

        node_embedding = self.compute_embedding(source_nodes=source_nodes,
                                                timestamps=edge_times,
                                                n_layers=self.n_layers,
                                                n_neighbors=self.n_neighbors)

        return node_embedding


    def get_message(self, source_nodes, destination_nodes, edge_times, edge_idxs):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_features[edge_idxs]

        source_memory = self.node_mem[source_nodes]  # get memory
        destination_memory = self.node_mem[destination_nodes]

        source_time_delta = edge_times - self.last_updated[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.to(self.device).unsqueeze(dim=1)).view(
            len(source_nodes), -1)  # time encoding

        source_message = torch.cat([source_memory, destination_memory, edge_features,
                                    source_time_delta_encoding], dim=1)

        message = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            message[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, message

    def update_memory(self, nodes, messages):
        """Message Agreegation: Only keep the last message for each node"""
        unique_nodes = np.unique(nodes)
        unique_messages = []
        unique_timestamps = []

        for node_id in unique_nodes:
            if len(messages[node_id]) > 0:
                # to_update_node_id.append(node_id)
                unique_messages.append(messages[node_id][-1][0])
                unique_timestamps.append(messages[node_id][-1][1])

        unique_messages = torch.stack(unique_messages) if len(unique_nodes) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(unique_nodes) > 0 else []

        memory = self.node_mem[unique_nodes]
        self.last_updated[unique_nodes] = unique_timestamps

        updated_memory = self.memory_updater(unique_messages, memory)

        self.node_mem[unique_nodes] = updated_memory

    def compute_embedding(self, source_nodes, timestamps, n_layers, n_neighbors):
        """Recursive implementation of curr_layers temporal graph attention layers.

        :param source_nodes: [batch size] * 2 sources+destinations.
        :param timestamps: [batch size] * 2
        :param n_layers: number of temporal convolutional layers to stack.
        :param n_neighbors: number of temporal neighbor to consider in each convolutional layer.
        :param time_diffs: time difference of a node compared with its last updated.

        :return:
        """

        assert (n_layers >= 0)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))

        source_node_features = self.temporal_node_features[source_nodes_torch, :]

        source_node_features = self.node_mem[source_nodes_torch, :] + source_node_features

        if n_layers == 0:
            return source_node_features
        else:
            neighbors, edge_idxs, edge_times, edge_weights = self.neighbor_finder.get_temporal_neighbor(
                source_nodes,
                timestamps,
                n_neighbors=n_neighbors)

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
            edge_weights_torch = torch.from_numpy(edge_weights).float().view(len(edge_weights), -1).to(self.device)
            edge_deltas = timestamps[:, np.newaxis] - edge_times
            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors = neighbors.flatten()
            neighbors_embeddings = self.compute_embedding(neighbors,
                                                          np.repeat(timestamps, n_neighbors),
                                                          n_layers=n_layers - 1,
                                                          n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbors_embeddings = neighbors_embeddings.view(len(source_nodes), effective_n_neighbors,
                                                             -1)
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            edge_features = self.edge_features[edge_idxs, :]

            mask = neighbors_torch == 0

            source_embedding = self.aggregate(n_layers, source_node_features,
                                              source_nodes_time_embedding,
                                              neighbors_embeddings,
                                              edge_time_embeddings,
                                              edge_weights_torch,
                                              edge_features,
                                              mask)

            return source_embedding

    def aggregate(self, n_layer, source_node_features, souce_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_weights, edge_features, mask):

        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                       dim=2)
        neighbors_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(
            torch.sum(torch.mul(neighbors_embeddings, edge_weights.unsqueeze(2)), dim=1))

        source_features = torch.cat([source_node_features,
                                     souce_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding


class TimeEncode(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension, device):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.device = device
        self.w = torch.nn.Linear(1, dimension).to(self.device)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))).to(self.device)
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float().to(self.device))

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)

        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))

        return output
