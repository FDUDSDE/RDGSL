import random
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, edge_weights, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.edge_weights = edge_weights
        self.labels = labels
        self.n_interactions = len(sources)


class Dataset:
    def __init__(self, root, name, args):
        self.root = root
        self.np_file = args.np_file
        self.dataset_used = args.dataset

        self.name = name
        self.node_set = set()
        self.degrees = dict()

        self.node_time_nodes = dict()

        self.max_d_time = -sys.maxsize

        self.time_stamp = []
        self.time_edges_dict = {}
        self.time_nodes_dict = {}

        self.node2hist = dict()
        self.node_dim = 0
        self.index = 0

        self.sources = []
        self.destinations = []
        self.timestamps = []
        self.edge_idxs = []
        self.edge_weights = []
        self.edge_features = None

        self.s_his_nodes = []
        self.t_his_nodes = []
        self.s_his_times = []
        self.t_his_times = []
        self.s_his_idxs = []
        self.t_his_idxs = []
        self.s_his_froms = []
        self.t_his_froms = []
        self.s_his_masks = []
        self.t_his_masks = []

        self.labels = []
        self.n_class = -1
        self.n_nei = args.neighbor_size
        self.emb_size = args.emb_size

        self.sources, self.destinations, self.timestamps, self.edge_idxs, self.edge_weights = self.getdata()

        self.node_features = np.random.uniform(-1., 1., (self.node_dim, self.emb_size))

    def getdata(self, ):
        print('loading data...')
        # loading edge_features
        edge_features = np.load(self.np_file)
        self.edge_features = edge_features[1:]

        # loading data
        with open(self.root, 'r') as infile:
            self.index = 0
            for line in infile:
                parts = line.strip().split(',')

                s_node = int(parts[1]) - 1
                t_node = int(parts[2]) - 1
                d_time = float(parts[3])
                node_label = float(parts[4])

                self.labels.append(node_label)

                self.node_set.update([s_node, t_node])

                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                if t_node not in self.node2hist:
                    self.node2hist[t_node] = list()
                self.node2hist[s_node].append((d_time, t_node, self.index))
                self.node2hist[t_node].append((d_time, s_node, self.index))

                if s_node not in self.node_time_nodes:
                    self.node_time_nodes[s_node] = dict()
                if t_node not in self.node_time_nodes:
                    self.node_time_nodes[t_node] = dict()
                if d_time not in self.node_time_nodes[s_node]:
                    self.node_time_nodes[s_node][d_time] = list()
                if d_time not in self.node_time_nodes[t_node]:
                    self.node_time_nodes[t_node][d_time] = list()
                self.node_time_nodes[s_node][d_time].append(t_node)
                self.node_time_nodes[t_node][d_time].append(s_node)

                if d_time > self.max_d_time:
                    self.max_d_time = d_time

                self.degrees[s_node] += 1
                self.degrees[t_node] += 1

                self.time_stamp.append(d_time)

                if d_time not in self.time_edges_dict:
                    self.time_edges_dict[d_time] = []
                self.time_edges_dict[d_time].append((s_node, t_node))

                if d_time not in self.time_nodes_dict:
                    self.time_nodes_dict[d_time] = []
                self.time_nodes_dict[d_time].append(s_node)
                self.time_nodes_dict[d_time].append(t_node)

                self.sources.append(s_node)
                self.destinations.append(t_node)
                self.timestamps.append(d_time)
                self.edge_idxs.append(self.index)
                self.index = self.index + 1

            self.n_class = len(set(self.labels))
            self.labels = torch.tensor(self.labels, dtype=torch.float)

            self.time_stamp = sorted(list(set(self.time_stamp)))

            self.node_dim = len(self.node_set)
            self.data_size = 0

            for s in self.node2hist:
                hist = self.node2hist[s]
                hist = sorted(hist, key=lambda x: x[0])
                self.node2hist[s] = hist
                self.data_size += len(self.node2hist[s])

            self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
            self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
            idx = 0
            for s_node in self.node2hist:
                for t_idx in range(len(self.node2hist[s_node])):
                    self.idx2source_id[idx] = s_node
                    self.idx2target_id[idx] = t_idx
                    idx += 1

            for idx in range(len(self.sources)):
                s_node = self.idx2source_id[idx]
                t_idx = self.idx2target_id[idx]
                t_node = self.node2hist[s_node][t_idx][1]
                e_time = self.node2hist[s_node][t_idx][0]

                if t_idx - self.n_nei < 0:
                    s_his = self.node2hist[s_node][0:t_idx]
                else:
                    s_his = self.node2hist[s_node][
                            t_idx - self.n_nei: t_idx]

                t_his_list = self.node2hist[t_node]
                s_idx = 0
                for i in range(len(t_his_list)):
                    if (t_his_list[i][1] == s_node and t_his_list[i][0] == e_time):
                        s_idx = i
                        break
                if s_idx - self.n_nei < 0:
                    t_his = t_his_list[:s_idx]
                else:
                    t_his = t_his_list[s_idx - self.n_nei: s_idx]

                s_his_node = np.zeros((self.n_nei,))
                s_his_node[:len(s_his)] = [h[1] for h in s_his]
                s_his_time = np.zeros((self.n_nei,))
                s_his_time[:len(s_his)] = [h[0] for h in s_his]
                s_his_mask = np.zeros((self.n_nei,))
                s_his_mask[:len(s_his)] = 1.

                t_his_node = np.zeros((self.n_nei,))
                t_his_node[:len(t_his)] = [h[1] for h in t_his]
                t_his_time = np.zeros((self.n_nei,))
                t_his_time[:len(t_his)] = [h[0] for h in t_his]
                t_his_mask = np.zeros((self.n_nei,))
                t_his_mask[:len(t_his)] = 1.

                self.s_his_nodes.append(s_his_node)
                self.t_his_nodes.append(t_his_node)
                self.s_his_times.append(s_his_time)
                self.t_his_times.append(t_his_time)
                self.s_his_masks.append(s_his_mask)
                self.t_his_masks.append(t_his_mask)

            self.max_nei_len = max(map(lambda x: len(x), self.node2hist.values()))
            print('#nodes: {}, #edge: {}, #time_stamp: {}'.format(self.node_dim, self.index, len(self.time_stamp)))
            print('avg_degree: {}'.format(sum(self.degrees.values()) / len(self.degrees)))
            print('max neighbors length: {}'.format(self.max_nei_len))

            return self.sources, self.destinations, self.timestamps, self.edge_idxs, self.edge_weights

    def get_data_node_classification(self, ):
        timestamps = np.array(self.timestamps)
        edge_idxs = np.array(self.edge_idxs)

        val_time, test_time = list(np.quantile(timestamps, [0.70, 0.85]))

        random.seed(2020)

        train_mask = timestamps <= val_time
        test_mask = timestamps > test_time
        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)

        return edge_idxs[train_mask], edge_idxs[val_mask], edge_idxs[test_mask]


def get_neighbor_finder(sources, destinations, timestamps, edge_idxs, edge_weights, adj_list, uniform, max_node_idx=None):
    for source, destination, edge_idx, timestamp, edge_weight in zip(sources, destinations,
                                                                     edge_idxs,
                                                                     timestamps, edge_weights):
        adj_list[source].append((destination, edge_idx, timestamp, edge_weight))
        adj_list[destination].append((source, edge_idx, timestamp, edge_weight))

    return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:

    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        self.node_to_edge_weights = []

        for neighbors in adj_list:
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))
            self.node_to_edge_weights.append(np.array([x[3] for x in sorted_neighhbors]))

        self.uniform = uniform

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], \
               self.node_to_edge_timestamps[src_idx][:i], self.node_to_edge_weights[src_idx][:i]

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):  # 给定节点和时间戳，返回邻居、边id、时间戳(已经按时间顺序排好了)
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_weights = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.float32)

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times, \
            source_edge_weights = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]
                    edge_weights[i, :] = source_edge_weights[sampled_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                    edge_weights[i, :] = edge_weights[i, :][pos]
                else:
                    # Take most recent interactions
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]
                    source_edge_weights = source_edge_weights[-n_neighbors:]

                    assert (len(source_neighbors) <= n_neighbors)
                    assert (len(source_edge_times) <= n_neighbors)
                    assert (len(source_edge_idxs) <= n_neighbors)
                    assert (len(source_edge_weights) <= n_neighbors)

                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
                    edge_weights[i, n_neighbors - len(source_edge_weights):] = source_edge_weights

        return neighbors, edge_idxs, edge_times, edge_weights


def accuracy(output, labels):  # accuracy 编写
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_account(output, labels):  # accuracy 编写
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def auc(output, labels):
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels).to('cuda')
    preds = output.max(1)[1].type_as(labels)

    auc_score = roc_auc_score(labels.cpu(), preds.cpu())

    return auc_score
