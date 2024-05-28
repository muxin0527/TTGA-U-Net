import json
import os
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from dgl import DGLGraph


class Dataset(object):

    def __init__(self, mode, thickness=None, path=None):
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        self.thickness = thickness
        self.path = path
        self._load()
        self._preprocess()

    def _load(self):
        path = self.path + '/dgl_graph_info_' + str(self.thickness)
        print(path)
        if self.mode == 'train':
            with open(os.path.join(path, 'train_graph.json')) as jsonfile:
                g_data = json.load(jsonfile)
            self.labels = np.load(os.path.join(path, 'train_labels.npy'.format(dir)))
            self.features = np.load(os.path.join(path, 'train_features.npy'.format(dir)))
            self.graph = DGLGraph(nx.Graph(json_graph.node_link_graph(g_data)))
            self.graph_id = np.load(os.path.join(path, 'train_graph_id.npy'.format(dir)))

        if self.mode == 'valid':
            with open(os.path.join(path, 'valid_graph.json'.format(dir))) as jsonfile:
                g_data = json.load(jsonfile)
            self.labels = np.load(os.path.join(path, 'valid_labels.npy'.format(dir)))
            self.features = np.load(os.path.join(path, 'valid_features.npy'.format(dir)))
            self.graph = DGLGraph(nx.Graph(json_graph.node_link_graph(g_data)))
            self.graph_id = np.load(os.path.join(path, 'valid_graph_id.npy'.format(dir)))

        if self.mode == 'test':
            with open(os.path.join(path, 'test_graph.json'.format(dir))) as jsonfile:
                g_data = json.load(jsonfile)
            self.labels = np.load(os.path.join(path, 'test_labels.npy'.format(dir)))
            self.features = np.load(os.path.join(path, 'test_features.npy'.format(dir)))
            self.graph = DGLGraph(nx.Graph(json_graph.node_link_graph(g_data)))
            self.graph_id = np.load(os.path.join(path, 'test_graph_id.npy'.format(dir)))

    def _preprocess(self):
        if self.mode == 'train':
            self.train_mask_list = []
            self.train_graphs = []
            self.train_labels = []
            print(len(set(self.graph_id)))
            for train_graph_id in set(self.graph_id):
                train_graph_mask = np.where(self.graph_id == train_graph_id)[0]
                self.train_mask_list.append(train_graph_mask)
                self.train_graphs.append(self.graph.subgraph(train_graph_mask))
                self.train_labels.append(self.labels[train_graph_mask])
        if self.mode == 'valid':
            self.valid_mask_list = []
            self.valid_graphs = []
            self.valid_labels = []
            for valid_graph_id in set(self.graph_id):
                valid_graph_mask = np.where(self.graph_id == valid_graph_id)[0]
                self.valid_mask_list.append(valid_graph_mask)
                self.valid_graphs.append(self.graph.subgraph(valid_graph_mask))
                self.valid_labels.append(self.labels[valid_graph_mask])
        if self.mode == 'test':
            self.test_mask_list = []
            self.test_graphs = []
            self.test_labels = []
            for test_graph_id in set(self.graph_id):
                test_graph_mask = np.where(self.graph_id == test_graph_id)[0]
                self.test_mask_list.append(test_graph_mask)
                self.test_graphs.append(self.graph.subgraph(test_graph_mask))
                self.test_labels.append(self.labels[test_graph_mask])

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_graphs)
        if self.mode == 'valid':
            return len(self.valid_graphs)
        if self.mode == 'test':
            return len(self.test_mask_list)

    def __getitem__(self, item):
        if self.mode == 'train':
            g = self.train_graphs[item]
            g.ndata['feat'] = self.features[self.train_mask_list[item]]
            label = self.train_labels[item]
        elif self.mode == 'valid':
            g = self.valid_graphs[item]
            g.ndata['feat'] = self.features[self.valid_mask_list[item]]
            label = self.valid_labels[item]
        elif self.mode == 'test':
            g = self.test_graphs[item]
            g.ndata['feat'] = self.features[self.test_mask_list[item]]
            label = self.test_labels[item]
        return g, label


class MRDataset(Dataset):

    def __getitem__(self, item):
        if self.mode == 'train':
            return self.train_graphs[item], self.features[self.train_mask_list[item]], self.train_labels[item]
        if self.mode == 'valid':
            return self.valid_graphs[item], self.features[self.valid_mask_list[item]], self.valid_labels[item]
        if self.mode == 'test':
            return self.test_graphs[item], self.features[self.test_mask_list[item]], self.test_labels[item]