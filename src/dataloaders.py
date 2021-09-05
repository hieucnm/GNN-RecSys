import dgl
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from dgl.dataloading.negative_sampler import _BaseNegativeSampler
from dgl.dataloading import EdgeDataLoader, NodeDataLoader


# ====================
# Negative Sampler ===

def get_link_dict(src_tensor, dst_tensor):
    return pd.DataFrame({'src': src_tensor, 'dst': dst_tensor}) \
        .groupby('src')['dst'].apply(lambda x: list(set(x))) \
        .to_dict()  # type = dict[int: list[int]]


def get_unlink_dict(src_tensor, dst_tensor, num_dst):
    all_dst = set(range(num_dst))
    return pd.DataFrame({'src': src_tensor, 'dst': dst_tensor}) \
        .groupby('src')['dst'] \
        .apply(lambda x: list(all_dst.difference(x))) \
        .to_dict()  # type = dict[int: list[int]]


class IsolateBasedNegativeSampler(_BaseNegativeSampler):
    def __init__(self, graph, canonical_etypes, sample_size):
        self.sample_size = sample_size
        neg_dict = dict()
        for e_type in canonical_etypes:
            src, dst = graph.edges(etype=e_type)
            num_dst = graph.num_nodes(ntype=e_type[2])
            neg_dict[e_type] = get_unlink_dict(src, dst, num_dst)
        self.neg_dict = neg_dict

    def _get_neg_dst(self, nid_list, e_type):
        return torch.from_numpy(np.concatenate([
            np.random.choice(self.neg_dict[e_type][nid], self.sample_size, replace=True)
            for i, nid in enumerate(nid_list)
        ]))

    def _generate(self, g, eids, canonical_etype):
        src, pos_dst = g.find_edges(eids, etype=canonical_etype)
        neg_dst = self._get_neg_dst(src.tolist(), e_type=canonical_etype)
        src = src.repeat(self.sample_size)
        return src, neg_dst


class EdgeBasedNegativeSampler(_BaseNegativeSampler):
    def __init__(self, graph, canonical_etypes, sample_size):
        # TODO: for now, only use 1 edge type as negative
        etype = canonical_etypes[0]
        src, dst = graph.edges(etype=etype)
        self.neg_dict = get_link_dict(dst, src)
        self.sample_size = sample_size

    def _generate(self, g, eids, canonical_etype):
        pos_src, dst = g.find_edges(eids, etype=canonical_etype)
        neg_src = self._get_neg_dst(dst.tolist())
        dst = dst.repeat(self.sample_size)
        return neg_src, dst

    def _verify_nid_exist(self, nid_list):
        # if alert, means every users who clicked this ad also converted
        diff = set(nid_list).difference(self.neg_dict.keys())
        assert len(diff) == 0, f"These nodes have no negative label to sample: {diff}"

    def _get_neg_dst(self, nid_list):
        self._verify_nid_exist(nid_list)
        return torch.from_numpy(np.concatenate([
            np.random.choice(self.neg_dict[nid], size=self.sample_size, replace=True)
            for nid in nid_list
        ]))


# ====================
# Edge Data Loader ===


def get_neighbor_sampler(n_layers, n_neighbors):
    if n_neighbors > 0:
        return dgl.dataloading.MultiLayerNeighborSampler([n_neighbors] * n_layers, replace=False)
    return dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)


def get_negative_sampler(name, sample_size, graph=None, pos_etypes=None, neg_etypes=None):
    if name == 'edge':
        return EdgeBasedNegativeSampler(graph, neg_etypes, sample_size)
    elif name == 'link':
        return IsolateBasedNegativeSampler(graph, pos_etypes, sample_size)
    return dgl.dataloading.negative_sampler.Uniform(sample_size)


class EdgeLoaderPlus(EdgeDataLoader):
    def __init__(self, graph, adjust_graph, pos_label_etypes, neg_label_etypes=None, **params):
        label_eid_dict = {
            e_type: torch.arange(graph.number_of_edges(e_type))
            for e_type in pos_label_etypes
        }
        sampler = get_neighbor_sampler(params['n_layers'] - 1, params['n_neighbors'])
        sampler_n = get_negative_sampler(name=params['sampler_n'],
                                         graph=graph,
                                         sample_size=params['neg_sample_size'],
                                         pos_etypes=pos_label_etypes,
                                         neg_etypes=neg_label_etypes
                                         )
        print('- EdgeDataLoader using {} and {}'.format(sampler.__class__, sampler_n.__class__))

        edge_param = {
            'g': graph,
            'eids': label_eid_dict,
            'g_sampling': adjust_graph,
            'block_sampler': sampler,
            'negative_sampler': sampler_n,
            'batch_size': params['edge_batch_size'],
            'shuffle': False,  # set to False when debugging
            'num_workers': params['num_workers'],
            'drop_last': False,
            'pin_memory': True,
        }
        if params['use_ddp']:
            edge_param.update({'use_ddp': params['use_ddp']})
        super(EdgeLoaderPlus, self).__init__(**edge_param)


# ====================
# Node Data Loader ===

def get_user_nodes_and_grountruth(graph, user_id, etypes=None, sample_size=None):
    if etypes is not None:
        label_eid_dict = {
            e_type: torch.arange(graph.number_of_edges(e_type))
            for e_type in etypes
        }
        all_user_nodes = []
        groundtruth_dict = dict()
        for edge_type, eid in label_eid_dict.items():
            user_nodes, item_nodes = graph.find_edges(eid, etype=edge_type)
            if sample_size is not None:
                _, user_nodes, _, item_nodes = train_test_split(user_nodes,
                                                                item_nodes,
                                                                test_size=sample_size,
                                                                stratify=item_nodes)
            all_user_nodes += user_nodes.tolist()
            groundtruth_dict[edge_type] = list(zip(user_nodes.tolist(), item_nodes.tolist()))
        unique_user_nodes = np.unique(all_user_nodes)
    else:
        # non etypes means no label, we are predicting
        groundtruth_dict = None
        unique_user_nodes = np.arange(graph.num_nodes(user_id))
    return unique_user_nodes, groundtruth_dict


class NodeLoaderPlus(NodeDataLoader):
    def __init__(self, graph, adjust_graph, user_id, item_id, e_types, sample_size=None, **params):
        unique_user_nodes, groundtruth_dict = get_user_nodes_and_grountruth(graph, user_id, e_types, sample_size)
        unique_item_nodes = np.arange(graph.num_nodes(item_id))
        sampler = get_neighbor_sampler(n_layers=params['n_layers'] - 1, n_neighbors=params['n_neighbors'])
        node_param = {
            'g': adjust_graph,
            'nids': {user_id: unique_user_nodes, item_id: unique_item_nodes},
            'block_sampler': sampler,
            'batch_size': params['node_batch_size'],
            'shuffle': False,
            'drop_last': False,
            'num_workers': params['num_workers'],
        }
        self.groundtruth_dict = groundtruth_dict
        super(NodeLoaderPlus, self).__init__(**node_param)


class ItemNodeLoaderPlus(NodeDataLoader):
    def __init__(self, adjust_graph, item_id, **params):
        unique_item_nodes = np.arange(adjust_graph.num_nodes(item_id))
        sampler = get_neighbor_sampler(n_layers=params['n_layers'] - 1, n_neighbors=params['n_neighbors'])
        node_param = {
            'g': adjust_graph,
            'nids': {item_id: unique_item_nodes},
            'block_sampler': sampler,
            'batch_size': params['node_batch_size'],
            'shuffle': False,
            'drop_last': False,
            'num_workers': params['num_workers'],
        }
        super(ItemNodeLoaderPlus, self).__init__(**node_param)


class UserNodeLoaderPlus(NodeDataLoader):
    def __init__(self, graph, to_infer_user_nid, user_id, **params):
        user_nodes = np.asarray(to_infer_user_nid)
        sampler = get_neighbor_sampler(n_layers=params['n_layers'] - 1, n_neighbors=params['n_neighbors'])
        node_param = {
            'g': graph,
            'nids': {user_id: user_nodes},
            'block_sampler': sampler,
            'batch_size': params['node_batch_size'],
            'shuffle': False,
            'drop_last': False,
            'num_workers': params['num_workers'],
        }
        super(UserNodeLoaderPlus, self).__init__(**node_param)
