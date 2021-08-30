import dgl
import numpy as np
from sklearn.model_selection import train_test_split


def get_neighbor_sampler(n_layer, n_neighbor):
    if n_neighbor == 0:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layer)
    else:
        sampler = dgl.dataloading.MultiLayerNeighborSampler([n_neighbor] * n_layer, replace=False)
    return sampler


def get_edge_loader(graph,
                    adjust_graph,
                    label_eid_dict,
                    **params,
                    ):
    sampler = get_neighbor_sampler(n_layer=params['n_layers'] - 1, n_neighbor=params['num_neighbors'])
    sampler_n = dgl.dataloading.negative_sampler.Uniform(params['neg_sample_size'])

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
    train_edge_loader = dgl.dataloading.EdgeDataLoader(**edge_param)
    return train_edge_loader


def get_node_loader(graph,
                    adjust_graph,
                    user_id,
                    item_id,
                    label_eid_dict=None,
                    sample_size=None,
                    **params):
    """
    Get node loader for given edge_types, and corresponding ground truth
    Parameters
    ----------
    user_id
    graph
    adjust_graph
    label_eid_dict
    sample_size
    item_id
    params

    Returns
    -------

    """
    if label_eid_dict is not None:
        all_user_nodes = []
        all_item_nodes = []
        for edge_type, eid in label_eid_dict.items():
            user_nodes, item_nodes = graph.find_edges(eid, etype=edge_type)
            if sample_size is not None:
                # TODO: stratified split by item_nodes (`ad_cate`)
                _, user_nodes, _, item_nodes = train_test_split(user_nodes, item_nodes, test_size=sample_size)
            all_user_nodes += user_nodes.tolist()
            all_item_nodes += item_nodes.tolist()
        ground_truth = list(zip(all_user_nodes, all_item_nodes))
        unique_user_nodes = np.unique(all_user_nodes)
    else:
        # no label_eid_dict means no label, we are predicting
        ground_truth = None
        unique_user_nodes = np.arange(graph.num_nodes(user_id))

    unique_item_nodes = np.arange(graph.num_nodes(item_id))

    sampler = get_neighbor_sampler(n_layer=params['n_layers'] - 1, n_neighbor=params['num_neighbors'])
    node_param = {
        'g': adjust_graph,
        'nids': {user_id: unique_user_nodes, item_id: unique_item_nodes},
        'block_sampler': sampler,
        'batch_size': params['node_batch_size'],
        'shuffle': False,
        'drop_last': False,
        'num_workers': params['num_workers'],
    }
    node_loader = dgl.dataloading.NodeDataLoader(**node_param)
    return node_loader, ground_truth


def get_item_node_loader(adjust_graph, item_id, **params):
    unique_item_nodes = np.arange(adjust_graph.num_nodes(item_id))
    sampler = get_neighbor_sampler(n_layer=params['n_layers'] - 1, n_neighbor=params['num_neighbors'])
    node_param = {
        'g': adjust_graph,
        'nids': {item_id: unique_item_nodes},
        'block_sampler': sampler,
        'batch_size': params['node_batch_size'],
        'shuffle': False,
        'drop_last': False,
        'num_workers': params['num_workers'],
    }
    node_loader = dgl.dataloading.NodeDataLoader(**node_param)
    return node_loader


# TODO: For now we use a uniform negative sampler for data loaders,
#  but what if we use a sampler that give (src_id, ad_cate) that will click but won't convert ???
#  Implement the class below to do that, and pass as `sampler_n` into data loaders
#  Source: https://docs.dgl.ai/en/0.6.x/guide/minibatch-link.html
#  If do this, change the way we evaluate model in the `Evaluator`

class NegativeSampler(object):
    def __init__(self, graph, sample_size, eid_dict):
        self.weights = {
            e_type: graph.in_degrees(etype=e_type).float() ** 0.75
            for _, e_type, _ in graph.canonical_etypes
        }
        self.sample_size = sample_size
        self.eid_dict = eid_dict

    def __call__(self, graph, eid_dict):
        result_dict = {}
        for e_type, eid in eid_dict.items():
            src, _ = graph.find_edges(eid, etype=e_type)
            src = src.repeat_interleave(self.sample_size)
            dst = self.weights[e_type].multinomial(len(src), replacement=True)
            result_dict[e_type] = (src, dst)
        return result_dict
