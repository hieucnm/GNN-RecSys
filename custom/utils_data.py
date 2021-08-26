import os
import pickle

import dgl
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_ids(df: pd.DataFrame, id_column, posfix='idx') -> pd.DataFrame:
    id_new_col = f"{id_column}_{posfix}"
    id_map_df = pd.DataFrame(df[id_column].unique(), columns=[id_column])
    id_map_df[id_new_col] = id_map_df.index
    df = df.merge(id_map_df, on=id_column)
    return df


def create_common_ids(df_list, id_columns, suffix='idx'):
    """
    Similar to create_ids function, but processing for list of dataframes on a common columns if exist (e.g `user_id`)
    """

    id_set = set()
    _ = [id_set.update(df[id_column]) for df in df_list
         for id_column in id_columns if id_column in df.columns]

    key_id = id_columns[0]
    id_map_df = pd.DataFrame(sorted(id_set), columns=[key_id])
    id_map_df[f"{key_id}_{suffix}"] = id_map_df.index

    df_list_res = []
    for df in df_list:
        for id_column in id_columns:
            if id_column not in df.columns:
                continue
            if id_column not in id_map_df:
                df = df.merge(
                    id_map_df.rename(columns={key_id: id_column, f'{key_id}_{suffix}': f'{id_column}_{suffix}'
                                              }), on=id_column)
            else:
                df = df.merge(id_map_df, on=key_id)
        df_list_res.append(df)
    return df_list_res, id_map_df


def read_data(file_path):
    """
    Generic function to read any kind of data. Extensions supported: '.gz', '.csv', '.pkl'
    """
    if isinstance(file_path, pd.DataFrame):
        return file_path

    if file_path.endswith('.gz'):
        obj = pd.read_csv(file_path, compression='gzip',
                          header=0, sep=';', quotechar='"',
                          error_bad_lines=False)
    elif file_path.endswith('.csv'):
        obj = pd.read_csv(file_path)
    elif file_path.endswith('.parquet') or os.path.isdir(file_path):
        obj = pd.read_parquet(file_path)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as handle:
            obj = pickle.load(handle)
    else:
        raise KeyError('File extension of {} not recognized.'.format(file_path))
    return obj


def remove_label_edges(graph, label_edge_types):
    # remove label_edge_types to avoid data leakage when aggregating
    train_eid_dict = {}
    train_graph = graph.clone()
    for e_type in label_edge_types:
        train_eid_dict[e_type] = torch.arange(graph.number_of_edges(e_type))
        train_graph.remove_edges(train_eid_dict[e_type], etype=e_type)
    return train_graph, train_eid_dict


def get_edge_loader(graph,
                    label_edge_types,
                    **params,
                    ):
    train_graph, train_eid_dict = remove_label_edges(graph, label_edge_types)
    sampler = get_neighbor_sampler(n_layer=params['n_layers'] - 1, n_neighbor=params['num_neighbors'])
    sampler_n = dgl.dataloading.negative_sampler.Uniform(params['neg_sample_size'])

    edge_param = {
        'g': graph,
        'eids': train_eid_dict,
        'g_sampling': train_graph,
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


def get_neighbor_sampler(n_layer, n_neighbor):
    if n_neighbor == 0:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layer)
    else:
        sampler = dgl.dataloading.MultiLayerNeighborSampler([n_neighbor] * n_layer, replace=False)
    return sampler


def get_node_loader(graph, label_edge_types, item_id, sample_size=None, **params):
    """
    Get node loader for given edge_types, and corresponding ground truth
    Parameters
    ----------
    graph
    label_edge_types
    sample_size
    item_id
    params

    Returns
    -------

    """
    train_graph, train_eid_dict = remove_label_edges(graph, label_edge_types)
    all_user_nodes = []
    all_item_nodes = []
    for edge_type in label_edge_types:
        user_nodes, item_nodes = graph.find_edges(train_eid_dict[edge_type], etype=edge_type)
        if sample_size is not None:
            # TODO: stratified split by item_nodes (`ad_cate`)
            _, user_nodes, _, item_nodes = train_test_split(user_nodes, item_nodes, test_size=sample_size)
        all_user_nodes += user_nodes.tolist()
        all_item_nodes += item_nodes.tolist()
    ground_truth = (np.array(all_user_nodes), np.array(all_item_nodes))
    unique_user_nodes = np.unique(all_user_nodes)
    unique_item_nodes = np.arange(graph.num_nodes(item_id))

    sampler = get_neighbor_sampler(n_layer=params['n_layers'] - 1, n_neighbor=params['num_neighbors'])
    node_param = {
        'g': train_graph,
        'nids': {'user': unique_user_nodes, 'item': unique_item_nodes},
        'block_sampler': sampler,
        'batch_size': params['node_batch_size'],
        'shuffle': False,
        'drop_last': False,
        'num_workers': params['num_workers'],
    }
    node_loader = dgl.dataloading.NodeDataLoader(**node_param)
    return node_loader, ground_truth


def get_sub_train(graph, label_edge_types, sample_size):

    # Get train edge ids
    _, train_eid_dict = remove_label_edges(graph, label_edge_types)
        
    # Generate inference nodes for mini-train & ground truth for mini-train
    
    # Step 1: Choose the subsample of training set
    # For simplicity, only use the first type train edge types, which should be `will-convert`
    train_uid, train_iid = graph.find_edges(train_eid_dict[label_edge_types[0]], etype=label_edge_types[0])
    unique_train_uid = np.unique(train_uid)
    sub_train_uid = np.random.choice(unique_train_uid, int(len(unique_train_uid) * sample_size), replace=False)
    
    # Step 2: Fetch uid and iid of sub-train sample for all edge types
    sub_train_uid_all = []
    sub_train_iid_all = []
    for e_type in train_eid_dict:
        train_uid, train_iid = graph.find_edges(train_eid_dict[e_type], etype=e_type)
        sub_train_eid = []
        for i in range(len(train_eid_dict[e_type])):
            if train_uid[i].item() in sub_train_uid:
                sub_train_eid.append(train_eid_dict[e_type][i].item())
        sub_train_uid, sub_train_iid = graph.find_edges(sub_train_eid, etype=e_type)
        sub_train_uid_all.extend(sub_train_uid.tolist())
        sub_train_iid_all.extend(sub_train_iid.tolist())
    ground_truth_sub_train = (np.array(sub_train_uid_all), np.array(sub_train_iid_all))
    sub_train_uid = np.array(np.unique(sub_train_uid_all))
    return sub_train_uid, ground_truth_sub_train
