import os
import pickle

import dgl
import torch
import pandas as pd


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


def get_edge_data_loaders(graph,
                          train_edge_types,
                          num_neighbors,
                          **params,
                          ):
    train_eid_dict = {}
    train_graph = graph.clone()
    for e_type in train_edge_types:
        train_eid_dict[e_type] = torch.arange(graph.number_of_edges(e_type))
        train_graph.remove_edges(train_eid_dict[e_type], etype=e_type)

    n_layers = params['n_layers'] - 1  # except the embedding layer

    if num_neighbors == 0:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    else:
        sampler = dgl.dataloading.MultiLayerNeighborSampler([num_neighbors] * n_layers, replace=False)

    sampler_n = dgl.dataloading.negative_sampler.Uniform(params['neg_sample_size'])

    edge_param_train = {
        'g': graph,
        'eids': train_eid_dict,
        'g_sampling': train_graph,
        'block_sampler': sampler,
        'negative_sampler': sampler_n,
        'batch_size': params['batch_size'],
        'shuffle': params['shuffle'],  # set to False when debugging
        'num_workers': params['num_workers'],
        'drop_last': False,
        'pin_memory': True,
    }

    if params['use_ddp']:
        edge_param_train.update({'use_ddp': params['use_ddp']})

    edge_loader_train = dgl.dataloading.EdgeDataLoader(**edge_param_train)
    return edge_loader_train
