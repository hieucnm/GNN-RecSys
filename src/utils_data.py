import json
import os
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import datetime as dt
import re


# ==========================
# Utils for reading data ===

def create_ids(df: pd.DataFrame, id_column, suffix='idx'):
    id_new_col = f"{id_column}_{suffix}"
    id_map_df = pd.DataFrame(df[id_column].unique(), columns=[id_column])
    id_map_df[id_new_col] = pd.Series(id_map_df.index).astype(np.int32)
    df = df.merge(id_map_df, on=id_column)
    return df, id_map_df


def create_common_ids(df_list, id_columns, suffix='idx'):
    """
    Similar to create_ids function, but processing for list of dataframes on a common columns if exist (e.g `user_id`)
    """

    id_set = set()
    _ = [id_set.update(df[id_column]) for df in df_list
         for id_column in id_columns if id_column in df.columns]

    key_id = id_columns[0]
    id_map_df = pd.DataFrame(sorted(id_set), columns=[key_id])
    id_map_df[f"{key_id}_{suffix}"] = pd.Series(id_map_df.index).astype(np.int32)

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


def create_common_user_ids(user_profile, df_list, id_columns, suffix='idx'):
    key_id = id_columns[0]
    user_profile[f'{key_id}_{suffix}'] = user_profile.index
    id_map_df = user_profile[[key_id, f'{key_id}_{suffix}']]

    df_list_res = []
    for df in df_list:
        for id_column in id_columns:
            if id_column not in df.columns:
                continue
            if id_column not in id_map_df.columns:
                df = df.merge(
                    id_map_df.rename(columns={key_id: id_column, f'{key_id}_{suffix}': f'{id_column}_{suffix}'
                                              }), on=id_column)
            else:
                df = df.merge(id_map_df, on=key_id)
        df_list_res.append(df)
    return user_profile, df_list_res


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
    elif file_path.endswith('json'):
        obj = json.load(open(file_path))
    elif file_path.endswith('pkl'):
        obj = pickle.load(open(file_path, 'rb'))
    else:
        raise KeyError('File extension of {} not recognized.'.format(file_path))
    return obj


def rename_id(df, suffix, id_cols):
    for col in id_cols:
        if col in df.columns:
            # df[f'{col}_raw'] = df[col]
            df[col] = df[col].astype(str) + f'_{suffix}'
    return df


def read_data_rename_id(dir_list, filename, id_cols):
    df_list = []
    for index, data_dir in enumerate(dir_list):
        df = read_data(f'{data_dir}/{filename}')
        df = rename_id(df, index, id_cols)
        df_list.append(df)
    return pd.concat(df_list).reset_index(drop=True)


# ==============================
# Utils for processing graph ===

def get_label_edges(graph, label_edge_types):
    label_eid_dict = {}
    for e_type in label_edge_types:
        label_eid_dict[e_type] = torch.arange(graph.number_of_edges(e_type))
    return label_eid_dict


# noinspection SpellCheckingInspection
def remove_label_edges(graph, label_edge_types):
    """
    Remove label_edge_types to avoid data leakage when aggregating
    Parameters
    ----------
    graph
    label_edge_types

    Returns
    -------

    """
    # Method 1: clone then remove.
    # Issue: remove edge_ids but edge_name still remain
    adjust_graph = graph.clone()
    for e_type in label_edge_types:
        label_edge_ids = torch.arange(graph.number_of_edges(e_type))
        adjust_graph.remove_edges(label_edge_ids, etype=e_type)

    # Method 2: migrate edges to new graph except the label ones.
    # Issue: `graph` & `adjust_graph` have different schema, which raise error
    # adjust_schema = {}
    # for edge_type in graph.canonical_etypes:
    #     if edge_type not in label_edge_types:
    #         src_nodes, dst_nodes, _ = graph.edges(form='all', etype=edge_type)
    #         adjust_schema[edge_type] = (src_nodes, dst_nodes)
    # adjust_graph = heterograph(adjust_schema)
    return adjust_graph


# =========================
# Utils for saving data ===

def mkdir_if_missing(path: str, _type: str = 'path'):
    assert _type in ['path', 'dir'], 'type must be `path` or `dir`'
    if _type == 'path':
        dir_path = osp.dirname(path)
    else:
        dir_path = path
    if not osp.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return True
    return False


def seed_everything():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# noinspection PyBroadException
def get_date(path):
    try:
        return re.search("(\d{4}/\d{2}/\d{2})", path).group(0)
    except:
        return None


def dir_with_timestamp(root_dir, data_dir=None):
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_date = get_date(data_dir)
    if data_date:
        return f'{root_dir}/{timestamp}/{data_date}'
    return f'{root_dir}/{timestamp}'


# noinspection SpellCheckingInspection
def save_plots(metrics, save_dir):
    """
    Visualize train & validation loss & metrics.
    """

    save_dir = f'{save_dir}/plots'
    mkdir_if_missing(save_dir, _type='dir')

    json.dump(metrics, open(f'{save_dir}/learning_metrics.json', 'w'))

    for metric_name, metric_dict in metrics.items():
        fig = plt.figure()
        plt.title(metric_name)
        fig.tight_layout()
        plt.rcParams["axes.titlesize"] = 6

        epochs = np.arange(len(metric_dict[list(metric_dict.keys())[0]]))
        for data_set, metric_list in metric_dict.items():
            plt.plot(epochs, metric_list)
        plt.legend(list(metric_dict.keys()))
        plt.savefig(f'{save_dir}/{metric_name}.png')
        plt.close(fig)


def get_mean_item_embed(iid_map_df: pd.DataFrame, item_embed: np.ndarray, item_id: str):
    # must consistent with read_data_change_uid
    if f'{item_id}_raw' not in iid_map_df.columns:
        # we didn't rename item_ids, each item has only 1 embedding, so we dont need to average anymore
        return iid_map_df, item_embed

    iid_map_df = iid_map_df.sort_values(by=f'{item_id}_idx').reset_index(drop=True)
    iid_map_df['embed'] = item_embed
    res = iid_map_df.groupby(f'{item_id}_raw')['embed'].apply(lambda x: np.mean(list(x), axis=0))
    final_item_embed = np.vstack(res.values)
    final_iid_map_df = pd.DataFrame({item_id: res.index, f'{item_id}_idx': range(len(res))})
    return final_iid_map_df, final_item_embed


# noinspection SpellCheckingInspection
def save_everything(model, args, metrics, dim_dict, train_data, item_embed, save_dir):

    epoch = len([f for f in os.listdir(save_dir) if f.startswith('model_ep_')])
    torch.save(model.state_dict(), f'{save_dir}/model_ep_{epoch}.pth')

    save_dir = f'{save_dir}/metadata'
    mkdir_if_missing(save_dir, _type='dir')

    with open(f'{save_dir}/model_structure.txt', 'w') as f:
        f.write(str(model.eval()))

    with open(f'{save_dir}/schemas.pkl', 'wb') as f:
        pickle.dump({'label_graph_schema': train_data.graph.canonical_etypes,
                     'model_schema': train_data.model_edge_types,
                     'num_edge_features_dict': train_data.num_edge_features_dict if args.use_edge_feature else None,
                     'user_id': train_data.user_id,
                     'item_id': train_data.item_id
                     }, f)

    if isinstance(item_embed, torch.Tensor):
        item_embed = item_embed.detach().cpu().numpy()
    iid_map_df, item_embed = get_mean_item_embed(train_data.iid_map_df, item_embed, train_data.item_id)
    iid_map_df.to_csv(f'{save_dir}/train_iid_map_df.csv', index=False)
    np.save(f'{save_dir}/item_embeddings.npy', item_embed)
    save_item_embed_df(item_emb=item_embed,
                       node2iid=train_data.node2item,
                       item_id=train_data.item_id,
                       save_path=f'{save_dir}/item_embedding.parquet'
                       )

    args = vars(args)
    args['dim_dict'] = dim_dict
    with open(f'{save_dir}/arguments.json', 'w') as f:
        json.dump(args, f)

    save_plots(metrics, save_dir=save_dir)

    print(f"Saved epoch {epoch}!")


def save_item_embed_df(item_emb, node2iid, item_id, save_path):
    embed_df = pd.DataFrame()
    embed_df[item_id] = [node2iid[nid] for nid in range(item_emb.shape[0])]
    embed_df[item_id] = embed_df[item_id].astype(np.int32)
    embed_df['embeddings'] = list(np.asarray(item_emb).astype(np.float32))
    embed_df.to_parquet(save_path)


def save_inference_result(user_emb_df: pd.DataFrame, score_df: pd.DataFrame, save_dir: str, filename: str):
    if not filename.endswith('parquet'):
        filename = filename + '.parquet'

    embed_path = f'{save_dir}/user_embeddings/{filename}'
    mkdir_if_missing(embed_path)
    user_emb_df.to_parquet(embed_path)
    score_path = f'{save_dir}/scores/{filename}'
    mkdir_if_missing(score_path)
    score_df.to_parquet(score_path)
