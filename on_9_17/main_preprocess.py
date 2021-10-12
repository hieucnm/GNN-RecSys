import argparse
import datetime as dt
import gc
import sys
import warnings

import torch
from dgl import heterograph, save_graphs

from utils import *

warnings.filterwarnings('ignore')
sys.path.append('/data/zmining/jupyter-hub/hieucnm')
from common.spark_processing import init_spark
from common.io import mkdir_if_missing


PROJECT_DIR = '/data/zmining/jupyter-hub/hieucnm/graph/test_deploy_local'
HDFS_PROJECT_DIR = '/data/jobs/rnd/development/hieucnm/graph/v5__with_edge_features'
DATA_NAMES = ['ad_form_d30', 'group_chat_d07', 'user_profile']


parser = argparse.ArgumentParser("Graph Deployment")
parser.add_argument('--hdfs-dir', type=str, help='', default=HDFS_PROJECT_DIR + '/inference_data/history/{}/%Y/%m/%d')
parser.add_argument('--local-dir', type=str, help='', default=PROJECT_DIR + '/data/history/{}/%Y/%m/%d')
parser.add_argument('--preprocessed-dir', type=str, help='', default=f'{PROJECT_DIR}/data/preprocessed/%Y/%m/%d')
parser.add_argument('--metadata-path', type=str, help='', default=f'{PROJECT_DIR}/data/metadata.pkl')
parser.add_argument('--iid-map-path', type=str, help='', default=f'{PROJECT_DIR}/data/train_iid_map_df.csv')
parser.add_argument('--duration', type=int, help='', default=7)
parser.add_argument('--weekday', type=int, help='', default=5)


def wrapup_get_data():
    # use try/catch to stop spark in case any error occurs
    spark = None
    try:
        spark = init_spark(n_ram=64)
        for data_name in DATA_NAMES:
            spark.read.parquet(date.strftime(args.hdfs_dir).format(data_name)) \
                .write.parquet("file://" + date.strftime(args.local_dir).format(data_name))
            print(f'--> saved {data_name}')
        spark.stop()
    except Exception as e:
        if spark is not None:
            spark.stop()
        print(e)


def wrapup_preprocess_data():
    metadata = pickle.load(open(args.metadata_path, "rb"))
    mkdir_if_missing(date.strftime(args.preprocessed_dir), type='dir')

    print('--> preprocessing user profiles ...')
    user_profile = pd.read_parquet(date.strftime(args.local_dir).format("user_profile"))
    user_profile = preprocess_user_profiles(user_profile, metadata=metadata)
    user_profile.to_parquet(date.strftime(args.preprocessed_dir) + "/user_profile.parquet")

    print('--> preprocessing group-chat data ...')
    df_group = pd.read_parquet(date.strftime(args.local_dir).format("group_chat_d07"))
    df_group = preprocess_group_features(df_group, metadata=metadata)
    df_group.to_parquet(date.strftime(args.preprocessed_dir) + "/group_chat.parquet")

    print('--> preprocessing ad data ...')
    df_ad = pd.read_parquet(date.strftime(args.local_dir).format("ad_form_d30"))
    df_ad = preprocess_ad_features(df_ad, metadata=metadata)
    df_ad.to_parquet(date.strftime(args.preprocessed_dir) + "/ad.parquet")


def wrapup_build_graph():
    iid_map_df = read_data(args.iid_map_path)
    iid_map_df[item_id] = iid_map_df[item_id].astype(np.int32)
    iid_map_df[item_idx] = iid_map_df[item_idx].astype(np.int32)

    print('--> loading preprocessed data ...')
    user_profile = pd.read_parquet(f'{date.strftime(args.preprocessed_dir)}/user_profile.parquet')
    df_group = pd.read_parquet(f'{date.strftime(args.preprocessed_dir)}/group_chat.parquet')
    df_ad = pd.read_parquet(f'{date.strftime(args.preprocessed_dir)}/ad.parquet')

    print('--> indexing ad_cate ...')
    df_ad = df_ad.merge(iid_map_df, on=item_id)

    print('--> indexing src_id ...')
    uid_map_df = user_profile[[user_id]]
    uid_map_df[user_idx] = uid_map_df.index.astype(np.int32)
    df_ad = df_ad.merge(uid_map_df, on=user_id)
    df_group = df_group \
        .merge(uid_map_df, on=user_id) \
        .merge(uid_map_df.rename(columns={user_id: des_uid, user_idx: des_uidx}), on=des_uid)

    print('--> creating graph ...')
    graph_schema = dict()
    for kind in ad_kind:
        e_type, reverse_e_type = message_edges[kind]
        pairs = df_ad[df_ad['kind'] == kind][[user_idx, item_idx]].values
        graph_schema[e_type] = (pairs[:, 0], pairs[:, 1])
        graph_schema[reverse_e_type] = (pairs[:, 1], pairs[:, 0])
    pairs = df_group[[user_idx, des_uidx]].values
    graph_schema[message_edges['group_chat'][0]] = (pairs[:, 0], pairs[:, 1])

    num_nodes_dict = {user_id: uid_map_df.shape[0], item_id: iid_map_df.shape[0]}
    graph = heterograph(graph_schema, num_nodes_dict=num_nodes_dict, idtype=torch.int32)

    print('--> importing features ...')
    graph.nodes[user_id].data['features'] = torch.FloatTensor(user_profile.drop(columns=[user_id]).values)
    graph.nodes[item_id].data['features'] = torch.IntTensor(list(range(iid_map_df.shape[0])))

    # ad
    feat_cols = df_ad.columns.difference(non_feature_columns)
    for kind in ad_kind:
        e_type, reverse_e_type = message_edges[kind]
        ad_features = torch.FloatTensor(df_ad[df_ad['kind'] == kind][feat_cols].values)
        graph.edges[e_type].data['features'] = ad_features
        graph.edges[reverse_e_type].data['features'] = ad_features

    # group
    feat_cols = df_group.columns.difference(non_feature_columns)
    graph.edges[message_edges['group_chat'][0]].data['features'] = torch.FloatTensor(df_group[feat_cols].values)

    print('--> saving graph & uid_map_df ...')
    save_graphs(f'{date.strftime(args.preprocessed_dir)}/graph.bin', [graph])
    uid_map_df.to_parquet(f'{date.strftime(args.preprocessed_dir)}/uid_map_df.parquet')


# ==============
# Main stage ===

def main():
    if os.path.exists(f'{date.strftime(args.preprocessed_dir)}/graph.bin'):
        print('Graph built! Finish!')
        return

    start_time = dt.datetime.now().replace(microsecond=0)
    if not os.path.exists(f'{date.strftime(args.preprocessed_dir)}/user_profile.parquet'):
        if not os.path.exists(date.strftime(args.local_dir).format(DATA_NAMES[0])):
            print('Getting data from hdfs to local ...')
            wrapup_get_data()
            gc.collect()

        print('Preprocessing data on local ...')
        wrapup_preprocess_data()
        gc.collect()

    print('Buiding graph ...')
    wrapup_build_graph()
    gc.collect()

    print(f'Finish! Elapsed time: {dt.datetime.now().replace(microsecond=0) - start_time}')


if __name__ == '__main__':
    args = parser.parse_args()
    date = pd.date_range("2021-09-11", periods=1)[0]  # dt.datetime.today()
    date = [d for d in pd.date_range(end=date, periods=args.duration) if d.weekday() == args.weekday][0]
    print(f'On date: {date}')
    main()

