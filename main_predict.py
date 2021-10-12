import argparse
import datetime as dt
import os.path
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm._tqdm import tqdm
from dgl import load_graphs
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from src.models import ConvModel
from src.utils_data import read_data, mkdir_if_missing

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = '/data/zmining/jupyter-notebook/hieucnm/graph/v5__with_edge_features'


parser = argparse.ArgumentParser("Graph Learning")
parser.add_argument('--data-dir', type=str, help='Directory containing the graph.bin & uid_map_df.parquet',
                    default=PROJECT_DIR + '/data/inference_data/preprocessed/%Y/%m/%d')
parser.add_argument('--result-dir', type=str, default=PROJECT_DIR + '/outputs/predict/%Y/%m/%d')
parser.add_argument('--train-output-dir', default=PROJECT_DIR + '/outputs/train/20211006_221245')
parser.add_argument('--model-file', help='Model file to load, exists in `train_output_dir`', default='model_ep_7.pth')
parser.add_argument('--node-batch-size', type=int, default=2**16, help='Number of nodes in a batch')
parser.add_argument('--duration', type=int, help='', default=7)
parser.add_argument('--weekday', type=int, help='', default=5)


def predict():
    mkdir_if_missing(result_dir, _type='dir')
    print(f'Everything will be saved to {result_dir}')
    print(f'Using device: {device.type}')
    print(f'All arguments: {args}')

    print('Loading metadata ...')
    params = read_data(f'{args.train_output_dir}/metadata/arguments.json')
    schemas = read_data(f'{args.train_output_dir}/metadata/schemas.pkl')
    iid_map_df = read_data(f'{args.train_output_dir}/metadata/train_iid_map_df.csv')
    item_emb = torch.from_numpy(np.load(f'{args.train_output_dir}/metadata/item_embeddings.npy')).to(device)
    item_emb_normed = item_emb / item_emb.norm(dim=1, keepdim=True)
    iid_columns = [str(c) for c in iid_map_df[schemas['item_id']].tolist()]
    user_id = schemas['user_id']
    item_id = schemas['item_id']

    print("Loading graph & uid_map_df ...")
    uid_map_df = read_data(f"{data_dir}/uid_map_df.parquet")
    graph, _ = load_graphs(f"{data_dir}/graph.bin")
    graph = graph[0].to(device)
    print('Summary graph:', graph)

    to_infer_user_node_id = np.arange(uid_map_df.shape[0], dtype=np.int32)
    sampler = MultiLayerNeighborSampler([params['n_neighbors']] * (params['n_layers'] - 1), replace=False)
    node_loader = NodeDataLoader(g=graph,
                                 nids={user_id: to_infer_user_node_id},
                                 block_sampler=sampler,
                                 batch_size=args.node_batch_size
                                 )
    print('Loading model ...')
    model = ConvModel(edge_types=schemas['model_schema'],
                      dim_dict=params['dim_dict'],
                      n_layers=params['n_layers'],
                      pred=params['pred'],
                      norm=True,
                      dropout=params['dropout'],
                      aggregator_homo=params['aggregator_homo'],
                      aggregator_hetero=params['aggregator_hetero'],
                      user_id=user_id,
                      item_id=item_id,
                      pre_aggregate=params['pre_aggregate'],
                      use_edge_feat=params['use_edge_feature'],
                      edge_agg_type=params['aggregator_edge'],
                      edge_feats_dict=schemas['num_edge_features_dict'],
                      )
    model.load_state_dict(torch.load(f'{args.train_output_dir}/{args.model_file}', map_location=device))
    model = model.to(device)
    model.eval()

    mkdir_if_missing(f'{result_dir}/user_embeddings', _type='dir')
    mkdir_if_missing(f'{result_dir}/scores', _type='dir')
    start_time = dt.datetime.now().replace(microsecond=0)

    print('Start predicting ...')
    for i, (_, output_nodes, blocks) in tqdm(enumerate(node_loader), total=len(node_loader)):
        user_ids = uid_map_df.iloc[output_nodes[user_id].cpu(), :][user_id]

        # forward
        blocks = [b.to(device) for b in blocks]
        input_features = blocks[0].srcdata['features']
        with torch.no_grad():
            embed_dict = model.get_repr(blocks, input_features)

        # embeddings df
        user_emb = embed_dict[user_id]
        user_emb_df = pd.DataFrame({user_id: user_ids, 'embeddings': list(user_emb.cpu().numpy())})
        user_emb_df.to_parquet(f'{result_dir}/user_embeddings/part_{i}.parquet')

        # scores (cosine similarities)
        user_emb_normed = user_emb / user_emb.norm(dim=1, keepdim=True)
        scores = torch.mm(user_emb_normed, item_emb_normed.transpose(0, 1))
        score_df = pd.DataFrame(data=scores.cpu().numpy(), columns=iid_columns)
        score_df[user_id] = user_ids
        score_df[[user_id] + iid_columns].to_parquet(f'{result_dir}/scores/part_{i}.parquet')

    print(f'Finish training! Elapsed time: {dt.datetime.now().replace(microsecond=0) - start_time}')


def main():
    user_embed_path = f'{result_dir}/user_embeddings'
    if os.path.exists(user_embed_path) and len(os.listdir(user_embed_path)) > 0:
        print("Outputs exist! Finish!")
        return

    if not os.path.exists(data_dir):
        print('Rsync data from 9.17 ... ')  # TODO: Get data from 9.17
    predict()


if __name__ == '__main__':
    args = parser.parse_args()
    date = pd.date_range("2021-09-11", periods=1)[0]  # TODO: change to dt.datetime.today()
    date = [d for d in pd.date_range(end=date, periods=args.duration) if d.weekday() == args.weekday][0]
    print('On date:', date)

    result_dir = date.strftime(args.result_dir)
    data_dir = date.strftime(args.data_dir)
    main()
