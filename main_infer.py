import argparse
import datetime as dt
import json
import os
import sys
import warnings

import numpy as np
import torch

from src.dataloaders import UserNodeLoaderPlus
from src.datasets import InferenceDataSet
from src.evaluation import Predictor
from src.logger import Logger
from src.models import ConvModel
from src.utils_data import read_data, dir_with_timestamp, save_inference_result

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')  # to debug


def main():

    result_dir = dir_with_timestamp(data_dir=args.data_dir, root_dir=args.result_dir)
    sys.stdout = Logger(f'{result_dir}/running_log.txt')
    print(f'Everything will be saved to {result_dir}')

    params = json.load(open(args.param_path))
    print(f'Using device: {device.type}')
    print(f'Infer arguments: {args}')
    print(f'Saved arguments: {params}')

    model = None
    predictor = None
    train_iid_map_df = read_data(args.iid_map_path)
    item_emb = torch.from_numpy(np.load(args.item_embed_path))
    data_dirs = sorted([f'{args.data_dir.rstrip("/")}/{f}' for f in os.listdir(args.data_dir)])

    start_time = dt.datetime.now().replace(microsecond=0)
    print('Start predicting:')
    for i, data_dir in enumerate(data_dirs):
        print("--> Part {:2d}/{:2d}: Loading user feature ...".format(i, len(data_dirs)))
        user_feature = read_data(data_dir + '/user_features.parquet')
        df_group = read_data(data_dir + '/group_chat.parquet')
        df_ad = read_data(data_dir + '/ad.parquet')

        data = InferenceDataSet(train_iid_map_df=train_iid_map_df,
                                user_feature=user_feature,
                                df_group=df_group,
                                df_ad=df_ad
                                )
        data.load_data()
        data.init_graph()
        node_loader = UserNodeLoaderPlus(graph=data.graph,
                                         to_infer_user_nid=data.to_infer_user_node_id,
                                         user_id=data.user_id,
                                         n_neighbors=0,
                                         n_layers=params['n_layers'],
                                         node_batch_size=args.node_batch_size,
                                         num_workers=args.num_workers
                                         )

        if model is None:
            print("Loading model only for the first time ...")
            model = ConvModel(graph=data.graph,
                              dim_dict=params['dim_dict'],
                              label_edge_types=[],
                              n_layers=params['n_layers'],
                              pred=params['pred'],
                              norm=True,
                              dropout=params['dropout'],
                              aggregator_homo=params['aggregator_homo'],
                              aggregator_hetero=params['aggregator_hetero'],
                              user_id=data.user_id,
                              item_id=data.item_id
                              )
            model.load_state_dict(torch.load(args.model_path))
            if device.type != 'cpu':
                model = model.to(device)

            predictor = Predictor(model=model,
                                  item_emb=item_emb,
                                  iid_columns=data.iid_columns,
                                  user_id=data.user_id,
                                  item_id=data.item_id,
                                  print_every=args.print_every
                                  )

        print("--> Part {:2d}/{:2d}: Predicting ...".format(i, len(data_dirs)))
        user_emb_df, score_df = predictor.predict(node_loader=node_loader, node2uid=data.node2user)

        print("--> Part {:2d}/{:2d}: Saving ...".format(i, len(data_dirs)))
        save_inference_result(user_emb_df, score_df, result_dir, os.path.basename(data_dir))

    print(f'Finish predicting! Elapsed time: {dt.datetime.now().replace(microsecond=0) - start_time}')


parser = argparse.ArgumentParser("Graph Learning")
parser.add_argument('--param-path',  type=str, help='Path of all saved arguments after training model')
parser.add_argument('--iid-map-path',  type=str, help='Path of the training item_id mapper dataframe')
parser.add_argument('--model-path',  type=str, help='Path of the trained model')
parser.add_argument('--item-embed-path',  type=str, help='Path of the pre-calculated item embeddings')
parser.add_argument('--data-dir',  type=str, help='Directory containing inference data')
parser.add_argument('--result-dir', type=str, default='examples/results', help='Directory to save everything')
parser.add_argument('--print-every', type=int, default=10, help='Print loss every these iterations')
parser.add_argument('--node-batch-size', type=int, default=1024 * 16, help='Number of nodes in a batch')
parser.add_argument('--num-workers', type=int, default=8, help='Number of cores of CPU to use')


if __name__ == '__main__':
    args = parser.parse_args()
    main()
