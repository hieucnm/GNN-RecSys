import argparse
import datetime as dt
import os
import sys
import warnings

import numpy as np
import torch

from src.dataloaders import UserNodeLoaderPlus
from src.datasets import PredictDataSet
from src.evaluation import Predictor
from src.logger import Logger
from src.models import ConvModel
from src.utils_data import read_data, dir_with_timestamp, save_inference_result, get_date

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')  # to debug


def main():

    if args.has_label:
        result_dir = dir_with_timestamp(root_dir=args.result_dir)
        data_dirs = [path.rstrip('/') for path in args.data_dir.split(',')]
    else:
        result_dir = dir_with_timestamp(root_dir=args.result_dir, data_dir=args.data_dir)
        data_dirs = sorted([f'{args.data_dir.rstrip("/")}/{f}' for f in os.listdir(args.data_dir)])

    sys.stdout = Logger(f'{result_dir}/running_log.txt')
    print(f'Everything will be saved to {result_dir}')
    print(f'Using device: {device.type}')
    print(f'Infer arguments: {args}')

    print("Loading metadata  ...")
    train_output_dir = args.train_output_dir.rstrip('/')
    params = read_data(f'{train_output_dir}/metadata/arguments.json')
    schemas = read_data(f'{train_output_dir}/metadata/schemas.json')
    train_iid_map_df = read_data(f'{train_output_dir}/metadata/train_iid_map_df.csv')
    item_emb = torch.from_numpy(np.load(f'{train_output_dir}/metadata/item_embeddings.npy'))
    print("--> Using trained item embeddings, shape =", item_emb.shape)

    print("Loading model  ...")
    model = ConvModel(edge_types=schemas['model_schema'],
                      dim_dict=params['dim_dict'],
                      n_layers=params['n_layers'],
                      pred=params['pred'],
                      norm=True,
                      dropout=params['dropout'],
                      aggregator_homo=params['aggregator_homo'],
                      aggregator_hetero=params['aggregator_hetero'],
                      user_id=schemas['user_id'],
                      item_id=schemas['item_id'],
                      pre_aggregate=params['pre_aggregate'],
                      use_edge_feat=params['use_edge_feature'],
                      edge_agg_type=params['aggregator_edge'],
                      edge_feats_dict=params['num_edge_features_dict'],
                      )
    model.load_state_dict(torch.load(f'{train_output_dir}/{args.model_file}'))
    if device.type != 'cpu':
        model = model.to(device)

    predictor = Predictor(model=model,
                          item_emb=item_emb,
                          iid_columns=train_iid_map_df[schemas['item_id']].tolist(),
                          user_id=schemas['user_id'],
                          item_id=schemas['item_id'],
                          print_every=args.print_every
                          )

    start_time = dt.datetime.now().replace(microsecond=0)
    print('Start predicting:')
    for i, data_dir in enumerate(data_dirs):
        print("--> Part {:2d}/{:2d}: Loading data ...".format(i, len(data_dirs)))

        to_infer_uid_df = read_data(data_dir + '/label.parquet') if args.has_label else None
        data = PredictDataSet(train_iid_map_df=train_iid_map_df,
                              user_feature=read_data(data_dir + '/user_features.parquet'),
                              df_group=read_data(data_dir + '/group_chat.parquet'),
                              df_ad=read_data(data_dir + '/ad.parquet'),
                              to_infer_uid_df=to_infer_uid_df
                              )
        data.load_data()
        data.init_graph()
        node_loader = UserNodeLoaderPlus(graph=data.graph,
                                         to_infer_user_nid=data.to_infer_user_node_id,
                                         user_id=data.user_id,
                                         n_neighbors=args.n_neighbors,
                                         n_layers=params['n_layers'],
                                         node_batch_size=args.node_batch_size,
                                         num_workers=args.num_workers
                                         )

        print("--> Part {:2d}/{:2d}: Predicting {} users ...".format(i, len(data_dirs), len(data.to_infer_uid_df)))
        user_emb_df, score_df = predictor.predict(node_loader=node_loader, node2uid=data.node2user)

        print("--> Part {:2d}/{:2d}: Saving ...".format(i, len(data_dirs)))
        if args.has_label:
            save_filename = get_date(data_dir) + '/part_0.parquet'
        else:
            save_filename = os.path.basename(data_dir)
        save_inference_result(user_emb_df, score_df, result_dir, save_filename)

    print(f'Finish predicting! Elapsed time: {dt.datetime.now().replace(microsecond=0) - start_time}')


parser = argparse.ArgumentParser("Graph Learning")
parser.add_argument('--train-output-dir', type=str, help='Directory containing everything after training')
parser.add_argument('--model-file', type=str, help='Filename of the trained model, exist in the `train_output_dir`')
parser.add_argument('--data-dir', type=str, help='Directory containing inference data')
parser.add_argument('--result-dir', type=str, default='examples/results', help='Directory to save everything')
parser.add_argument('--has-label', action='store_true', default=False,
                    help='If yes, the file containing user_ids to predict must exists. If no, predict population')

parser.add_argument('--print-every', type=int, default=10, help='Print loss every these iterations')
parser.add_argument('--node-batch-size', type=int, default=1024 * 2, help='Number of nodes in a batch')
parser.add_argument('--num-workers', type=int, default=8, help='Number of cores of CPU to use')
parser.add_argument('--n-neighbors', type=int, default=512,
                    help='Number of random neighbors to aggregate. '
                         'Set 0 to use all neighbors, but not recommended because the memory will explode. '
                         'For now we use the same number for all layers and all edge types. '
                         'Later, we will set different numbers by passing a list[dict[e_type, int]].')


if __name__ == '__main__':
    args = parser.parse_args()
    main()
