import math
import datetime

import click
import numpy as np
import torch
from dgl.data.utils import save_graphs

from src.builder import create_graph
from src.utils_data import DataLoader, assign_graph_features
from src.utils import read_data, save_txt, save_outputs
from src.model import ConvModel, max_margin_loss
from src.sampling import train_valid_split, generate_dataloaders
from src.train.run import train_model, get_embeddings
from src.utils_vizualization import plot_train_loss
from src.metrics import (create_already_bought, create_ground_truth,
                         get_metrics_at_k, get_recs)
from src.evaluation import explore_recs, explore_sports, check_coverage
from presplit import presplit_data
from src.utils_data import FixedParameters

from logging_config import get_logger

log = get_logger(__name__)

cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')
num_workers = 4 if cuda else 0


class TrainDataPaths:
    def __init__(self):
        self.result_filepath = 'result_log.txt'
        self.full_interaction_path = 'examples/user_item_clicks.csv'
        self.user_feat_path = 'examples/user_features.csv'


def train_full_model(fixed_params_path,
                     visualization,
                     check_embedding,
                     remove,
                     edge_batch_size,
                     **params,):

    fixed_params = FixedParameters(
        num_epochs=params['num_epochs'],
        start_epoch=params['start_epoch'],
        patience=params['patience'],
        remove=remove,
        edge_batch_size=edge_batch_size,
        duplicates='count_occurrence'
    )

    # Create full train set
    train_data_paths = TrainDataPaths()
    full_interaction_data = read_data(train_data_paths.full_interaction_path)
    train_df, test_df = presplit_data(full_interaction_data,
                                      uid_column=fixed_params.uid_column,
                                      date_column=fixed_params.date_column,
                                      num_min=3,
                                      test_size_days=1,
                                      sort=True
                                      )

    train_data_paths.train_path = train_df
    train_data_paths.test_path = test_df
    data = DataLoader(train_data_paths, fixed_params)

    valid_graph = create_graph(
        data.graph_schema,
    )

    valid_graph = assign_graph_features(valid_graph,
                                        fixed_params,
                                        data,
                                        **params,
                                        )

    dim_dict = {'user': valid_graph.nodes['user'].data['features'].shape[1],
                'item': params['hidden_dim'],
                'out':  fixed_params.out_dim,
                'hidden': params['hidden_dim']}

    model = ConvModel(valid_graph,
                      params['n_layers'],
                      dim_dict,
                      params['norm'],
                      params['dropout'],
                      fixed_params.aggregator_type,
                      fixed_params.pred,
                      fixed_params.aggregator_hetero
                      )
    if cuda:
        model = model.to(device)

    # Initialize data_loaders
    # get training and test ids
    (
        train_graph,
        train_eids_dict,
        valid_eids_dict,
        subtrain_uids,
        valid_uids,
        test_uids,
        all_iids,
        ground_truth_subtrain,
        ground_truth_valid,
        all_eids_dict
    ) = train_valid_split(
        valid_graph,
        data.ground_truth_test,
        fixed_params.etype,
        fixed_params.subtrain_size,
        fixed_params.valid_size,
        fixed_params.reverse_etype,
        fixed_params.train_on_clicks,
        fixed_params.remove_train_eids,
        params['clicks_sample'],
        params['converts_sample'],
    )


@click.command()
@click.option('--fixed_params_path', default='fixed_params.pkl',
              help='Path where the fixed parameters used in the hyperparametrization were saved.')
@click.option('--params_path', default='params.pkl',
              help='Path where the optimal hyperparameters found in the hyperparametrization were saved.')
@click.option('-viz', '--visualization', count=True, help='Visualize result')
@click.option('--check_embedding', count=True, help='Explore embedding result')
@click.option('--remove', default=.99, help='Percentage of users to remove from train set. Ideally,'
                                            ' remove would be 0. However, higher "remove" accelerates training.')
@click.option('--edge_batch_size', default=2048, help='Number of edges in a train / validation batch')
def main(fixed_params_path, params_path, visualization, check_embedding, remove, edge_batch_size):
    # params = read_data(params_path)
    # params.pop('remove', None)
    # params.pop('edge_batch_size', None)
    params = {
        'use_recency': True,
        'hidden_dim': 64,
        'n_layers': 3,
        'dropout': 0.1,
        'norm': True,
        'num_epochs': 2,
        'start_epoch': 0,
        'patience': 1,
        'clicks_sample': 1.,
        'converts_sample': 1.,
    }

    train_full_model(fixed_params_path=fixed_params_path,
                     visualization=visualization,
                     check_embedding=check_embedding,
                     remove=remove,
                     edge_batch_size=edge_batch_size,
                     **params)


if __name__ == '__main__':
    main()
