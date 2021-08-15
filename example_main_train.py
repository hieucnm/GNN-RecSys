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

from logging_config import get_logger

log = get_logger(__name__)

cuda = torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')
num_workers = 4 if cuda else 0


class TrainDataPaths:
    def __init__(self):
        self.result_filepath = 'result_log.txt'
        # self.sport_feat_path = 'FEATURE DATASET, SPORTS (sport names) .csv'
        self.full_interaction_path = 'examples/user_item_clicks.csv'
        # self.item_sport_path = 'INTERACTION LIST, ITEM-SPORT .csv'
        # self.user_sport_path = 'INTERACTION LIST, USER-SPORT .csv'
        # self.sport_sportg_path = 'INTERACTION LIST, SPORT-SPORT .csv'
        # self.item_feat_path = 'FEATURE DATASET, ITEMS .csv'
        self.user_feat_path = 'examples/user_features.csv'
        # self.sport_onehot_path = 'FEATURE DATASET, SPORTS (one-hot vectors) .csv'


def train_full_model(fixed_params_path,
                     visualization,
                     check_embedding,
                     remove,
                     edge_batch_size,
                     **params,):

    # Create full train set
    train_data_paths = TrainDataPaths()
    full_interaction_data = read_data(train_data_paths.full_interaction_path)
    train_df, test_df = presplit_data(full_interaction_data,
                                      item_feature_data=None,
                                      num_min=1,
                                      remove_unk=True,
                                      sort=True,
                                      test_size_days=1,
                                      item_id_column='item_id',
                                      user_id_column='user_id',
                                      date_column="date")

    train_data_paths.train_path = train_df
    train_data_paths.test_path = test_df
    data = DataLoader(train_data_paths, None)


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
    params = {}
    train_full_model(fixed_params_path=fixed_params_path,
                     visualization=visualization,
                     check_embedding=check_embedding,
                     remove=remove,
                     edge_batch_size=edge_batch_size,
                     **params)


if __name__ == '__main__':
    main()
