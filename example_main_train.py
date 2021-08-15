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
        self.full_interaction_path = 'examples/user_item_clicks.csv'
        self.user_feat_path = 'examples/user_features.csv'


def train_full_model(fixed_params_path,
                     visualization,
                     check_embedding,
                     remove,
                     edge_batch_size,
                     **params,):

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    # fixed_params = objectview(read_data(fixed_params_path))
    fixed_params = objectview({})
    fixed_params.uid_column = 'user_id'
    fixed_params.iid_column = 'item_id'
    fixed_params.date_column = 'date'
    fixed_params.conv_column = 'converted'
    fixed_params.duplicates = 'count_occurrence'
    fixed_params.discern_clicks = True

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
    print("valid_graph:")
    print(valid_graph)


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
        'use_recency': True
    }
    train_full_model(fixed_params_path=fixed_params_path,
                     visualization=visualization,
                     check_embedding=check_embedding,
                     remove=remove,
                     edge_batch_size=edge_batch_size,
                     **params)


if __name__ == '__main__':
    main()
