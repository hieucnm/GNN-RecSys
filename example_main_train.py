import math
import datetime

import click
import numpy as np
import torch
from dgl.data.utils import save_graphs
from dgl import heterograph

from src.utils_data import DataLoader, assign_graph_features, print_data_loaders, calculate_num_batches
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


class TrainDataPaths:
    def __init__(self):
        self.result_filepath = 'examples/result_log.txt'
        self.full_interaction_path = 'examples/user_item_clicks.csv'
        self.user_feat_path = 'examples/user_features.csv'


def train_full_model(fixed_params_path,
                     visualization,
                     check_embedding,
                     edge_batch_size,
                     **params,):

    fixed_params = FixedParameters(
        num_epochs=params['num_epochs'],
        start_epoch=params['start_epoch'],
        patience=params['patience'],
        edge_batch_size=edge_batch_size,
        duplicates='count_occurrence'
    )

    # First of all, load data in form of dataframes, split into train dataframe and test dataframe
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

    # Create whole graph (train, valid, test)
    valid_graph = heterograph(data.graph_schema)

    valid_graph = assign_graph_features(valid_graph,
                                        fixed_params,
                                        data,
                                        **params,
                                        )

    # Split user_ids in to train, valid and test set
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

    # If starting with a new graph, you should check data after splitting before continue
    # Get into the function below to print out anything you wanna check
    # print_data_loaders(train_eids_dict, valid_eids_dict, subtrain_uids, valid_uids, test_uids,
    #                    all_iids, ground_truth_subtrain, ground_truth_valid, all_eids_dict)

    # Initialize edges and nodes loaders for the above data sets
    (
        edgeloader_train,
        edgeloader_valid,
        nodeloader_subtrain,
        nodeloader_valid,
        nodeloader_test
    ) = generate_dataloaders(valid_graph,
                             train_graph,
                             train_eids_dict,
                             valid_eids_dict,
                             subtrain_uids,
                             valid_uids,
                             test_uids,
                             all_iids,
                             fixed_params,
                             num_workers=params['num_workers'],
                             n_layers=params['n_layers'],
                             neg_sample_size=params['neg_sample_size'],
                             )

    # Calculate approximate number of nodes in a batch, based on number of edges in a batch
    (
        num_batches_train,
        num_batches_subtrain,
        num_batches_test,
        num_batches_val_loss,
        num_batches_val_metrics
    ) = calculate_num_batches(train_eids_dict,
                              valid_eids_dict,
                              subtrain_uids,
                              valid_uids,
                              test_uids,
                              all_iids,
                              fixed_params)

    # Init model
    dim_dict = {'user': valid_graph.nodes['user'].data['features'].shape[1],
                # 'item': params['hidden_dim'],
                'n_item': data.item_id_df.shape[0],
                'out': params['out_dim'],
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

    print(model.eval())

    # Train
    hp_sentence = params
    hp_sentence.update(vars(fixed_params))
    hp_sentence = f'{str(hp_sentence)[1: -1]} \n'
    save_txt(f'\n \n START - Hyperparameters \n{hp_sentence}', train_data_paths.result_filepath, "a")
    trained_model, viz, best_metrics = train_model(
        model,
        num_epochs=fixed_params.num_epochs,
        num_batches_train=num_batches_train,
        num_batches_val_loss=num_batches_val_loss,
        edgeloader_train=edgeloader_train,
        edgeloader_valid=edgeloader_valid,
        loss_fn=max_margin_loss,
        delta=params['delta'],
        neg_sample_size=params['neg_sample_size'],
        use_recency=params['use_recency'],
        cuda=cuda,
        device=device,
        optimizer=fixed_params.optimizer,
        lr=params['lr'],
        get_metrics=True,
        train_graph=train_graph,
        valid_graph=valid_graph,
        nodeloader_valid=nodeloader_valid,
        nodeloader_subtrain=nodeloader_subtrain,
        k=fixed_params.k,
        out_dim=params['out_dim'],
        num_batches_val_metrics=num_batches_val_metrics,
        num_batches_subtrain=num_batches_subtrain,
        bought_eids=train_eids_dict[('user', 'converts', 'item')],
        ground_truth_subtrain=ground_truth_subtrain,
        ground_truth_valid=ground_truth_valid,
        remove_already_bought=True,
        result_filepath=train_data_paths.result_filepath,
        start_epoch=fixed_params.start_epoch,
        patience=fixed_params.patience,
        pred=params['pred'],
        remove_false_negative=fixed_params.remove_false_negative
    )


@click.command()
@click.option('--fixed_params_path', default='fixed_params.pkl',
              help='Path where the fixed parameters used in the hyperparametrization were saved.')
@click.option('--params_path', default='params.pkl',
              help='Path where the optimal hyperparameters found in the hyperparametrization were saved.')
@click.option('-viz', '--visualization', count=True, help='Visualize result')
@click.option('--check_embedding', count=True, help='Explore embedding result')
@click.option('--edge_batch_size', default=16, help='Number of edges in a train / validation batch')
def main(fixed_params_path, params_path, visualization, check_embedding, edge_batch_size):
    # params = read_data(params_path)
    # params.pop('remove', None)
    # params.pop('edge_batch_size', None)
    params = {
        'use_recency': True,
        'hidden_dim': 16,
        'out_dim': 16,
        'n_layers': 3,
        'dropout': 0.1,
        'norm': True,
        'num_epochs': 3,
        'start_epoch': 0,
        'patience': 1,
        'clicks_sample': 1.,
        'converts_sample': 1.,
        'neg_sample_size': 5,
        'num_workers': 4 if cuda else 0,
        'delta': 0.05,
        'lr': 0.001,

    }

    train_full_model(fixed_params_path=fixed_params_path,
                     visualization=visualization,
                     check_embedding=check_embedding,
                     edge_batch_size=edge_batch_size,
                     **params)


if __name__ == '__main__':
    main()
