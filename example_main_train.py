import os
import numpy as np
import torch
from dgl import heterograph
import datetime as dt
import argparse

from src.utils_data import DataLoader, assign_graph_features, print_data_loaders, calculate_num_batches
from src.utils import read_data, save_txt, save_everything
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
    def __init__(self, args):
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_interaction_path = args.interaction_path
        self.user_feat_path = args.user_feature_path
        self.result_dir = f'{args.result_dir}/{timestamp}'
        os.makedirs(self.result_dir)
        self.log_filepath = f'{self.result_dir}/running_log.txt'


def main(args):

    fixed_params = FixedParameters(
        num_epochs=args.num_epochs,
        start_epoch=args.start_epoch,
        patience=args.patience,
        edge_batch_size=args.edge_batch_size,
        duplicates=args.duplicates,
    )

    # First of all, load data in form of dataframes, split into train dataframe and test dataframe
    train_data_paths = TrainDataPaths(args)
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
                                        data
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
        fixed_params.clicks_sample,
        fixed_params.converts_sample,
    )

    # If starting with a new graph, you should check data after splitting before continue
    # Get into the function below to print out anything you wanna check
    print_data_loaders(train_eids_dict, valid_eids_dict, subtrain_uids, valid_uids, test_uids,
                       all_iids, ground_truth_subtrain, ground_truth_valid, all_eids_dict)

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
                             num_workers=args.num_workers,
                             n_layers=args.n_layers,
                             neg_sample_size=args.neg_sample_size,
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
                'item': data.item_id_df.shape[0],
                'out': args.out_dim,
                'hidden': args.hidden_dim}

    model = ConvModel(valid_graph,
                      args.n_layers,
                      dim_dict,
                      fixed_params.norm,
                      args.dropout,
                      fixed_params.aggregator_type,
                      fixed_params.pred,
                      fixed_params.aggregator_hetero
                      )
    if cuda:
        model = model.to(device)

    print(model.eval())

    # Train
    hp_sentence = vars(args)
    hp_sentence.update(vars(fixed_params))
    hp_sentence = f'{str(hp_sentence)[1: -1]} \n'
    save_txt(f'\n \n START - Hyper-parameters \n{hp_sentence}', train_data_paths.log_filepath, "a")
    model, viz, best_metrics = train_model(
        model,
        num_epochs=fixed_params.num_epochs,
        num_batches_train=num_batches_train,
        num_batches_val_loss=num_batches_val_loss,
        edgeloader_train=edgeloader_train,
        edgeloader_valid=edgeloader_valid,
        loss_fn=max_margin_loss,
        delta=args.delta,
        neg_sample_size=args.neg_sample_size,
        use_recency=fixed_params.use_recency,
        device=device,
        optimizer=fixed_params.optimizer,
        lr=args.lr,
        get_metrics=True,
        train_graph=train_graph,
        valid_graph=valid_graph,
        nodeloader_valid=nodeloader_valid,
        nodeloader_subtrain=nodeloader_subtrain,
        k=fixed_params.k,
        out_dim=args.out_dim,
        num_batches_val_metrics=num_batches_val_metrics,
        num_batches_subtrain=num_batches_subtrain,
        bought_eids=train_eids_dict[('user', 'converts', 'item')],
        ground_truth_subtrain=ground_truth_subtrain,
        ground_truth_valid=ground_truth_valid,
        remove_already_bought=True,
        result_filepath=train_data_paths.log_filepath,
        start_epoch=fixed_params.start_epoch,
        patience=fixed_params.patience,
        pred=args.pred,
        remove_false_negative=fixed_params.remove_false_negative
    )

    # Save everything
    save_everything(model, valid_graph, data, args, fixed_params, save_dir=train_data_paths.result_dir)
    plot_train_loss(hp_sentence, viz, save_dir=train_data_paths.result_dir)

    # Report performance on validation set
    sentence = ("BEST VALIDATION Precision "
                "{:.3f}% | Recall {:.3f}% | Coverage {:.2f}%"
                .format(best_metrics['precision'] * 100,
                        best_metrics['recall'] * 100,
                        best_metrics['coverage'] * 100))
    log.info(sentence)
    save_txt(sentence, train_data_paths.log_filepath, mode='a')

    # Report performance on test set
    log.debug('Test metrics start ...')
    model.eval()
    with torch.no_grad():
        embeddings = get_embeddings(valid_graph, args.out_dim, model, nodeloader_test, num_batches_test, device)

        for ground_truth in [data.ground_truth_convert_test, data.ground_truth_test]:
            precision, recall, coverage = get_metrics_at_k(embeddings,
                                                           valid_graph,
                                                           model,
                                                           args.out_dim,
                                                           ground_truth,
                                                           all_eids_dict[('user', 'converts', 'item')],
                                                           fixed_params.k,
                                                           False,  # Remove already bought
                                                           device,
                                                           args.pred)
            sentence = ("TEST Precision {:.3f}% | Recall {:.3f}% | Coverage {:.2f}%"
                        .format(precision * 100,
                                recall * 100,
                                coverage * 100))
            log.info(sentence)
            save_txt(sentence, train_data_paths.log_filepath, mode='a')


parser = argparse.ArgumentParser("Graph Learning")
parser.add_argument('-ip', '--interaction-path', type=str,
                    default='/home/ubuntu/workspace/GNN-RecSys/examples/user_item_clicks.parquet',
                    help='Path to load the historical interactions of user-item to build the graph.')
parser.add_argument('-up', '--user-feature-path', type=str,
                    default='/home/ubuntu/workspace/GNN-RecSys/examples/user_features.csv',
                    help='Path to load the features to assign to users in the graph.')
parser.add_argument('-rp', '--result-dir', type=str,
                    default='/home/ubuntu/workspace/GNN-RecSys/examples/results',
                    help='Directory to save everything.')
parser.add_argument('--out-dim', type=int, default=16, help='Output dimension')
parser.add_argument('--hidden-dim', type=int, default=16, help='Hidden dimension')
parser.add_argument('--n-layers', type=int, default=3, help='Number of layers')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout ratio')
parser.add_argument('--pred', type=str, default='cos', choices=['nn', 'cos'], help='Way to predict scores of link')
parser.add_argument('--delta', type=float, default=0.05, help='Margin used in maximal margin loss')

parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay in SGD')
parser.add_argument('--num-epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='Starting from this epoch')
parser.add_argument('--patience', type=int, default=1, help='Number of iteration to wait for early stopping')
parser.add_argument('--neg-sample-size', type=int, default=5, help='Number of samples when doing negative sampling')
parser.add_argument('--edge-batch-size', default=16, help='Number of edges in a train / validation batch')
parser.add_argument('--num-workers', type=int, default=4, help='Number of cores of CPU to use')
parser.add_argument('--check-embedding', action='store_true', default=False, help='Explore embedding result')
parser.add_argument('--duplicates', type=str, default='count_occurrence',
                    choices=['count_occurrence', 'keep_all', 'keep_last'],
                    help='Way to handle duplicate interactions')


if __name__ == '__main__':
    print(f"Using device: {device.type}")
    args = parser.parse_args()
    print(args)
    main(args)
