import argparse
import datetime as dt
import sys
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.optim

from src.dataloaders import NodeLoaderPlus, ItemNodeLoaderPlus, EdgeLoaderPlus
from src.datasets import TrainGroupChatDataSet
from src.evaluation import LabelBasedEvaluator
from src.logger import Logger
from src.losses import MaxMarginLoss, BCELossCustom
from src.models import ConvModel
from src.trainers import Trainer
from src.utils_data import save_everything, seed_everything, dir_with_timestamp

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')  # to debug

ADS_TO_TRACK = [20018, 20255, 20147]


def get_report(auc_dict, item2nid, loss=None):
    report = "Length {:2d} | Mean AUC {:.2f}%".format(len(auc_dict), np.mean(list(auc_dict.values())) * 100)
    if loss is not None:
        report = "Loss {:.5f} | ".format(loss) + report
    for item in ADS_TO_TRACK:
        report += " | {} AUC: {:.2f}%".format(item, auc_dict[item2nid[item]] * 100)
    return report


def main():

    result_dir = dir_with_timestamp(args.result_dir)
    sys.stdout = Logger(f'{result_dir}/running_log.txt')
    print(f'Everything will be saved to {result_dir}')
    print(f'Using device: {device.type}')
    print(f'All arguments: {args}')

    print("Loading training data ...")
    train_data = TrainGroupChatDataSet(data_dirs=args.train_dirs, use_edge_features=args.use_edge_feature)
    train_data.init_graph()
    train_edge_loader = EdgeLoaderPlus(graph=train_data.graph,
                                       train_graph=train_data.train_graph,
                                       pos_label_etypes=train_data.pos_label_edge_types,
                                       neg_label_etypes=train_data.neg_label_edge_types,
                                       sampler_n=args.neg_sampler,
                                       n_neighbors=args.n_neighbors,
                                       n_layers=args.n_layers,
                                       neg_sample_size=args.neg_sample_size,
                                       edge_batch_size=args.edge_batch_size,
                                       num_workers=args.num_workers,
                                       use_ddp=args.use_ddp
                                       )

    train_node_loader = NodeLoaderPlus(graph=train_data.graph,
                                       train_graph=train_data.train_graph,
                                       user_id=train_data.user_id,
                                       item_id=train_data.item_id,
                                       e_types=train_data.label_edge_types,
                                       sample_size=args.sub_train_sample_size,
                                       n_neighbors=args.n_neighbors,
                                       n_layers=args.n_layers,
                                       node_batch_size=args.node_batch_size,
                                       num_workers=args.num_workers
                                       )
    train_ground_truth = train_node_loader.groundtruth_dict

    print("Loading validation data ...")
    valid_data = TrainGroupChatDataSet(data_dirs=args.valid_dirs,
                                       train_iid_map_df=train_data.iid_map_df,
                                       use_edge_features=args.use_edge_feature)
    valid_data.init_graph()
    valid_edge_loader = EdgeLoaderPlus(graph=valid_data.graph,
                                       train_graph=valid_data.train_graph,
                                       pos_label_etypes=valid_data.pos_label_edge_types,
                                       neg_label_etypes=valid_data.neg_label_edge_types,
                                       sampler_n=args.neg_sampler,
                                       n_neighbors=args.n_neighbors,
                                       n_layers=args.n_layers,
                                       neg_sample_size=args.neg_sample_size,
                                       edge_batch_size=args.edge_batch_size,
                                       num_workers=args.num_workers,
                                       use_ddp=args.use_ddp
                                       )

    valid_node_loader = NodeLoaderPlus(graph=valid_data.graph,
                                       train_graph=valid_data.train_graph,
                                       user_id=valid_data.user_id,
                                       item_id=valid_data.item_id,
                                       e_types=valid_data.label_edge_types,
                                       n_neighbors=args.n_neighbors,
                                       n_layers=args.n_layers,
                                       node_batch_size=args.node_batch_size,
                                       num_workers=args.num_workers
                                       )
    valid_ground_truth = valid_node_loader.groundtruth_dict

    item_node_loader = ItemNodeLoaderPlus(adjust_graph=train_data.train_graph,
                                          item_id=train_data.item_id,
                                          n_neighbors=args.n_neighbors,
                                          n_layers=args.n_layers,
                                          node_batch_size=args.node_batch_size,
                                          num_workers=args.num_workers
                                          )
    item2node = train_data.item2node
    node2item = train_data.node2item

    print("Initializing model ...")
    dim_dict = {'user': train_data.num_user_features,
                'item': train_data.num_items,
                'out': args.out_dim,
                'hidden': args.hidden_dim}

    model = ConvModel(edge_types=train_data.model_edge_types,
                      dim_dict=dim_dict,
                      n_layers=args.n_layers,
                      pred=args.pred,
                      norm=True,
                      dropout=args.dropout,
                      aggregator_homo=args.aggregator_homo,
                      aggregator_hetero=args.aggregator_hetero,
                      user_id=train_data.user_id,
                      item_id=train_data.item_id,
                      pre_aggregate=args.pre_aggregate,
                      use_edge_feat=args.use_edge_feature,
                      edge_agg_type=args.aggregator_edge,
                      edge_feats_dict=train_data.num_edge_features_dict,
                      )
    if device.type != 'cpu':
        model = model.to(device)

    criterion = MaxMarginLoss(delta=args.delta) if args.loss == 'hinge' else BCELossCustom()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      device=device,
                      print_every=args.print_every
                      )
    evaluator = LabelBasedEvaluator(model=model,
                                    user_id=train_data.user_id,
                                    item_id=train_data.item_id,
                                    print_every=args.print_every,
                                    pos_etype=train_data.pos_label_edge_types,
                                    neg_etype=train_data.neg_label_edge_types
                                    )

    print('Evaluate before training:')
    print('--> Evaluating sub-train ...')
    train_auc_dict = evaluator.evaluate(
        graph=train_data.graph,
        node_loader=train_node_loader,
        item_node_loader=item_node_loader,
        ground_truth=train_ground_truth
    )
    print('--> Evaluating validation ...')
    val_auc_dict = evaluator.evaluate(
        graph=valid_data.graph,
        node_loader=valid_node_loader,
        item_node_loader=item_node_loader,
        ground_truth=valid_ground_truth
    )

    report = "--> Before training: \nTraining:   {} \nValidation: {}" \
        .format(get_report(train_auc_dict, item2node),
                get_report(val_auc_dict, item2node))
    print(report)

    # TRAIN
    metrics = defaultdict(lambda: defaultdict(list))
    start_time = dt.datetime.now().replace(microsecond=0)
    print('Start training:')
    for epoch in range(1, args.num_epochs + 1):

        print('--> Epoch {}/{}: Training ...'.format(epoch, args.num_epochs))
        train_avg_loss = trainer.train(train_edge_loader)

        print('--> Epoch {}/{}: Evaluating sub-train ...'.format(epoch, args.num_epochs))
        train_auc_dict = evaluator.evaluate(
            graph=train_data.graph,
            node_loader=train_node_loader,
            item_node_loader=item_node_loader,
            ground_truth=train_ground_truth
        )

        print('--> Epoch {}/{}: Calculating validation loss ...'.format(epoch, args.num_epochs))
        val_avg_loss = trainer.calculate_loss(valid_edge_loader)

        print('--> Epoch {}/{}: Evaluating validation ...'.format(epoch, args.num_epochs))
        val_auc_dict = evaluator.evaluate(
            graph=valid_data.graph,
            node_loader=valid_node_loader,
            item_node_loader=item_node_loader,
            ground_truth=valid_ground_truth
        )

        report = "--> Finish epoch {}/{}: \nTraining:   {} \nValidation: {}" \
            .format(epoch, args.num_epochs,
                    get_report(train_auc_dict, item2node, train_avg_loss),
                    get_report(val_auc_dict, item2node, val_avg_loss))
        print(report)
        metrics['Loss']['training'].append(train_avg_loss)
        metrics['Loss']['validation'].append(val_avg_loss)
        for item_nid, auc in train_auc_dict.items():
            metrics[f'AUC {node2item[item_nid]}']['training'].append(auc)
        for item_nid, auc in val_auc_dict.items():
            metrics[f'AUC {node2item[item_nid]}']['validation'].append(auc)

        # Save every epoch, not after all epochs
        item_embed = evaluator.get_all_item_embeddings(train_data.train_graph, item_node_loader)
        save_everything(model=model,
                        args=args,
                        metrics=metrics,
                        dim_dict=dim_dict,
                        train_data=train_data,
                        item_embed=item_embed,
                        save_dir=result_dir
                        )
    print(f'Finish training! Elapsed time: {dt.datetime.now().replace(microsecond=0) - start_time}')


parser = argparse.ArgumentParser("Graph Learning")

# Datasets
parser.add_argument('--train-dirs', type=str, help='Directories containing training data, sep by commas')
parser.add_argument('--valid-dirs', type=str, help='Directories containing validation data, sep by commas')
parser.add_argument('--result-dir', type=str, help='Directory to save everything')
parser.add_argument('--rename-item', action='store_true', default=False,
                    help='Same items in different timeframes will have different nodes or not.'
                         'Only work for training data. For validation set, must use only 1 timeframe.'
                         'Still meet error, dont enable.')

# Model
parser.add_argument('--n-layers', type=int, default=3, help='Number of layers, including embedding layer.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout ratio')
parser.add_argument('--pred', type=str, default='cos', choices=['sigmoid', 'cos'], help='Prediction method')
parser.add_argument('--out-dim', type=int, default=128, help='Output dimension')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='Hidden dimension. Be careful! Increasing this number will increase the memory so much')

# Trainer
parser.add_argument('--sub-train-sample-size', type=float, default=0.2, help='Subset of training data to validate')
parser.add_argument('--aggregator-hetero', type=str, default='sum', choices=['mean', 'sum', 'max'],
                    help='Function to aggregate messages from different edge type')
parser.add_argument('--aggregator-homo', type=str, default='mean', choices=['mean', 'mean_nn', 'max_nn'],
                    help='Function to aggregate messages from same edge type')
parser.add_argument('--loss', type=str, default='hinge', choices=['hinge', 'bce'], help='Loss function')
parser.add_argument('--delta', type=float, default=0.1, help='Margin in hinge loss if used')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay in SGD')
parser.add_argument('--num-epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--print-every', type=int, default=10, help='Print loss every these iterations')
parser.add_argument('--neg-sample-size', type=int, default=4, help='Number of samples for negative sampling')
parser.add_argument('--edge-batch-size', type=int, default=1024, help='Number of edges in a batch')
parser.add_argument('--node-batch-size', type=int, default=2048, help='Number of nodes in a batch')
parser.add_argument('--neg-sampler', type=str, default='edge', help='Negative sampling', choices=['edge', 'link'])
# parser.add_argument('--precision-at-k', type=int, default=5, help='Precision/Recall at k to evaluate (deprecated)')
parser.add_argument('--num-workers', type=int, default=10, help='Number of cores of CPU to use')
parser.add_argument('--use-ddp', action='store_true', default=False, help='Only use for multi-GPU')
parser.add_argument('--n-neighbors', type=int, default=256,
                    help='Number of random neighbors to aggregate. '
                         'Set 0 to use all neighbors, but not recommended because the memory will explode. '
                         'For now we use the same number for all layers and all edge types. '
                         'Later, we will set different numbers by passing a list[dict[e_type, int]].')

parser.add_argument('--pre-aggregate', action='store_true', default=False, help='Pre-aggreate')
parser.add_argument('--use-edge-feature', action='store_true', default=True, help='Use edge features')
parser.add_argument('--aggregator-edge', type=str, default='u_cat_e', choices=['u_mul_e', 'u_add_e', 'u_cat_e'],
                    help='Function to aggregate messages from edge if `use-edge-feature` is True')


# TODO: Best params for now
#   loss: hinge
#   pred: cos
#   num-neighbors: 256
#   delta: 0.2 (the higher, the more epochs to converge. 0.1 worse, didn't try 0.3)
#   neg-sample-size: 64

if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything()
    main()
