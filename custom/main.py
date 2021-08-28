import argparse
import datetime as dt
import sys
import warnings
from collections import defaultdict

import torch
import torch.optim

from custom.datasets import DataSet
from custom.losses import MaxMarginLoss, BCELossCustom
from custom.models import ConvModel
from custom.trainers import Trainer
from custom.evaluation import Evaluator
from custom.utils_data import get_edge_loader, get_node_loader, remove_label_edges, get_label_edges
from custom.logger import Logger

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')  # to debug


def main():

    # Redirect print to both console and log file
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f'{args.result_dir}/{timestamp}'
    # sys.stdout = Logger(f'{result_dir}/running_log.txt')

    print(f"Using device: {device.type}. All arguments: {args}")

    print("Loading training data ...")
    train_data = DataSet(data_dir=args.train_dir)
    train_graph = train_data.init_graph()
    label_edge_types = train_data.label_edge_types
    train_label_eid = get_label_edges(graph=train_graph, label_edge_types=label_edge_types)
    train_adjust_graph = remove_label_edges(graph=train_graph, label_edge_types=label_edge_types)

    train_edge_loader = get_edge_loader(graph=train_graph,
                                        adjust_graph=train_adjust_graph,
                                        label_eid_dict=train_label_eid,
                                        label_edge_types=label_edge_types,
                                        num_neighbors=args.num_neighbors,
                                        n_layers=args.n_layers,
                                        neg_sample_size=args.neg_sample_size,
                                        edge_batch_size=args.edge_batch_size,
                                        num_workers=args.num_workers,
                                        use_ddp=args.use_ddp
                                        )
    sub_train_node_loader, sub_train_ground_truth = get_node_loader(graph=train_graph,
                                                                    adjust_graph=train_adjust_graph,
                                                                    label_eid_dict=train_label_eid,
                                                                    label_edge_types=label_edge_types,
                                                                    user_id=train_data.user_id,
                                                                    item_id=train_data.item_id,
                                                                    sample_size=args.sub_train_sample_size,
                                                                    num_neighbors=args.num_neighbors,
                                                                    n_layers=args.n_layers,
                                                                    node_batch_size=args.node_batch_size,
                                                                    num_workers=args.num_workers
                                                                    )

    print("Loading validation data ...")
    valid_data = DataSet(data_dir=args.valid_dir)
    valid_graph = valid_data.init_graph()
    label_edge_types = valid_data.label_edge_types
    valid_label_eid = get_label_edges(graph=valid_graph, label_edge_types=label_edge_types)
    valid_adjust_graph = remove_label_edges(graph=valid_graph, label_edge_types=label_edge_types)

    valid_edge_loader = get_edge_loader(graph=valid_graph,
                                        adjust_graph=valid_adjust_graph,
                                        label_eid_dict=valid_label_eid,
                                        num_neighbors=args.num_neighbors,
                                        n_layers=args.n_layers,
                                        neg_sample_size=args.neg_sample_size,
                                        edge_batch_size=args.edge_batch_size,
                                        num_workers=args.num_workers,
                                        use_ddp=args.use_ddp
                                        )
    valid_node_loader, valid_ground_truth = get_node_loader(graph=valid_graph,
                                                            adjust_graph=valid_adjust_graph,
                                                            label_eid_dict=valid_label_eid,
                                                            label_edge_types=label_edge_types,
                                                            user_id=valid_data.user_id,
                                                            item_id=valid_data.item_id,
                                                            num_neighbors=args.num_neighbors,
                                                            n_layers=args.n_layers,
                                                            node_batch_size=args.node_batch_size,
                                                            num_workers=args.num_workers
                                                            )
    print("Creating model ...")
    dim_dict = {'user': train_data.num_user_features,
                'item': train_data.num_items,
                'out': args.out_dim,
                'hidden': args.hidden_dim}

    # We want the model to not contain the label edges
    model = ConvModel(graph=train_adjust_graph,
                      dim_dict=dim_dict,
                      label_edge_types=label_edge_types,
                      n_layers=args.n_layers,
                      pred=args.pred,
                      norm=True,
                      dropout=args.dropout,
                      aggregator_homo=args.aggregator_homo,
                      aggregator_hetero=args.aggregator_hetero,
                      user_id=train_data.user_id,
                      item_id=train_data.item_id
                      )
    if device.type != 'cpu':
        model = model.to(device)
    print(model.eval())

    criterion = MaxMarginLoss(delta=args.delta) if args.loss == 'hinge' else BCELossCustom()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      device=device,
                      print_every=args.print_every
                      )
    evaluator = Evaluator(model=model,
                          k=args.precision_at_k,
                          user_id=train_data.user_id,
                          item_id=train_data.item_id,
                          print_every=args.print_every
                          )

    # TRAIN
    metrics = defaultdict(list)
    start_time = dt.datetime.now()
    print('Start training:')
    for epoch in range(1, args.num_epochs + 1):

        print('--> Epoch {}/{}: Training ...'.format(epoch, args.num_epochs))
        train_avg_loss = trainer.train(train_edge_loader)

        # torch.save(model.state_dict(), f'{result_dir}/model_ep_{epoch}.pth')
        # print(f"--> Model at epoch {epoch} saved!")

        print('--> Epoch {}/{}: Evaluating sub-train ...'.format(epoch, args.num_epochs))
        train_acc, train_auc, train_coverage = evaluator.evaluate_on_batches(
            graph=train_graph,
            node_loader=sub_train_node_loader,
            ground_truth=sub_train_ground_truth
        )

        print('--> Epoch {}/{}: Calculating validation loss ...'.format(epoch, args.num_epochs))
        val_avg_loss = trainer.calculate_loss(valid_edge_loader)

        print('--> Epoch {}/{}: Evaluating validation ...'.format(epoch, args.num_epochs))
        val_acc, val_auc, val_coverage = evaluator.evaluate_on_batches(
            graph=valid_graph,
            node_loader=valid_node_loader,
            ground_truth=valid_ground_truth
        )

        report = "--> Finish epoch {}/{}\n" \
            "--> Training:   Loss {:.5f} | Acc. {:.3f}% | AUC {:.2f}% | Coverage {:.2f}%\n"\
            "--> Validation: Loss {:.5f} | Acc. {:.3f}% | AUC {:.2f}% | Coverage {:.2f}%"\
            .format(epoch, args.num_epochs,
                    train_avg_loss, train_acc * 100, train_auc * 100, train_coverage * 100,
                    val_avg_loss, val_acc * 100, val_auc * 100, val_coverage * 100)
        print(report)
        metrics['train_avg_loss'].append(train_avg_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_auc'].append(train_auc)
        metrics['train_coverage'].append(train_coverage)
        metrics['val_avg_loss'].append(val_avg_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_auc'].append(val_auc)
        metrics['val_coverage'].append(val_coverage)

    print(f'Finish training! Elapsed time: {dt.datetime.now() - start_time} seconds')
    # TODO: save params, save metrics and metrics plots


parser = argparse.ArgumentParser("Graph Learning")

# Paths
parser.add_argument('--train-dir',  type=str, help='Directory contains all training data')
parser.add_argument('--valid-dir',  type=str, help='Directory contains all validation data')
parser.add_argument('--test-dir',   type=str, help='Directory contains all testing data')
parser.add_argument('--result-dir', type=str, default='examples/results', help='Directory to save everything')

# Model
parser.add_argument('--n-layers', type=int, default=4, help='Number of layers, including embedding layer.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout ratio')
parser.add_argument('--pred', type=str, default='cos', choices=['sigmoid', 'cos'], help='Prediction method')
parser.add_argument('--out-dim', type=int, default=128, help='Output dimension')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='Hidden dimension. Be careful! Increasing this number will increase the memory so much')


# Trainer
parser.add_argument('--sub-train-sample-size', type=float, default=0.1, help='Fraction to get subset of training data')
parser.add_argument('--aggregator-hetero', type=str, default='sum', choices=['mean', 'sum', 'max'],
                    help='Function to aggregate messages from different edge type')
parser.add_argument('--aggregator-homo', type=str, default='mean', choices=['mean', 'mean_nn', 'max_nn'],
                    help='Function to aggregate messages from same edge type')
parser.add_argument('--loss', type=str, default='hinge', choices=['hinge', 'bce'], help='Loss function')
parser.add_argument('--delta', type=float, default=0.05, help='Margin in hinge loss if used')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay in SGD')
parser.add_argument('--num-epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--print-every', type=int, default=10, help='Print loss every these iterations')
parser.add_argument('--neg-sample-size', type=int, default=4, help='Number of samples when doing negative sampling')
parser.add_argument('--edge-batch-size', type=int, default=64, help='Number of edges in a batch')
parser.add_argument('--node-batch-size', type=int, default=64, help='Number of nodes in a batch')
parser.add_argument('--precision-at-k', type=int, default=5, help='Precision/Recall at this number will be computed')
parser.add_argument('--num-workers', type=int, default=8, help='Number of cores of CPU to use')
parser.add_argument('--use-ddp', action='store_true', default=False, help='Only use for multi-GPU')
parser.add_argument('--num-neighbors', type=int, default=256,
                    help='Number of random neighbors to aggregate. '
                         'Set 0 to use all neighbors, but not recommended because the memory will explode. '
                         'For now we use the same number for all layers. '
                         'Later, we will set different numbers for different layers by passing a list.')

if __name__ == '__main__':
    args = parser.parse_args()
    main()
