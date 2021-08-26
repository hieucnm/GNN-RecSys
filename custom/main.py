import argparse
import warnings

import torch
import torch.optim

from custom.datasets import DataSet
from custom.utils_data import get_edge_loader, get_node_loader
from custom.models import ConvModel
from custom.trainers import Trainer, get_embeddings
from losses import MaxMarginLoss, BCELossCustom
from collections import defaultdict

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    train_data = DataSet(data_dir=args.train_dir)
    train_graph = train_data.init_graph()
    label_edge_types = train_data.label_edge_types
    train_edge_loader = get_edge_loader(train_graph,
                                        label_edge_types=label_edge_types,
                                        num_neighbors=args.num_neighbors,
                                        n_layers=args.n_layers,
                                        neg_sample_size=args.neg_sample_size,
                                        edge_batch_size=args.edge_batch_size,
                                        num_workers=args.num_workers,
                                        use_ddp=args.use_ddp
                                        )
    sub_train_node_loader, sub_train_ground_truth = get_node_loader(graph=train_graph,
                                                                    label_edge_types=label_edge_types,
                                                                    item_id=train_data.item_id,
                                                                    sample_size=args.sub_train_sample_size,
                                                                    num_neighbors=args.num_neighbors,
                                                                    n_layers=args.n_layers,
                                                                    node_batch_size=args.node_batch_size,
                                                                    num_workers=args.num_workers
                                                                    )
    valid_data = DataSet(data_dir=args.valid_dir)
    valid_graph = valid_data.init_graph()
    valid_edge_loader = get_edge_loader(valid_graph,
                                        label_edge_types=label_edge_types,
                                        num_neighbors=args.num_neighbors,
                                        n_layers=args.n_layers,
                                        neg_sample_size=args.neg_sample_size,
                                        edge_batch_size=args.edge_batch_size,
                                        num_workers=args.num_workers,
                                        use_ddp=args.use_ddp
                                        )
    valid_node_loader, valid_ground_truth = get_node_loader(graph=valid_graph,
                                                            label_edge_types=label_edge_types,
                                                            item_id=valid_data.item_id,
                                                            num_neighbors=args.num_neighbors,
                                                            n_layers=args.n_layers,
                                                            node_batch_size=args.node_batch_size,
                                                            num_workers=args.num_workers
                                                            )

    dim_dict = {'user': train_data.num_user_features,
                'item': train_data.num_items,
                'out': args.out_dim,
                'hidden': args.hidden_dim}

    model = ConvModel(g=train_graph,
                      dim_dict=dim_dict,
                      n_layers=args.n_layers,
                      pred=args.pred,
                      norm=True,
                      dropout=args.dropout,
                      aggregator_homo=args.aggregator_homo,
                      aggregator_hetero=args.aggregator_hetero
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

    # TRAIN
    metrics = defaultdict(list)
    print('Start training:')
    for epoch in range(args.num_epochs):
        print('--> Epoch {}/{}: Training ...'.format(epoch, args.num_epochs))
        epoch_avg_train_loss = trainer.train(train_edge_loader)
        metrics['train_loss'].append(epoch_avg_train_loss)

        # print('--> Epoch {}/{}: Calculating training metrics ...'.format(epoch, args.num_epochs))
        # y = get_embeddings(graph=train_graph, embed_dim=dim_dict['out'],
        #                    model=model, node_loader=sub_train_node_loader)

        print('--> Epoch {}/{}: Calculating validation loss ...'.format(epoch, args.num_epochs))
        epoch_avg_valid_loss = trainer.calculate_loss(valid_edge_loader)
        metrics['valid_loss'].append(epoch_avg_valid_loss)

        # calculate valid loss
        # calculate train & valid precision, recall, coverage, auc
        # save model & write log


parser = argparse.ArgumentParser("Graph Learning")
parser.add_argument('--train-dir', type=str, help='Directory contains all training data')
parser.add_argument('--valid-dir', type=str, help='Directory contains all validation data')
parser.add_argument('--test-dir', type=str, help='Directory contains all testing data')
parser.add_argument('--result-dir', type=str, default='examples/results', help='Directory to save everything')
parser.add_argument('--sub-train-sample-size', type=float, default=0.1, help='Fraction to get subset of training data')

parser.add_argument('--out-dim', type=int, default=128, help='Output dimension')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='Hidden dimension. Be careful! Increasing this number will increase the memory so much')
parser.add_argument('--n-layers', type=int, default=4, help='Number of layers, including embedding layer.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout ratio')
parser.add_argument('--pred', type=str, default='cos', choices=['sigmoid', 'cos'], help='Way to predict scores of link')
parser.add_argument('--loss', type=str, default='hinge', choices=['hinge', 'bce'], help='Loss function')
parser.add_argument('--aggregator-hetero', type=str, default='sum', choices=['mean', 'sum', 'max'],
                    help='Function to aggregate messages from different edge type')
parser.add_argument('--aggregator-homo', type=str, default='mean', choices=['mean', 'sum', 'max'],
                    help='Function to aggregate messages from same edge type')

parser.add_argument('--delta', type=float, default=0.05, help='Margin in hinge loss if used')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay in SGD')
parser.add_argument('--num-epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--print-every', type=int, default=10, help='Print loss every these iterations')
parser.add_argument('--neg-sample-size', type=int, default=3, help='Number of samples when doing negative sampling')
parser.add_argument('--edge-batch-size', type=int, default=1024, help='Number of edges in a train / validation batch')
parser.add_argument('--node-batch-size', type=int, default=1024, help='Number of nodes in a train / validation batch')
parser.add_argument('--precision-at-k', type=int, default=5, help='Precision/Recall at this number will be computed')
parser.add_argument('--num-workers', type=int, default=8, help='Number of cores of CPU to use')
parser.add_argument('--use-ddp', action='store_true', default=False, help='Only use for multi-GPU')
parser.add_argument('--num-neighbors', type=int, default=512,
                    help='Number of random neighbors to aggregate. '
                         'Set 0 to use all neighbors, but not recommended because the memory will explode. '
                         'For now we use the same number for all layers. '
                         'Later, we will set different numbers for different layers by passing a list.')

if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Using device: {device.type}. All arguments: {args}")
    main()
