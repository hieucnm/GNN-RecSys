import argparse
import warnings

import torch

from custom.datasets import DataSet
from custom.utils_data import get_edge_data_loaders

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    train_data = DataSet(args)
    train_graph = train_data.init_graph()
    print(train_graph)

    # edge_loader_train = get_edge_data_loaders(train_graph)
    # print(edge_loader_train)


parser = argparse.ArgumentParser("Graph Learning")
parser.add_argument('-ap', '--ad-path', type=str, help='Path of past interactions of user-item.')
parser.add_argument('-gp', '--group-chat-path', type=str, help='Path of past interactions of user-user in group_chat.')
parser.add_argument('-lp', '--label-path', type=str, help='Path of future conversions of user-item.')
parser.add_argument('-up', '--user-feature-path', type=str, help='Path of user features.')
parser.add_argument('-rp', '--result-dir', type=str, default='examples/results', help='Directory to save everything.')
parser.add_argument('--out-dim', type=int, default=128, help='Output dimension')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='Hidden dimension. Be careful! Increasing this number will increase the memory so much. ')

parser.add_argument('--n-layers', type=int, default=4, help='Number of layers, including embedding layer.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout ratio')
parser.add_argument('--pred', type=str, default='cos', choices=['nn', 'cos'], help='Way to predict scores of link')
parser.add_argument('--delta', type=float, default=0.05, help='Margin used in maximal margin loss')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay in SGD')
parser.add_argument('--num-epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--neg-sample-size', type=int, default=3, help='Number of samples when doing negative sampling')
parser.add_argument('--edge-batch-size', type=int, default=1024, help='Number of edges in a train / validation batch')
parser.add_argument('--node-batch-size', type=int, default=1024, help='Number of nodes in a train / validation batch')
parser.add_argument('--precision-at-k', type=int, default=5, help='Precision/Recall at this number will be computed')
parser.add_argument('--num-workers', type=int, default=8, help='Number of cores of CPU to use')
parser.add_argument('--num-neighbors', type=int, default=512,
                    help='Number of random neighbors to aggregate. '
                         'Set 0 to use all neighbors, but not recommended because the memory will explode. '
                         'For now we use the same number for all layers. '
                         'Later, we will set different numbers for different layers by passing a list.')

if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Using device: {device.type}. All arguments: {args}")
    main()
