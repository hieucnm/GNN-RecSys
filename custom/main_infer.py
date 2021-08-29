import argparse
import datetime as dt
import pickle
import sys
import warnings

import torch

from custom.dataloaders import get_node_loader
from custom.datasets import DataSet
from custom.evaluation import Predictor
from custom.logger import Logger
from custom.models import ConvModel

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    # Redirect print to both console and log file
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f'{args.result_dir}/{timestamp}'
    sys.stdout = Logger(f'{result_dir}/running_log.txt')
    print(f'Everything will be saved to {result_dir}')

    params = pickle.load(open(args.args_path))

    print(f'Using device: {device.type}')
    print(f'All arguments: {args}')

    print("Loading inference data ...")
    data = DataSet(data_dirs=args.data_dir)
    assert data.num_items == params['dim_dict']['item'], \
        "Number of items not equal: {} in data and {} in model".format(data.num_items, params['dim_dict']['item'])

    graph = data.init_graph()
    node_loader, _ = get_node_loader(graph=graph,
                                     adjust_graph=graph,
                                     user_id=data.user_id,
                                     item_id=data.item_id,
                                     num_neighbors=params['num_neighbors'],
                                     n_layers=params['n_layers'],
                                     node_batch_size=args.node_batch_size,
                                     num_workers=args.num_workers
                                     )

    print("Initializing model ...")
    model = ConvModel(graph=graph,
                      dim_dict=params['dim_dict'],
                      label_edge_types=[],
                      n_layers=params['n_layers'],
                      pred=params['pred'],
                      norm=params['norm'],
                      dropout=params['dropout'],
                      aggregator_homo=params['aggregator_homo'],
                      aggregator_hetero=params['aggregator_hetero'],
                      user_id=data.user_id,
                      item_id=data.item_id
                      )
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    predictor = Predictor(model=model,
                          user_id=data.user_id,
                          item_id=data.item_id,
                          save_dir=args.result_dir,
                          node2uid=data.get_node2uid_dict(),
                          node2iid=data.get_node2iid_dict(),
                          print_every=args.print_every
                          )

    # INFERENCE
    start_time = dt.datetime.now().replace(microsecond=0)
    print('Start predicting:')
    predictor.predict_and_save_on_batches(graph, node_loader)
    print(f'Finish predicting! Elapsed time: {dt.datetime.now().replace(microsecond=0) - start_time}')


parser = argparse.ArgumentParser("Graph Learning")

# Paths
parser.add_argument('--args-path',  type=str, help='Path containing all saved arguments after training model')
parser.add_argument('--model-path',  type=str, help='Path of the trained model')
parser.add_argument('--inference-dir',  type=str, help='Directory containing inference data')
parser.add_argument('--result-dir', type=str, default='examples/results', help='Directory to save everything')
parser.add_argument('--print-every', type=int, default=4, help='Print loss every these iterations')
parser.add_argument('--node-batch-size', type=int, default=128, help='Number of nodes in a batch')
parser.add_argument('--num-workers', type=int, default=8, help='Number of cores of CPU to use')


if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything()
    main()
