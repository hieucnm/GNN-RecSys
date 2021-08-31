import os
import time

import numpy as np
import torch
from dgl import heterograph
import datetime as dt
import argparse
from glob import glob

from src.utils_data import DataLoader, assign_graph_features, summary_data_sets, calculate_num_batches
from src.utils import read_data, save_txt, save_everything
from src.model import ConvModel
from src.sampling import train_valid_split, generate_dataloaders, generate_test_loaders
from src.train.run import train_model, get_embeddings
from src.utils_vizualization import plot_train_loss
from src.metrics import (create_already_bought, create_ground_truth,
                         get_metrics_at_k, get_recs)
from src.evaluation import explore_recs, explore_sports, check_coverage
from presplit import presplit_data
from src.utils_data import FixedParameters

# import dgl.multiprocessing as mp
import torch.distributed as dist
import torch.multiprocessing as mp

from logging_config import get_logger
log = get_logger(__name__)


class TrainDataPaths:
    def __init__(self):
        global args
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_interaction_path = args.interaction_path
        self.user_feat_path = args.user_feature_path
        self.result_dir = f'{args.result_dir}/{timestamp}'
        os.makedirs(self.result_dir)
        self.log_filepath = f'{self.result_dir}/running_log.txt'


# noinspection PyShadowingNames
def setup():
    global args

    fixed_params = FixedParameters(args)

    print('--> loading data ...')
    start_time = time.time()
    train_data_paths = TrainDataPaths()
    full_interaction_data = read_data(train_data_paths.full_interaction_path)
    train_df, test_df = presplit_data(full_interaction_data,
                                      uid_column=fixed_params.uid_column,
                                      date_column=fixed_params.date_column,
                                      num_min=3,
                                      test_size_days=1,
                                      sort=True
                                      )
    print('--> loaded data, elapsed time =', time.time() - start_time)

    train_data_paths.train_path = train_df
    train_data_paths.test_path = test_df
    print('--> initializing data ...')
    start_time = time.time()
    data = DataLoader(train_data_paths, fixed_params)
    print('--> initialized data, elapsed time =', time.time() - start_time)

    print('--> creating graph ...')
    start_time = time.time()
    valid_graph = heterograph(data.graph_schema)
    print('--> created graph, elapsed time =', time.time() - start_time)

    print('--> assigning features to graph ...')
    start_time = time.time()
    valid_graph = assign_graph_features(valid_graph, fixed_params, data)
    print('--> assigned features to graph, elapsed time =', time.time() - start_time)

    print('--> splitting train, valid, test set ...')
    start_time = time.time()
    (
        train_graph,
        train_eid_dict,
        valid_eid_dict,
        sub_train_uid,
        valid_uid,
        test_uid,
        all_iid,
        ground_truth_sub_train,
        ground_truth_valid,
        all_eid_dict
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
    print('--> split done, elapsed time =', time.time() - start_time)
    summary_data_sets(train_eid_dict, valid_eid_dict, sub_train_uid, valid_uid, test_uid,
                      all_iid, ground_truth_sub_train, ground_truth_valid, all_eid_dict)

    # Calculate approximate number of nodes in a batch, based on number of edges in a batch
    (
        num_batches_train,
        num_batches_sub_train,
        num_batches_test,
        num_batches_val_loss,
        num_batches_val_metrics
    ) = calculate_num_batches(train_eid_dict,
                              valid_eid_dict,
                              sub_train_uid,
                              valid_uid,
                              test_uid,
                              all_iid,
                              edge_bs=fixed_params.edge_batch_size,
                              node_bs=fixed_params.node_batch_size)

    dim_dict = {'user': valid_graph.nodes['user'].data['features'].shape[1],
                'item': data.item_id_df.shape[0],
                'out': args.out_dim,
                'hidden': args.hidden_dim}

    return (valid_graph, train_graph, train_eid_dict, valid_eid_dict, all_eid_dict,
            sub_train_uid, valid_uid, test_uid, all_iid,
            fixed_params, dim_dict, train_data_paths, data,
            num_batches_train, num_batches_val_loss, ground_truth_sub_train, ground_truth_valid)


def train(device, params):

    data, args = params
    (
        valid_graph,
        train_graph,
        train_eid_dict,
        valid_eid_dict,
        all_eid_dict,
        sub_train_uid,
        valid_uid,
        test_uid,
        all_iid,
        fixed_params,
        dim_dict,
        train_data_paths,
        data,
        num_batches_train,
        num_batches_val_loss,
        ground_truth_sub_train,
        ground_truth_valid
    ) = data

    (
        edge_loader_train,
        edge_loader_valid,
        node_loader_sub_train,
        node_loader_valid,
        _
    ) = generate_dataloaders(valid_graph,
                             train_graph,
                             train_eid_dict,
                             valid_eid_dict,
                             sub_train_uid,
                             valid_uid,
                             test_uid,
                             all_iid,
                             fixed_params,
                             num_workers=0,
                             n_layers=args.n_layers,
                             neg_sample_size=args.neg_sample_size,
                             device=device,
                             use_ddp=True
                             )

    model = ConvModel(valid_graph,
                      args.n_layers,
                      dim_dict,
                      fixed_params.norm,
                      args.dropout,
                      fixed_params.aggregator_type,
                      fixed_params.pred,
                      fixed_params.aggregator_hetero
                      ).to(device)
    # Wrap the model with distributed data parallel module.
    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Train
    hp_sentence = vars(args)
    hp_sentence.update(vars(fixed_params))
    hp_sentence = f'{str(hp_sentence)[1: -1]} \n'
    save_txt(f'\n \n START - Hyper-parameters \n{hp_sentence}', train_data_paths.log_filepath, "a")
    print('->> training ...')
    trained_model, viz, best_metrics = train_model(
        model,
        train_graph=train_graph,
        valid_graph=valid_graph,
        edgeloader_train=edge_loader_train,
        edgeloader_valid=edge_loader_valid,
        nodeloader_valid=node_loader_valid,
        nodeloader_subtrain=node_loader_sub_train,
        num_batches_train=num_batches_train,
        num_batches_val_loss=num_batches_val_loss,
        ground_truth_subtrain=ground_truth_sub_train,
        ground_truth_valid=ground_truth_valid,
        bought_eids=train_eid_dict[('user', 'converts', 'item')],
        remove_already_bought=True,
        get_metrics=True,
        loss_fn=fixed_params.loss_fn,
        device=device,
        lr=args.lr,
        pred=args.pred,
        delta=args.delta,
        out_dim=args.out_dim,
        patience=args.patience,
        num_epochs=args.num_epochs,
        start_epoch=args.start_epoch,
        neg_sample_size=args.neg_sample_size,
        k=fixed_params.sample_size,
        optimizer=fixed_params.optimizer,
        use_recency=fixed_params.use_recency,
        result_filepath=train_data_paths.log_filepath,
        remove_false_negative=fixed_params.remove_false_negative,
        save_dir=train_data_paths.result_dir,
        gpu_id=0 if device.type == 'cpu' else device.index
    )

    # Save everything
    save_everything(valid_graph, data, args, fixed_params, hp_sentence, viz,
                    save_dir=train_data_paths.result_dir)

    # Report performance on validation set
    print('--> training finshed! Report metrics:')
    sentence = ("BEST VALIDATION: Precision "
                "{:.3f}% | Recall {:.3f}% | Coverage {:.2f}%"
                .format(best_metrics['precision'] * 100,
                        best_metrics['recall'] * 100,
                        best_metrics['coverage'] * 100))
    log.info(sentence)
    save_txt(sentence, train_data_paths.log_filepath, mode='a')


# def test():
#     # Report performance on test set
#     log.debug('--> testing ...')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     node_loader_test = generate_test_loaders(train_graph, test_uid, all_iid, fixed_params,
#                                              num_workers=0,
#                                              n_layers=args.n_layers)
#     model = ConvModel(valid_graph,
#                       args.n_layers,
#                       dim_dict,
#                       fixed_params.norm,
#                       args.dropout,
#                       fixed_params.aggregator_type,
#                       fixed_params.pred,
#                       fixed_params.aggregator_hetero
#                       )
#     latest_model_path = sorted([f for f in glob(train_data_paths.result_dir) if f.endswith('pth')])[-1]
#     model.load_state_dict(torch.load(latest_model_path, map_location=device))
#     log.debug('--> model loaded! Calculating test metrics ...')
#     model.eval()
#     with torch.no_grad():
#         embeddings = get_embeddings(valid_graph, args.out_dim, model, node_loader_test, device)
#
#         for ground_truth in [data.ground_truth_convert_test, data.ground_truth_test]:
#             precision, recall, coverage, auc = get_metrics_at_k(embeddings,
#                                                            valid_graph,
#                                                            model,
#                                                            args.out_dim,
#                                                            ground_truth,
#                                                            all_eid_dict[('user', 'converts', 'item')],
#                                                            fixed_params.k,
#                                                            False,  # Remove already bought
#                                                            device,
#                                                            args.pred)
#             sentence = ("TEST Precision {:.3f}% | Recall {:.3f}% | Coverage {:.2f}%"
#                         .format(precision * 100,
#                                 recall * 100,
#                                 coverage * 100))
#             log.info(sentence)
#             save_txt(sentence, train_data_paths.log_filepath, mode='a')


def init_process(pid, num_pid, fn, fn_params=None):
    dist_init_method = 'tcp://{ip}:{port}'.format(ip='10.50.9.13', port=12345)
    if torch.cuda.is_available():
        backend = 'nccl'
        device = torch.device(f'cuda:{pid}')
        torch.cuda.set_device(pid)
    else:
        backend = 'gloo'
        device = torch.device('cpu')
    dist.init_process_group(backend, init_method=dist_init_method, rank=pid, world_size=num_pid)

    # Run
    fn(device, fn_params)


def parse_args():
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

    parser.add_argument('--use-ddp', action='store_true', default=True, help='Use DistributedDataParallel')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("args: ", vars(args))

    training_data = setup()
    valid_graph = training_data[0]
    valid_graph.create_formats_()

    num_gpus = torch.cuda.device_count()
    print(f'--> using {num_gpus} GPUs')

    processes = []
    mp.set_start_method("spawn")
    for gpu_id in range(num_gpus):
        p = mp.Process(target=init_process, args=(gpu_id, num_gpus, train, (training_data, args)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()