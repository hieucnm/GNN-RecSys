import math

import pandas as pd
import torch

from base.src.builder import (create_ids, df_to_adjacency_list, import_features,
                              filter_unseen_item, report_user_coverage)
from base.src.model import MaxMarginLoss, BCELossCustom
from base.src.utils import read_data


class DataPaths:
    def __init__(self):
        self.result_filepath = 'TXT FILE WHERE TO LOG THE RESULTS .txt'
        self.sport_feat_path = 'FEATURE DATASET, SPORTS (sport names) .csv'
        self.train_path = 'INTERACTION LIST, USER-ITEM (Train dataset).csv'
        self.test_path = 'INTERACTION LIST, USER-ITEM (Train dataset).csv'
        self.item_sport_path = 'INTERACTION LIST, ITEM-SPORT .csv'
        self.user_sport_path = 'INTERACTION LIST, USER-SPORT .csv'
        self.sport_sportg_path = 'INTERACTION LIST, SPORT-SPORT .csv'
        self.item_feat_path = 'FEATURE DATASET, ITEMS .csv'
        self.user_feat_path = 'FEATURE DATASET, USERS.csv'
        self.sport_onehot_path = 'FEATURE DATASET, SPORTS (one-hot vectors) .csv'


# noinspection PyUnresolvedReferences
class FixedParameters:
    def __init__(self, args):
        """
        All parameters that are fixed, i.e. not part of the hyperparametrization.

        Attributes
        ----------
        uid_column :
            Identifier for the users.
        Days_of_purchases (Days_of_clicks) :
            Number of days of purchases (clicks) that should be kept in the dataset.
            Intuition is that interactions of 12+ months ago might not be relevant. Max is 710 days
            Those that do not have any remaining interactions will be fed recommendations from another
            model.
        Discern_clicks :
            Clicks and purchases will be considered as 2 different edge types
        Duplicates :
            Determines how to handle duplicates in the training set. 'count_occurrence' will drop all
            duplicates except last, and the number of interactions will be stored in the edge feature.
            If duplicates == 'count_occurrence', aggregator_type needs to handle edge feature. 'keep_last'
            will drop all duplicates except last. 'keep_all' will conserve all duplicates.
        Explore :
            Print examples of recommendations and of similar sports
        uid_column, iid_column :
            Identifier for the users & items.
        Lifespan_of_items :
            Number of days since most recent transactions for an item to be considered by the
            model. Max is 710 days. Won't make a difference is it is > Days_of_interaction.
        Num_choices :
            Number of examples of recommendations and similar sports to print
        Patience :
            Number of epochs to wait for Early stopping
        Pred :
            Function that takes as input embedding of user and item, and outputs ratings. Choices : 'cos' for cosine
            similarity, 'nn' for multilayer perceptron with sigmoid function at the end
        Start_epoch :
            Load model from a previous epoch
        Train_on_clicks :
            When parametrizing the GNN, edges of purchases are always included. If true, clicks will also
            be included
        """

        self.days_of_converts = 365  # Max is 710
        self.days_of_clicks = 30  # Max is 710
        self.lifespan_of_items = 180
        self.etype = [('user', 'converts', 'item')]
        self.reverse_etype = {('user', 'converts', 'item'): ('item', 'converted-by', 'user')}
        self.discern_clicks = True
        if self.discern_clicks:
            self.etype.append(('user', 'clicks', 'item'))
            self.reverse_etype[('user', 'clicks', 'item')] = ('item', 'clicked-by', 'user')

        self.num_choices = 10
        self.explore = True
        self.remove_false_negative = True  # default True
        self.remove_train_eids = True  # remove the edges you want to predict to avoid data leakage, default False
        self.train_on_clicks = True
        self.remove_on_inference = .7
        self.run_inference = 1
        self.subtrain_size = 0.05
        self.valid_size = 0.05
        self.optimizer = torch.optim.Adam

        self.duplicates = args.duplicates  # 'keep_last', 'keep_all', 'count_occurrence'
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.patience = args.patience
        self.edge_batch_size = args.edge_batch_size
        self.node_batch_size = args.node_batch_size
        self.k = args.precision_at_k
        self.num_neighbors = args.num_neighbors
        self.pred = args.pred
        if self.pred == 'cos':
            self.loss_fn = MaxMarginLoss(
                delta=args.delta,
                neg_sample_size=args.neg_sample_size,
                remove_false_negative=self.remove_false_negative
            )
        else:
            self.loss_fn = BCELossCustom()

        self.uid_column = 'src_id'
        self.iid_column = 'ad_cate'
        self.date_column = 'date'
        self.conv_column = 'converted'
        self.use_recency = True
        self.clicks_sample = 1.
        self.converts_sample = 1.
        self.norm = True

        # self.dropout = .5  # HP
        # self.norm = False  # HP
        # self.use_popularity = False  # HP
        # self.days_popularity = 0  # HP
        # self.weight_popularity = 0.  # HP
        # self.use_recency = False  # HP
        self.aggregator_type = 'mean'  # HP, search `aggregator_type` in this project
        self.aggregator_hetero = 'mean'  # HP
        # self.purchases_sample = .5  # HP
        # self.clicks_sample = .4  # HP
        # self.embedding_layer = False  # HP
        # self.edge_update = True  # Removed implementation; not useful
        # self.automatic_precision = False  # Removed implementation; not useful


class DataLoader:
    """Data loading, cleaning and pre-processing."""

    # noinspection PyTupleAssignmentBalance
    def __init__(self, data_paths, fixed_params):

        self.data_paths = data_paths
        self.user_feat_df = read_data(data_paths.user_feat_path)
        self.user_item_train = read_data(data_paths.train_path)
        self.user_item_test = read_data(data_paths.test_path)

        self.user_item_test = filter_unseen_item(self.user_item_train, self.user_item_test, fixed_params.iid_column)
        report_user_coverage(self.user_item_train, self.user_item_test, fixed_params.uid_column)

        self.user_id_df = create_ids(self.user_item_train, fixed_params.uid_column)
        self.item_id_df = create_ids(self.user_item_train, fixed_params.iid_column)

        print("--> running df_to_adjacency_list ...")
        (
            self.adjacency_dict,
            self.ground_truth_test,
            self.ground_truth_convert_test,
            self.user_item_train_grouped
        ) = df_to_adjacency_list(
            user_item_train=self.user_item_train,
            user_item_test=self.user_item_test,
            user_id_df=self.user_id_df,
            item_id_df=self.item_id_df,
            uid_column=fixed_params.uid_column,
            iid_column=fixed_params.iid_column,
            date_column=fixed_params.date_column,
            conv_column=fixed_params.conv_column,
            discern_clicks=fixed_params.discern_clicks,
            duplicates=fixed_params.duplicates
        )

        print("--> assigning graph schema ...")
        if fixed_params.discern_clicks:
            self.graph_schema = {
                ('user', 'converts', 'item'):
                    list(zip(self.adjacency_dict['convert_src'], self.adjacency_dict['convert_dst'])),
                ('item', 'converted-by', 'user'):
                    list(zip(self.adjacency_dict['convert_dst'], self.adjacency_dict['convert_src'])),
                ('user', 'clicks', 'item'):
                    list(zip(self.adjacency_dict['clicks_src'], self.adjacency_dict['clicks_dst'])),
                ('item', 'clicked-by', 'user'):
                    list(zip(self.adjacency_dict['clicks_dst'], self.adjacency_dict['clicks_src'])),
            }
        else:
            self.graph_schema = {
                ('user', 'converts', 'item'):
                    list(zip(self.adjacency_dict['user_item_src'], self.adjacency_dict['user_item_dst'])),
                ('item', 'converted-by', 'user'):
                    list(zip(self.adjacency_dict['user_item_dst'], self.adjacency_dict['user_item_src'])),
            }


def assign_graph_features(graph,
                          fixed_params,
                          data,
                          ):
    """
    Assigns features to graph nodes and edges, based on data previously provided in the dataloader.

    Parameters
    ----------
    graph:
        Graph of type dgl.DGLGraph, with all the nodes & edges.
    fixed_params:
        All fixed parameters. The only fixed params used are related to id types and occurrences.
    data:
        Object that contains node feature dataframes, ID mapping dataframes and user item interactions.
    params:
        Parameters used in this function include popularity & recency hyperparameters.

    Returns
    -------
    graph:
        The input graph but with features assigned to its nodes and edges.
    """
    # Assign features
    features_dict = import_features(data.user_feat_df,
                                    data.user_id_df,
                                    data.item_id_df,
                                    fixed_params.uid_column,
                                    fixed_params.iid_column)
    graph.nodes['user'].data['features'] = features_dict['user']
    graph.nodes['item'].data['features'] = features_dict['item']

    # add date as edge feature
    date_col = fixed_params.date_column
    conv_col = fixed_params.conv_column
    if fixed_params.use_recency:
        df = data.user_item_train_grouped
        df['max_date'] = max(df[date_col])
        df['days_recency'] = (pd.to_datetime(df.max_date) - pd.to_datetime(df[date_col])).dt.days + 1
        if fixed_params.discern_clicks:
            recency_tensor_buys = torch.tensor(df[df[conv_col] == 1].days_recency.values)
            recency_tensor_clicks = torch.tensor(df[df[conv_col] == 0].days_recency.values)
            graph.edges['converts'].data['recency'] = recency_tensor_buys
            graph.edges['converted-by'].data['recency'] = recency_tensor_buys
            graph.edges['clicks'].data['recency'] = recency_tensor_clicks
            graph.edges['clicked-by'].data['recency'] = recency_tensor_clicks
        else:
            recency_tensor = torch.tensor(df.days_recency.values)
            graph.edges['converts'].data['recency'] = recency_tensor
            graph.edges['converted-by'].data['recency'] = recency_tensor

    if fixed_params.duplicates == 'count_occurrence':
        if fixed_params.discern_clicks:
            graph.edges['clicks'].data['occurrence'] = torch.tensor(data.adjacency_dict['clicks_num'])
            graph.edges['clicked-by'].data['occurrence'] = torch.tensor(data.adjacency_dict['clicks_num'])
            graph.edges['converts'].data['occurrence'] = torch.tensor(data.adjacency_dict['converts_num'])
            graph.edges['converted-by'].data['occurrence'] = torch.tensor(data.adjacency_dict['converts_num'])
        else:
            graph.edges['converts'].data['occurrence'] = torch.tensor(data.adjacency_dict['user_item_num'])
            graph.edges['converted-by'].data['occurrence'] = torch.tensor(data.adjacency_dict['user_item_num'])

    return graph


def calculate_num_batches(train_eid_dict,
                          valid_eid_dict,
                          sub_train_uid,
                          valid_uid,
                          test_uid,
                          all_iid,
                          edge_bs,
                          node_bs):
    train_eid_len = 0
    valid_eid_len = 0
    for e_type in train_eid_dict.keys():
        train_eid_len += len(train_eid_dict[e_type])
        valid_eid_len += len(valid_eid_dict[e_type])
    num_batches_train = math.ceil(train_eid_len / edge_bs)
    num_batches_sub_train = math.ceil(
        (len(sub_train_uid) + len(all_iid)) / node_bs
    )
    num_batches_val_loss = math.ceil(valid_eid_len / edge_bs)
    num_batches_val_metrics = math.ceil(
        (len(valid_uid) + len(all_iid)) / node_bs
    )
    num_batches_test = math.ceil(
        (len(test_uid) + len(all_iid)) / node_bs
    )
    return num_batches_train, num_batches_sub_train, num_batches_test, num_batches_val_loss, num_batches_val_metrics


def summary_data_sets(train_eid_dict, valid_eid_dict, sub_train_uid, valid_uid, test_uid,
                      all_iid, ground_truth_sub_train, ground_truth_valid, all_eid_dict):
    print("train_eid.converts =", len(train_eid_dict[('user', 'converts', 'item')]))
    print("train_eid.clicks   =", len(train_eid_dict[('user', 'clicks', 'item')]))
    print("valid_eid.converts =", len(valid_eid_dict[('user', 'converts', 'item')]))
    print("valid_eid.clicks   =", len(valid_eid_dict[('user', 'clicks', 'item')]))
    print("all_eid.converts   =", len(all_eid_dict[('user', 'converts', 'item')]))
    print("all_eid.clicks     =", len(all_eid_dict[('user', 'clicks', 'item')]))
    print("sub_train_uid      =", len(sub_train_uid))
    print("valid_uid          =", len(valid_uid))
    print("test_uid           =", len(test_uid))
    print("all_iid            =", len(all_iid))
    print("ground_truth_sub_train.uid =", len(ground_truth_sub_train[0]))
    print("ground_truth_sub_train.iid =", len(ground_truth_sub_train[1]))
    print("ground_truth_valid.uid     =", len(ground_truth_valid[0]))
    print("ground_truth_valid.iid     =", len(ground_truth_valid[1]))