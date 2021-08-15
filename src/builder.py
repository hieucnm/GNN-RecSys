from datetime import datetime, timedelta
from typing import Tuple

import dgl
import numpy as np
import pandas as pd
import torch

from src.utils import read_data


def filter_unseen_item(train_path: str,
                       test_path: str,
                       item_id_column: str = "item_id"):
    train_data = read_data(train_path)
    test_data = read_data(test_path)
    train_items = train_data[item_id_column].unique()
    test_items = test_data[item_id_column].unique()
    diff_items = set(test_items).difference(train_items)
    print("--- Check to filter unseen items in test data ---")
    if len(diff_items) > 0:
        print(f"There are {len(diff_items)} items exist in test data but not exist in training data: "
              f"{diff_items}"
              f", interactions between them and any users are now removed from test data to prevent error.")
        test_data = test_data[~test_data[item_id_column].isin(diff_items)].reset_index(drop=True)
    else:
        print("All items in test data already exist in training data")
    return test_data


def report_user_coverage(train_path: str,
                         test_path: str,
                         user_id_column: str = "user_id"):
    train_data = read_data(train_path)
    test_data = read_data(test_path)
    train_users = train_data[user_id_column].unique()
    test_users = test_data[user_id_column].unique()
    overlap_users = set(test_users).intersection(train_users)
    print('--- Report user coverage ---')
    print("# num train users =", len(train_users))
    print("# num test users  =", len(test_users))
    print("# num test users existing in training data  =", len(overlap_users))


def create_ids(user_item_train: pd.DataFrame,
               user_sport_interaction: pd.DataFrame,
               sport_sportg_interaction: pd.DataFrame,
               item_feat_df,
               item_id_type: str = 'SPECIFIC ITEM IDENTIFIER',
               ctm_id_type: str = 'CUSTOMER IDENTIFIER',
               spt_id_type: str = 'sport_id',
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create ids needed for creating the graph (nodes cannot have arbitrary ids, i.e. it couldn't be directly
    the item identifier).

    Parameters
    ----------
    See parameters and outputs of format_dfs for details.

    Returns
    -------
    ctm_id, pdt_id, spt_id:
        Mapping between Organisation info (e.g. customer, item and sport ID) and new node ID.

    """

    # Create user ids
    ctm_id = pd.DataFrame(user_item_train[ctm_id_type].unique(),
                          columns=[ctm_id_type])
    ctm_id['ctm_new_id'] = ctm_id.index

    # Create item ids
    train_pdt = user_item_train[item_id_type].unique().tolist()
    all_pdt = item_feat_df[item_id_type].unique().tolist()
    unseen_pdt = [pdt for pdt in all_pdt if pdt not in train_pdt]
    train_pdt.extend(unseen_pdt)  # DGL requires that node IDs are continuous; unseen are at the end
    pdt_id = pd.DataFrame(train_pdt,
                          columns=[item_id_type])
    pdt_id['pdt_new_id'] = pdt_id.index

    # Create sport ids
    unique_sports = np.append(sport_sportg_interaction.sports_id.unique(),
                              sport_sportg_interaction.sportsgroup_id.unique())
    unique_sports = np.unique(np.append(unique_sports,
                                        user_sport_interaction[spt_id_type].unique()))
    spt_id = pd.DataFrame(unique_sports, columns=[spt_id_type])
    spt_id['spt_new_id'] = spt_id.index

    return ctm_id, pdt_id, spt_id


def df_to_adjacency_list(user_item_train: pd.DataFrame,
                         user_item_test: pd.DataFrame,
                         item_sport_interaction: pd.DataFrame,
                         user_sport_interaction: pd.DataFrame,
                         sport_sportg_interaction: pd.DataFrame,
                         ctm_id: pd.DataFrame,
                         pdt_id: pd.DataFrame,
                         spt_id: pd.DataFrame,
                         item_id_type: str,
                         ctm_id_type: str,
                         spt_id_type: str,
                         discern_clicks: bool = False,
                         duplicates: str = 'keep_all'
                         ):
    """
    Takes dataframes & ids for the nodes, and return adjacency lists (in the form of src nodes and dst nodes.)

    Parameters
    ----------
    discern_clicks, duplicates:
        See utils_data for details.
    all other parameters:
        See parameters & outputs of other functions in this file for details.

    Returns
    -------
    adjacency_dict:
        This will be used to build the graph. It contains id of source and destination nodes for all edge types.
    ground_truth_test, ground_truth_purchase_test:
        This will be used to compute metrics (i.e. check if recommended items can be found in the ground_truth). It
        contains user and item ids for all interactions in the test set.
    user_item_train:
        In this function, if duplicates == 'count_occurrence' or 'keep_last', some grouping manipulations are done on
        the user_item_train dataframe. Returning it will allow to attribute features to "grouped" edges.

    """
    adjacency_dict = {}
    # User item : join new ids with old ids
    user_item_train = user_item_train.merge(ctm_id,
                                            how='left',
                                            on=ctm_id_type)
    user_item_train = user_item_train.merge(pdt_id,
                                            how='left',
                                            on=item_id_type)

    if duplicates in ['keep_last', 'count_occurrence']:
        grouped_df = user_item_train.groupby(['buy', 'ctm_new_id', 'pdt_new_id']).specific_item_identifier.count()
        grouped_df = pd.DataFrame(grouped_df).reset_index()
        grouped_df.columns = ['buy', 'ctm_new_id', 'pdt_new_id', 'num_interaction']

        user_item_train.drop_duplicates(subset=['buy', 'ctm_new_id', 'pdt_new_id'],
                                        keep='last',
                                        inplace=True)  # Keep last interaction
        user_item_train.sort_values(by=['buy', 'ctm_new_id', 'pdt_new_id'],
                                    ignore_index=True,
                                    inplace=True)  # Have same order as grouped_df
        assert len(user_item_train) == len(grouped_df)
        user_item_train['num_interaction'] = grouped_df.num_interaction.values
        user_item_train.sort_values(by='hit_timestamp',
                                    ignore_index=True,
                                    inplace=True)  # Reorder by date to keep sequential order
        if discern_clicks:
            adjacency_dict.update(
                {
                    'clicks_num': user_item_train[user_item_train.buy == 0].num_interaction.values,
                    'purchases_num': user_item_train[user_item_train.buy == 1].num_interaction.values
                }
            )
        else:
            adjacency_dict.update(
                {
                    'user_item_num': user_item_train.num_interaction.values
                }
            )

    if discern_clicks:
        adjacency_dict.update(
            {
                'clicks_src': user_item_train[user_item_train.buy == 0].ctm_new_id.values,
                'clicks_dst': user_item_train[user_item_train.buy == 0].pdt_new_id.values,
                'purchases_src': user_item_train[user_item_train.buy == 1].ctm_new_id.values,
                'purchases_dst': user_item_train[user_item_train.buy == 1].pdt_new_id.values,
            }
        )

    else:
        adjacency_dict.update(
            {
                'user_item_src': user_item_train.ctm_new_id.values,
                'user_item_dst': user_item_train.pdt_new_id.values,
            }
        )

    user_item_test = user_item_test.merge(ctm_id,
                                          how='left',
                                          on=ctm_id_type)
    user_item_test = user_item_test.merge(pdt_id,
                                          how='left',
                                          on=item_id_type)
    test_purchase_src = user_item_test[user_item_test.buy == 1].ctm_new_id.values
    test_purchase_dst = user_item_test[user_item_test.buy == 1].pdt_new_id.values
    ground_truth_purchase_test = (test_purchase_src, test_purchase_dst)

    test_src = user_item_test.ctm_new_id.values
    test_dst = user_item_test.pdt_new_id.values
    ground_truth_test = (test_src, test_dst)

    # Item sport : merge new ids with old ids
    item_sport_interaction = item_sport_interaction.merge(spt_id,
                                                          how='left',
                                                          on=spt_id_type)
    item_sport_interaction = item_sport_interaction.merge(pdt_id,
                                                          how='left',
                                                          on=item_id_type)
    item_sport_interaction.dropna(inplace=True)  # drop items with no sports associated

    adjacency_dict['item_sport_src'] = item_sport_interaction.pdt_new_id.values
    adjacency_dict['item_sport_dst'] = item_sport_interaction.spt_new_id.values

    # User sport : merge new ids with old ids
    user_sport_interaction = user_sport_interaction.merge(spt_id,
                                                          how='left',
                                                          on=spt_id_type)
    user_sport_interaction = user_sport_interaction.merge(ctm_id,
                                                          how='left',
                                                          on=ctm_id_type)
    user_sport_interaction.dropna(inplace=True)

    adjacency_dict['user_sport_src'] = user_sport_interaction.ctm_new_id.values
    adjacency_dict['user_sport_dst'] = user_sport_interaction.spt_new_id.values

    # Sport sportgroups
    sport_sportg_interaction = sport_sportg_interaction.merge(spt_id,
                                                              how='left',
                                                              left_on='sports_id',
                                                              right_on=spt_id_type)
    sport_sportg_interaction = sport_sportg_interaction.merge(spt_id,
                                                              how='left',
                                                              left_on='sportsgroup_id',
                                                              right_on=spt_id_type)

    adjacency_dict['sport_sportg_src'] = sport_sportg_interaction.spt_new_id_x.values
    adjacency_dict['sport_sportg_dst'] = sport_sportg_interaction.spt_new_id_y.values

    return adjacency_dict, ground_truth_test, ground_truth_purchase_test, user_item_train


def create_graph(graph_schema,
                 ) -> dgl.DGLHeteroGraph:
    """
    Create graph based on adjacency list.
    """
    g = dgl.heterograph(graph_schema)
    return g


def import_features(g: dgl.DGLHeteroGraph,
                    user_feat_df,
                    item_feat_df,
                    sport_onehot_df,
                    ctm_id: pd.DataFrame,
                    pdt_id: pd.DataFrame,
                    spt_id: pd.DataFrame,
                    user_item_train,
                    get_popularity: bool,
                    num_days_pop: int,
                    item_id_type: str,
                    ctm_id_type: str,
                    spt_id_type: str,
                    ):
    """
    Import features to a dict for all node types.

    For user and item, initializes feature arrays with only 0, then fills the values if they are available.

    Parameters
    ----------
    get_popularity, num_days_pop:
        The recommender system can be enhanced by giving score boost for items that were popular. If get_popularity,
        popularity of the items will be computed. Num_days_pop defines the number of days to include in the
        computation.
    item_id_type, ctm_id_type, spt_id_type:
        See utils_data for details.
    all other parameters:
        See other functions in this file for details.

    Returns
    -------
    features_dict:
        Dictionary with all the features imported here.
    """
    features_dict = {}
    # User
    user_feat_df = user_feat_df.merge(ctm_id, how='inner', on=ctm_id_type)

    ids = user_feat_df.ctm_new_id.values.astype(int)
    feats = np.stack((user_feat_df.is_male.values,
                      user_feat_df.is_female.values),
                     axis=1)

    user_feat = np.zeros((g.number_of_nodes('user'), 2))
    user_feat[ids] = feats

    user_feat = torch.tensor(user_feat).float()
    features_dict['user_feat'] = user_feat

    # Item
    if item_id_type in ['SPECIFIC ITEM IDENTIFIER']:
        item_feat_df = item_feat_df.merge(pdt_id,
                                          how='left',
                                          on=item_id_type)
        item_feat_df = item_feat_df[item_feat_df.pdt_new_id < g.number_of_nodes('item')]  # Only IDs that are in graph

        ids = item_feat_df.pdt_new_id.values.astype(int)
        feats = np.stack((item_feat_df.is_junior.values,
                          item_feat_df.is_male.values,
                          item_feat_df.is_female.values,
                          item_feat_df.eco_design.values,
                          ),
                         axis=1)

        item_feat = np.zeros((g.number_of_nodes('item'), feats.shape[1]))
        item_feat[ids] = feats
        item_feat = torch.tensor(item_feat).float()
    elif item_id_type in ['GENERAL ITEM IDENTIFIER']:
        item_feat = torch.zeros((g.number_of_nodes('item'), 4))
    else:
        raise KeyError(f'Item ID {item_id_type} not recognized.')

    features_dict['item_feat'] = item_feat

    # Sport one-hot
    if 'sport' in g.ntypes:
        sport_onehot_df = sport_onehot_df.merge(spt_id, how='inner', on=spt_id_type)
        sport_onehot_df.sort_values(by='spt_new_id',
                                    inplace=True)  # Values need to be sorted by node id to align with g.nodes['sport']
        feats = sport_onehot_df.drop(labels=[spt_id_type, 'spt_new_id'], axis=1).values
        assert feats.shape[0] == g.num_nodes('sport')
        sport_feat = torch.tensor(feats).float()
        features_dict['sport_feat'] = sport_feat

    # Popularity
    if get_popularity:
        item_popularity = np.zeros((g.number_of_nodes('item'), 1))
        pop_df = user_item_train.merge(pdt_id,
                                       how='left',
                                       on=item_id_type)
        most_recent_date = datetime.strptime(max(pop_df.hit_date), '%Y-%m-%d')
        limit_date = datetime.strftime(
            (most_recent_date - timedelta(days=num_days_pop)),
            format='%Y-%m-%d'
        )
        pop_df = pop_df[pop_df.hit_date >= limit_date]
        pop_df = pd.DataFrame(pop_df.pdt_new_id.value_counts())
        pop_df.columns = ['purchases']
        pop_df['score'] = pop_df.purchases / pop_df.purchases.sum()
        pop_df.sort_index(inplace=True)
        ids = pop_df.index.values.astype(int)
        scores = pop_df.score.values
        item_popularity[ids] = np.expand_dims(scores, axis=1)
        item_popularity = torch.tensor(item_popularity).float()
        features_dict['item_pop'] = item_popularity

    return features_dict
