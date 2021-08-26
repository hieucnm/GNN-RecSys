from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def create_ground_truth(users, items):
    """
    Creates a dictionary, where the keys are user ids and the values are item ids that the user actually bought.
    """
    ground_truth_arr = np.stack((np.asarray(users), np.asarray(items)), axis=1)
    ground_truth_dict = defaultdict(list)
    for key, val in ground_truth_arr:
        ground_truth_dict[key].append(val)
    return ground_truth_dict


def get_recs(embed_dict,
             model,
             k,
             unique_user_nodes,
             user_id: str,
             item_id: str,
             ):
    """
    Computes K recommendation for all users, given hidden states, the model and what they already bought.
    """
    recs = {}
    similarities = {}
    for user_node in unique_user_nodes:
        num_items = embed_dict[item_id].shape[0]
        user_emb = embed_dict[user_id][user_node].repeat(num_items, 1)
        ratings = model.get_ratings(user_emb, embed_dict[item_id]) \
            .cpu().detach().numpy().reshape(num_items)

        order = np.argsort(-ratings)
        rec = order[:k]
        recs[user_node] = rec
        similarities[user_node] = ratings
    return recs, similarities


def recs_to_metrics(recs, similarities, ground_truth_dict, num_unique_items):
    """
    Given the recommendations and the ground_truth, computes precision, recall & coverage.
    """
    # precision
    k_relevant = 0
    k_total = 0
    for user, items in recs.items():
        k_total += len(items)
        k_relevant += len([id_ for id_ in items if id_ in ground_truth_dict[user]])
    precision = k_relevant / k_total

    # recall
    k_relevant = 0
    k_total = 0
    for user, items in recs.items():
        k_total += len(ground_truth_dict[user])
        k_relevant += len([id_ for id_ in ground_truth_dict[user] if id_ in items])
    recall = k_relevant / k_total

    # coverage
    recs_flatten = [item for sublist in list(recs.values()) for item in sublist]
    num_recommended = len(set(recs_flatten))
    coverage = num_recommended / num_unique_items

    # auc
    y_true = []
    y_score = []
    for user, scores in similarities.items():
        # scores = (scores + 1) / 2  # convert range (-1,1) to (0,1) if necessary
        y_score.extend(scores)
        y_true.extend([int(iid in ground_truth_dict[user]) for iid in range(num_unique_items)])
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return precision, recall, coverage, auc


# noinspection PyTypeChecker
def get_metrics_at_k(embed_dict,
                     ground_truth,
                     model,
                     num_unique_items,
                     k: int,
                     user_id: str,
                     item_id: str):
    """
    Function combining all previous functions: create already_bought & ground_truth dict, get recs and compute metrics.
    """
    user_nodes, item_nodes = ground_truth
    unique_user_nodes = np.unique(user_nodes).tolist()
    ground_truth_dict = create_ground_truth(user_nodes, item_nodes)
    recs, similarities = get_recs(embed_dict, model, k, unique_user_nodes, user_id, item_id)
    precision, recall, coverage, auc = recs_to_metrics(recs, similarities, ground_truth_dict, num_unique_items)
    return precision, recall, coverage, auc
