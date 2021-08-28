import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from collections import defaultdict


def get_embeddings(model,
                   graph,
                   node_loader,
                   embed_dim: int,
                   print_every: int = 1
                   ):
    """
    Fetch the embeddings for all the nodes in the node_loader.

    Node Loader is preferable when computing embeddings because we can specify which nodes to compute the embedding for,
    and only have relevant nodes in the computational blocks. Whereas Edgeloader is preferable for training, because
    we generate negative edges also.
    """
    device = next(model.parameters()).device

    embed_dict = {node_type: torch.zeros(graph.num_nodes(node_type), embed_dim).to(device)
                  for node_type in graph.ntypes}

    # TODO: I still don't understand what is the difference between input_nodes and output_nodes
    for i, (input_nodes, output_nodes, blocks) in enumerate(node_loader):
        blocks = [b.to(device) for b in blocks]
        input_features = blocks[0].srcdata['features']
        h = model.get_repr(blocks, input_features)
        for node_type, embedding in h.items():
            embed_dict[node_type][output_nodes[node_type]] = embedding

        if (i + 1) % print_every == 0:
            print("Batch {}/{}".format(i, len(node_loader)))
    return embed_dict


def repeat_tensors(x, y):
    x_len, y_len = x.shape[0], y.shape[0]
    x = x.repeat(1, y_len).reshape(x_len * y_len, -1)
    y = y.repeat(x_len, 1)
    return x, y


def create_ground_truth_dict(ground_truth):
    """
    Creates a dictionary, where the keys are user ids and the values are item ids that the user actually bought.
    Parameters
    ----------
    ground_truth: tuple(list of user_nodes, list of item_nodes)

    Returns
    -------
    """
    ground_truth_dict = defaultdict(list)
    for user_node, item_node in ground_truth:
        ground_truth_dict[user_node].append(item_node)
    return ground_truth_dict


class Evaluator:
    def __init__(self,
                 model,
                 k,
                 user_id,
                 item_id,
                 print_every: int = 1
                 ):
        self.model = model
        self.user_id = user_id
        self.item_id = item_id
        self.k = k
        self.print_every = print_every

        self.embed_dim = model.embed_dim
        self.device = next(model.parameters()).device

    def _forward(self,
                 blocks
                 ):
        blocks = [b.to(self.device) for b in blocks]
        input_features = blocks[0].srcdata['features']
        with torch.no_grad():
            embed_dict = self.model.get_repr(blocks, input_features)
        return embed_dict

    def _get_all_item_embeddings(self,
                                 graph,
                                 node_loader,
                                 ):
        num_unique_items = graph.num_nodes(self.item_id)
        item_embeddings = torch.zeros(num_unique_items, self.embed_dim).to(self.device)
        for i, (_, output_nodes, blocks) in enumerate(node_loader):
            if output_nodes[self.item_id].nelement() > 0:
                embed_dict = self._forward(blocks)
                if self.item_id not in embed_dict:
                    continue
                item_emb = embed_dict[self.item_id]
                item_embeddings[output_nodes[self.item_id]] = item_emb
        return item_embeddings

    def _get_top_k_recommends(self,
                              user_emb,
                              item_emb
                              ):

        # first, repeat tensors to pass into prediction layer correctly
        user_emb_rpt, item_emb_rpt = repeat_tensors(user_emb, item_emb)

        with torch.no_grad():
            similarities = self.model.get_ratings(user_emb_rpt, item_emb_rpt) \
                .cpu().detach().numpy().reshape(-1, item_emb.shape[0])
        recs = np.argsort(-similarities)
        recs = recs[:, :self.k]
        return similarities, recs

    def evaluate_on_batches(self,
                            graph,
                            node_loader,
                            ground_truth
                            ):

        ground_truth_dict = create_ground_truth_dict(ground_truth)
        item_emb = self._get_all_item_embeddings(graph, node_loader)

        all_scores = []
        all_labels = []
        num_gt = 0
        num_rec = 0
        num_rec_in_gt = 0
        num_gt_in_rec = 0
        rec_item_set = set()
        user_set = set()

        for i, (input_nodes, output_nodes, blocks) in enumerate(node_loader):

            if (i + 1) % self.print_every == 0:
                print("Batch {}/{}".format(i + 1, len(node_loader)))

            embed_dict = self._forward(blocks)
            if self.user_id not in embed_dict:
                continue

            user_emb = embed_dict[self.user_id]
            similarities, top_recommends = self._get_top_k_recommends(user_emb, item_emb)

            for user_node, sim, rec in zip(output_nodes[self.user_id].tolist(),
                                           similarities,
                                           top_recommends
                                           ):

                user_set.add(user_node)

                # For now, just evaluate on users in ground truth
                if user_node not in ground_truth_dict:
                    continue

                # to compute auc
                all_scores += sim.tolist()
                all_labels += [int(iid in ground_truth_dict[user_node]) for iid, _ in enumerate(sim)]

                # to compute precision
                num_rec += len(rec)
                num_gt_in_rec += len([iid for iid in ground_truth_dict[user_node] if iid in rec])

                # to compute recall
                num_gt += len(ground_truth_dict[user_node])
                num_rec_in_gt += len([iid for iid in rec if iid in ground_truth_dict[user_node]])

                # to compute coverage
                rec_item_set.update(rec)

        auc = roc_auc_score(y_true=all_labels, y_score=all_scores)
        precision = num_gt_in_rec / num_rec
        recall = num_rec_in_gt / num_gt
        coverage = len(rec_item_set) / graph.num_nodes(self.item_id)
        return precision, recall, coverage, auc
