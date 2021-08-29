from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.utils_data import mkdir_if_missing


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
                 user_id,
                 item_id,
                 k: int = 5,
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
        # recs = recs[:, :self.k]
        return similarities, recs

    def evaluate_on_batches(self,
                            graph,
                            node_loader,
                            ground_truth
                            ):
        item_emb = self._get_all_item_embeddings(graph, node_loader)
        ground_truth_dict = create_ground_truth_dict(ground_truth)

        all_scores = []
        all_labels = []
        num_gt = 0
        num_gt_in_rec = 0
        rec_item_set = set()

        for i, (_, output_nodes, blocks) in enumerate(node_loader):

            if (i + 1) % self.print_every == 0:
                print("Batch {}/{}".format(i + 1, len(node_loader)))

            if self.user_id not in output_nodes:
                continue
            embed_dict = self._forward(blocks)

            user_emb = embed_dict[self.user_id]
            similarities, top_recommends = self._get_top_k_recommends(user_emb, item_emb)

            for user_node, sim, rec in zip(output_nodes[self.user_id].tolist(),
                                           similarities,
                                           top_recommends
                                           ):

                # for users having no label, only compute auc
                if user_node not in ground_truth_dict:
                    all_scores += sim.tolist()
                    all_labels += [0] * len(sim)
                    continue

                # to compute auc
                all_scores += sim.tolist()
                all_labels += [int(iid in ground_truth_dict[user_node]) for iid, _ in enumerate(sim)]

                # to compute accuracy (accuracy at 1, very strictly)
                rec = rec[:len(ground_truth_dict[user_node])]
                overlap = set(rec).intersection(ground_truth_dict[user_node])
                num_gt_in_rec += len(overlap)
                num_gt += len(ground_truth_dict[user_node])

                # to compute coverage
                rec_item_set.update(rec)

        auc = roc_auc_score(y_true=all_labels, y_score=all_scores)
        acc = num_gt_in_rec / num_gt
        coverage = len(rec_item_set) / graph.num_nodes(self.item_id)
        return acc, auc, coverage


class Predictor(Evaluator):
    def __init__(self,
                 model,
                 user_id,
                 item_id,
                 save_dir,
                 node2uid: dict,
                 node2iid: dict,
                 print_every: int = 1
                 ):
        super(Predictor, self).__init__(model=model,
                                        user_id=user_id,
                                        item_id=item_id,
                                        print_every=print_every
                                        )
        self.node2uid = node2uid
        self.iid_columns = [f'cate_{v}' for k, v in sorted(node2iid.items())]
        self.emb_columns = [f'_c{i}' for i in range(self.embed_dim)]

        self.save_dir = save_dir
        self.user_embed_dir = f'{save_dir}/user_embeddings'
        self.score_dir = f'{save_dir}/scores'
        mkdir_if_missing(self.user_embed_dir, _type='dir')
        mkdir_if_missing(self.score_dir, _type='dir')

    def _save_scores(self, scores, output_nodes, index):
        score_df = pd.DataFrame(data=scores, columns=self.iid_columns)
        score_df[self.user_id] = [self.node2uid[nid] for nid in output_nodes[self.user_id]]
        score_df = score_df[[self.user_id] + self.iid_columns]
        score_df.to_parquet(f'{self.score_dir}/part_{index:05d}.parquet')

    def _save_user_embeds(self, embeds, index):
        embed_df = pd.DataFrame(data=embeds, columns=self.emb_columns)
        embed_df.to_parquet(f'{self.user_embed_dir}/part_{index:05d}.parquet')

    def _save_item_embeds(self, embeds):
        embed_df = pd.DataFrame(data=embeds, columns=self.emb_columns)
        embed_df.to_parquet(f'{self.save_dir}/item_embeddings.parquet')

    def predict_and_save_on_batches(self, graph, node_loader):
        item_emb = self._get_all_item_embeddings(graph, node_loader)
        self._save_item_embeds(item_emb)

        for i, (_, output_nodes, blocks) in enumerate(node_loader):

            if self.user_id not in output_nodes:
                continue
            embed_dict = self._forward(blocks)

            user_emb = embed_dict[self.user_id]
            self._save_user_embeds(user_emb, i)
            scores, _ = self._get_top_k_recommends(user_emb, item_emb)
            self._save_scores(scores, output_nodes, i)

            if (i + 1) % self.print_every == 0:
                print("Batch {}/{}".format(i + 1, len(node_loader)))
