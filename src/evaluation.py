from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def repeat_tensors(x, y):
    x_len, y_len = x.shape[0], y.shape[0]
    x = x.repeat(1, y_len).reshape(x_len * y_len, -1)
    y = y.repeat(x_len, 1)
    return x, y


class BaseEvaluator:
    def __init__(self,
                 model,
                 user_id,
                 item_id,
                 print_every: int = 1
                 ):
        self.model = model
        self.user_id = user_id
        self.item_id = item_id
        self.print_every = print_every
        self.embed_dim = model.embed_dim
        self.device = next(model.parameters()).device

    def _forward(self, blocks):
        blocks = [b.to(self.device) for b in blocks]
        input_features = blocks[0].srcdata['features']
        with torch.no_grad():
            embed_dict = self.model.get_repr(blocks, input_features)
        return embed_dict

    def get_all_item_embeddings(self, graph, item_node_loader):
        num_unique_items = graph.num_nodes(self.item_id)
        item_embed = torch.zeros(num_unique_items, self.embed_dim).to(self.device)
        for i, (_, output_nodes, blocks) in enumerate(item_node_loader):
            embed_dict = self._forward(blocks)
            item_embed[output_nodes[self.item_id]] = embed_dict[self.item_id]
        return item_embed

    def _get_similarities(self,
                          user_emb,
                          item_emb,
                          k=5,
                          ):

        # first, repeat tensors to pass into prediction layer correctly
        user_emb_rpt, item_emb_rpt = repeat_tensors(user_emb, item_emb)

        with torch.no_grad():
            similarities = self.model.get_ratings(user_emb_rpt, item_emb_rpt) \
                .cpu().detach().numpy().reshape(-1, item_emb.shape[0])
        recs = np.argsort(-similarities)
        # TODO: for now, we only use top-1 accuracy, not use k
        # recs = recs[:, :self.k]
        return similarities, recs

    def evaluate(self, graph, node_loader, item_node_loader, ground_truth, **params):
        raise NotImplementedError

    def _create_ground_truth_dict(self, ground_truth_dict):
        raise NotImplementedError


class LinkBasedEvaluator(BaseEvaluator):
    def __init__(self, model, user_id, item_id, print_every=1):
        super(LinkBasedEvaluator, self).__init__(model, user_id, item_id, print_every)

    def _create_ground_truth_dict(self, ground_truth_dict):
        res = defaultdict(list)
        for ground_truth_pairs in ground_truth_dict.values():
            for user_node, item_node in ground_truth_pairs:
                res[user_node].append(item_node)
        return res

    def evaluate(self, graph, node_loader, item_node_loader, ground_truth, **params):
        ground_truth_dict = self._create_ground_truth_dict(ground_truth)
        item_emb = self.get_all_item_embeddings(graph, item_node_loader)

        all_scores = []
        all_labels = []
        num_gt = 0
        num_gt_in_rec = 0
        rec_item_set = set()

        for i, (_, output_nodes, blocks) in enumerate(node_loader):
            if (i + 1) % self.print_every == 0:
                print("Batch {}/{}".format(i + 1, len(node_loader)))

            if output_nodes[self.user_id].nelement() == 0:
                continue
            embed_dict = self._forward(blocks)
            user_emb = embed_dict[self.user_id]

            similarities, top_recommends = self._get_similarities(user_emb, item_emb)
            for user_node, sim, rec in zip(output_nodes[self.user_id].tolist(),
                                           similarities,
                                           top_recommends
                                           ):
                # to compute auc
                all_scores += sim.tolist()
                all_labels += [int(iid in ground_truth_dict[user_node]) for iid, _ in enumerate(sim)]

                # to compute accuracy
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


class LabelBasedEvaluator(BaseEvaluator):
    def __init__(self, model, user_id, item_id, pos_etype, neg_etype, print_every=1):
        # TODO: for now, `pos_etype` and `neg_etype`  only have 1 element
        self.pos_etype = pos_etype[0]
        self.neg_etype = neg_etype[0]
        super(LabelBasedEvaluator, self).__init__(model, user_id, item_id, print_every)

    def _create_ground_truth_dict(self, ground_truth_dict):
        pos_gt_dict = defaultdict(list)
        for user_node, item_node in ground_truth_dict[self.pos_etype]:
            pos_gt_dict[user_node].append(item_node)
        neg_gt_dict = defaultdict(list)
        for user_node, item_node in ground_truth_dict[self.neg_etype]:
            neg_gt_dict[user_node].append(item_node)
        return pos_gt_dict, neg_gt_dict

    def _verify_grounthuth_exist(self, ground_truth):
        assert self.pos_etype in ground_truth, f"`{self.pos_etype}` not in groundtruth: {ground_truth.keys()}"
        assert self.neg_etype in ground_truth, f"`{self.neg_etype}` not in groundtruth: {ground_truth.keys()}"

    def evaluate(self, graph, node_loader, item_node_loader, ground_truth, **params):
        self._verify_grounthuth_exist(ground_truth)
        pos_gt_dict, neg_gt_dict = self._create_ground_truth_dict(ground_truth)
        item_emb = self.get_all_item_embeddings(graph, item_node_loader)

        score_dict = defaultdict(list)
        label_dict = defaultdict(list)

        for i, (_, output_nodes, blocks) in enumerate(node_loader):
            if (i + 1) % self.print_every == 0:
                print("Batch {}/{}".format(i + 1, len(node_loader)))

            if output_nodes[self.user_id].nelement() == 0:
                continue
            embed_dict = self._forward(blocks)
            if self.user_id not in embed_dict:
                continue
            user_emb = embed_dict[self.user_id]

            similarities, _ = self._get_similarities(user_emb, item_emb)
            for user_node, sim in zip(output_nodes[self.user_id].tolist(), similarities):

                if user_node in pos_gt_dict:
                    for item_node in pos_gt_dict[user_node]:
                        score_dict[item_node].append(sim[item_node])
                        label_dict[item_node].append(1)

                if user_node in neg_gt_dict:
                    for item_node in neg_gt_dict[user_node]:
                        score_dict[item_node].append(sim[item_node])
                        label_dict[item_node].append(0)
        auc_dict = {
            item_node: roc_auc_score(y_true=label_dict[item_node], y_score=score_dict[item_node])
            for item_node in score_dict
        }
        return auc_dict


# =============
# Predictor ===

class Predictor(BaseEvaluator):
    def __init__(self,
                 model,
                 item_emb,
                 iid_columns,
                 user_id,
                 item_id,
                 print_every: int = 1
                 ):
        super(Predictor, self).__init__(model=model,
                                        user_id=user_id,
                                        item_id=item_id,
                                        print_every=print_every
                                        )
        self.item_emb = item_emb
        if not self.item_emb.is_cuda:
            self.item_emb = self.item_emb.to(self.device)
        self._verify_num_items()
        self.iid_columns = iid_columns

    # Just implement the abstract method
    def _create_ground_truth_dict(self, ground_truth_dict):
        pass

    # Just implement the abstract method
    def evaluate(self, graph, node_loader, item_node_loader, ground_truth, **params):
        pass

    def _verify_num_items(self):
        assert len(self.iid_columns) == self.item_emb.shape[0], \
            "no. of items in given dataset ({}) not equal no. of items in pre-calculated item embeddings ({})" \
            .format(len(self.iid_columns), self.item_emb.shape[0])

    def _create_score_df(self,
                         scores,
                         output_nodes,
                         node2uid
                         ):
        score_df = pd.DataFrame(data=scores, columns=self.iid_columns)
        score_df[self.user_id] = [node2uid[nid] for nid in output_nodes[self.user_id].tolist()]
        score_df = score_df[[self.user_id] + self.iid_columns]
        return score_df

    def _create_user_embed_df(self,
                              embeds,
                              output_nodes,
                              node2uid
                              ):
        embed_df = pd.DataFrame()
        embed_df[self.user_id] = [node2uid[nid] for nid in output_nodes[self.user_id].tolist()]
        embed_df['embeddings'] = embeds.detach().cpu().tolist()
        return embed_df

    def predict(self,
                node_loader,
                node2uid: dict,
                ):

        user_emb_df_list = []
        score_df_list = []

        for i, (_, output_nodes, blocks) in enumerate(node_loader):

            if output_nodes[self.user_id].nelement() == 0:
                continue
            embed_dict = self._forward(blocks)
            if self.user_id not in embed_dict:
                continue

            user_emb = embed_dict[self.user_id]
            user_emb_df = self._create_user_embed_df(user_emb, output_nodes, node2uid)
            user_emb_df_list.append(user_emb_df)

            scores, _ = self._get_similarities(user_emb, self.item_emb)
            score_df = self._create_score_df(scores, output_nodes, node2uid=node2uid)
            score_df_list.append(score_df)

            if (i + 1) % self.print_every == 0:
                print("Batch {}/{}".format(i + 1, len(node_loader)))

        user_emb_df = pd.concat(user_emb_df_list).reset_index(drop=True)
        score_df = pd.concat(score_df_list).reset_index(drop=True)
        return user_emb_df, score_df
