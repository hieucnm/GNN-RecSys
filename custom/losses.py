import torch
import torch.nn as nn


class MaxMarginLoss(nn.Module):
    """
        Simple max margin loss.

        Parameters
        ----------
        pos_score:
            All similarity scores for positive examples.
        neg_score:
            All similarity scores for negative examples.
        delta:
            Delta from which the pos_score should be higher than all its corresponding neg_score.
        recency_scores:
            If not None, loss will be divided by the recency, i.e. more recent positive examples will be given a
            greater weight in the total loss. Those are the recency, for all training edges.
        """

    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta
        self.relu = nn.ReLU()

    def forward(self, pos_score, neg_score):
        device = pos_score[list(pos_score.keys())[0]].device
        all_scores = torch.empty(0).to(device)

        for edge_type in pos_score.keys():
            pos_score_tensor = pos_score[edge_type]
            neg_score_tensor = neg_score[edge_type]
            if pos_score_tensor.shape[0] == 0:
                # we don't train this edge
                continue

            neg_score_tensor = neg_score_tensor.reshape(pos_score_tensor.shape[0], -1)
            scores = neg_score_tensor + self.delta - pos_score_tensor
            scores = self.relu(scores)
            all_scores = torch.cat((all_scores, scores), 0)

        # print("scores: max = {:.3f} | min = {:.3f} | len = {}".format(
        #     torch.max(all_scores).item(), torch.min(all_scores).item(), all_scores.shape[0]
        # ))
        return torch.mean(all_scores)


class BCELossCustom(nn.Module):
    """
        See MaxMarginLoss for detail.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pos_score, neg_score):
        device = pos_score[list(pos_score.keys())[0]].device
        all_scores = torch.empty(0).to(device)
        all_labels = torch.empty(0).to(device)
        for e_type in pos_score.keys():
            pos_score_tensor = pos_score[e_type].flatten()
            neg_score_tensor = neg_score[e_type].flatten()
            if pos_score_tensor.shape[0] == 0:
                # we don't train this edge
                continue

            all_scores = torch.cat((all_scores, pos_score_tensor, neg_score_tensor), dim=0)
            all_labels = torch.cat((all_labels,
                                    torch.ones_like(pos_score_tensor),
                                    torch.zeros_like(neg_score_tensor)), dim=0)
        # print("scores: max = {:.3f} | min = {:.3f} | len = {}".format(
        #     torch.max(all_scores).item(), torch.min(all_scores).item(), all_scores.shape[0]
        # ))
        return self.bce(all_scores, all_labels)
