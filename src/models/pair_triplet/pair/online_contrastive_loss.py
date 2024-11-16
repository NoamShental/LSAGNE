import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings_t: Tensor, target_t: Tensor):
        positive_pairs_t, negative_pairs_t = self.pair_selector.get_pairs(embeddings_t, target_t)
        positive_loss = (embeddings_t[positive_pairs_t[:, 0]] - embeddings_t[positive_pairs_t[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings_t[negative_pairs_t[:, 0]] - embeddings_t[negative_pairs_t[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()

    # def forward(self, embeddings, target):
    #     positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
    #     if embeddings.is_cuda:
    #         positive_pairs = positive_pairs.cuda()
    #         negative_pairs = negative_pairs.cuda()
    #     positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
    #     negative_loss = F.relu(
    #         self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
    #             1).sqrt()).pow(2)
    #     loss = torch.cat([positive_loss, negative_loss], dim=0)
    #     return loss.mean()