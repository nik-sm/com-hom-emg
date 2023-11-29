import random
from itertools import product

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn


def get_masks(labels, is_real):
    directions_match = labels[:, 0].unsqueeze(0) == labels[:, 0].unsqueeze(1)
    modifiers_match = labels[:, 1].unsqueeze(0) == labels[:, 1].unsqueeze(1)
    labels_match = directions_match & modifiers_match

    is_same_realness = is_real.unsqueeze(0) == is_real.unsqueeze(1)

    # Given a real Up+Pinch, a positive item would be a fake Up+Pinch
    # (i.e. same label, opposite realness)
    # A negative would be a fake Down+Thumb
    # (i.e. different label, opposite realness)
    positive_mask = labels_match & ~is_same_realness
    negative_mask = ~labels_match & ~is_same_realness
    return positive_mask, negative_mask


class TripletLossHardMining(nn.Module):
    """The farthest positive and the closest negative item are used for a triplet"""

    # see:
    # https://omoindrot.github.io/triplet-loss
    # https://arxiv.org/abs/1703.07737
    # https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        real_double_features: torch.Tensor,
        fake_double_features: torch.Tensor,
        real_double_labels: torch.Tensor,
        fake_double_labels: torch.Tensor,
    ):
        assert len(real_double_features) == len(real_double_labels)
        assert len(fake_double_features) == len(fake_double_labels)
        assert len(real_double_features) > 0
        assert len(fake_double_features) > 0

        embeddings = torch.cat([real_double_features, fake_double_features], dim=0)
        labels = torch.cat([real_double_labels, fake_double_labels], dim=0)
        device = embeddings.device
        is_real = torch.cat(
            [
                torch.ones(len(real_double_labels), device=device),
                torch.zeros(len(fake_double_labels), device=device),
            ],
        )
        pairwise_dist = torch.cdist(embeddings, embeddings)

        # Masks: for each row, which items are valid as a positive or negative item
        # Positive items: same label, opposite realness
        # Negative items: diff label, opposite realness
        positive_mask, negative_mask = get_masks(labels, is_real)
        positive_mask = positive_mask.float()
        negative_mask = negative_mask.float()

        # Subset to rows with at least 1 positive and at least 1 negative so we can form a triplet
        subset_idx = (positive_mask.sum(1) > 0) & (negative_mask.sum(1) > 0)
        if subset_idx.sum() == 0:
            return torch.tensor(0.0).to(embeddings.device)
        pairwise_dist = pairwise_dist[subset_idx, :]
        positive_mask = positive_mask[subset_idx, :]
        negative_mask = negative_mask[subset_idx, :]

        # Use mask to zero out any distances where (a, p) not valid.
        # (a, p) is valid if label(a) == label(p) and is_real(a) != is_real(p)
        # Thus when we select the largest dist, we'll select a valid positive
        anchor_positive_dist = positive_mask * pairwise_dist
        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        # Thus when we select the minimum dist, we'll select a valid negative
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - negative_mask)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin, inplace=True).mean()
        return triplet_loss


class TripletLoss(nn.Module):
    """A random positive and a random negative item are used for a triplet"""

    def __init__(self, margin: float, triplets_per_item: int = 1):
        super().__init__()
        self.margin = margin
        self.triplets_per_item = triplets_per_item

    def forward(
        self,
        real_double_features: torch.Tensor,
        fake_double_features: torch.Tensor,
        real_double_labels: torch.Tensor,
        fake_double_labels: torch.Tensor,
    ):
        assert len(real_double_features) == len(real_double_labels)
        assert len(fake_double_features) == len(fake_double_labels)
        assert len(real_double_features) > 0
        assert len(fake_double_features) > 0

        embeddings = torch.cat([real_double_features, fake_double_features], dim=0)
        labels = torch.cat([real_double_labels, fake_double_labels], dim=0)
        device = embeddings.device
        is_real = torch.cat(
            [
                torch.ones(len(real_double_labels), device=device),
                torch.zeros(len(fake_double_labels), device=device),
            ],
        )
        pairwise_dist = torch.cdist(embeddings, embeddings)

        # Masks: for each row, which items are valid as a positive or negative item
        # Positive items: same label, opposite realness
        # Negative items: diff label, opposite realness
        positive_mask, negative_mask = get_masks(labels, is_real)
        positive_mask = positive_mask.int()
        negative_mask = negative_mask.int()

        # Subset to rows with at least K positive and at least K negative so we can form K triplets per row
        subset_idx = (positive_mask.sum(1) >= self.triplets_per_item) & (negative_mask.sum(1) >= self.triplets_per_item)
        if subset_idx.sum() == 0:
            logger.warning(f"Not enough triplets per item (wanted: {self.triplets_per_item})")
            return torch.tensor(0.0).to(embeddings.device)

        pairwise_dist = pairwise_dist[subset_idx, :]
        positive_mask = positive_mask[subset_idx, :]
        negative_mask = negative_mask[subset_idx, :]

        # The masks contain all "0" and "1" integers.
        # topk returns indices of first K "1" values in each row
        # Since batch contains shuffled items, the first K neighbors are random
        first_k_positive_idx = positive_mask.topk(self.triplets_per_item, dim=1, sorted=False).indices
        first_k_negative_idx = negative_mask.topk(self.triplets_per_item, dim=1, sorted=False).indices

        anchor_positive_dist = pairwise_dist.gather(1, first_k_positive_idx)
        anchor_negative_dist = pairwise_dist.gather(1, first_k_negative_idx)
        triplet_loss = F.relu(anchor_positive_dist - anchor_negative_dist + self.margin, inplace=True).mean()

        return triplet_loss


class TripletCentroids(nn.Module):
    """
    Randomly initialize a centroid for each class.
    For each item, form triplets by comparing it to class centroids
    (there is exactly one positive centroid, and one randomly chosen negative centroid)
    Update centroids gradually using momentum.
    """

    def __init__(self, margin, feature_dim: int, device: str, momentum=0.9):
        super().__init__()
        self.margin = margin
        self.momentum = momentum
        # TODO - init the centroids farther apart or closer together?
        #
        # https://math.stackexchange.com/questions/917292/expected-distance-between-two-vectors-that-belong-to-two-different-gaussian-dist  # noqa
        # Expected distance between two independent gaussian vectors of dimension D is:
        # E[ || x - y || ^ 2 ] = || mu_x - mu_y || ^ 2 +  tr(Cov_x + Cov_y)
        # torch.randn(n_items, n_features) * sigma has (approximately) mean = 0,
        # and spherical covariance = sigma**2 * torch.eye(n_features)
        # Expected distance between any pair of centroids will be:
        #
        # dist = 0 + trace(Cov_1 + Cov_2) = 2 * sigma**2 * n_features
        # dist = 0 + trace(2 * sigma**2 * n_features)
        # dist = 2 * sigma**2 * n_features
        self.keys = {torch.tensor([d, m], device=device, requires_grad=False) for (d, m) in product(range(4), range(4))}
        self.real_centroids = {k: torch.randn((feature_dim,), device=device, requires_grad=False) for k in self.keys}
        self.fake_centroids = {k: torch.randn((feature_dim,), device=device, requires_grad=False) for k in self.keys}

    def forward(
        self,
        real_double_features: torch.Tensor,
        fake_double_features: torch.Tensor,
        real_double_labels: torch.Tensor,
        fake_double_labels: torch.Tensor,
    ):
        assert len(real_double_features) == len(real_double_labels)
        assert len(fake_double_features) == len(fake_double_labels)
        assert len(real_double_features) > 0
        assert len(fake_double_features) > 0

        # Loop over real classes, computing triplet losses
        # In first iteration, anchor items all belong to c0.
        # Next iter, all anchors belong to c1, etc.
        # For each anchor item, just compute 1 triplet.
        anchors, positives, negatives = [], [], []
        for label in self.keys:
            anchor_idx = real_double_labels.eq(label).all(-1)
            _anchors = real_double_features[anchor_idx]
            if len(_anchors) == 0:
                continue

            # Use the matching centroid from fake items as positive
            positive_centroid = self.fake_centroids[label]
            # Sample 1 negative centroid (with replacement) for each anchor item
            negative_classes = list(self.keys - {label})
            negative_centroid_labels = random.choices(negative_classes, k=len(_anchors))
            for a, n in zip(_anchors, negative_centroid_labels):
                negative_centroid = self.fake_centroids[n]
                anchors.append(a)
                positives.append(positive_centroid)
                negatives.append(negative_centroid)

        # Loop over fake classes as anchors
        anchors, positives, negatives = [], [], []
        for label in self.keys:
            anchor_idx = fake_double_labels.eq(label).all(-1)
            _anchors = fake_double_features[anchor_idx]
            if len(_anchors) == 0:
                continue

            # Use the matching centroid from real items as positive
            positive_centroid = self.real_centroids[label]
            # Sample 1 negative centroid (with replacement) for each anchor item
            negative_classes = list(self.keys - {label})
            negative_centroid_labels = random.choices(negative_classes, k=len(_anchors))
            for a, n in zip(_anchors, negative_centroid_labels):
                negative_centroid = self.real_centroids[n]
                anchors.append(a)
                positives.append(positive_centroid)
                negatives.append(negative_centroid)

        if len(anchors) == 0:
            logger.warning("No triplets found")
            loss = torch.tensor(0.0)
        else:
            anchors = torch.stack(anchors)
            positives = torch.stack(positives)
            negatives = torch.stack(negatives)

            # Compute loss
            loss = F.triplet_margin_loss(anchors, positives, negatives, margin=self.margin)

        # Update centroids with momentum
        # (update after computing loss; same order as in SGD with momentum)
        with torch.no_grad():
            for label, prev in self.real_centroids.items():
                match_idx = real_double_labels.eq(label).all(-1)
                if match_idx.sum() == 0:
                    continue
                curr = real_double_features[match_idx].mean(0).detach()
                self.real_centroids[label] = self.momentum * prev + (1 - self.momentum) * curr

            for label, prev in self.fake_centroids.items():
                match_idx = fake_double_labels.eq(label).all(-1)
                if match_idx.sum() == 0:
                    continue
                curr = fake_double_features[match_idx].mean(0).detach()
                self.fake_centroids[label] = self.momentum * prev + (1 - self.momentum) * curr

        return loss
