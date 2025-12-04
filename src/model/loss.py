import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = torch.norm(anchor - positive, p=2, dim=1)
        negative_distance = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.clamp(positive_distance - negative_distance + self.margin, min=0.0)
        return loss.mean()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Loss with Batch Hard strategy.
    For each anchor, selects the hardest positive (farthest same-class sample) 
    and hardest negative (closest different-class sample).
    """
    def __init__(self, margin=0.3):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,)
        """
        # Pairwise distances
        # dist[i, j] = ||emb[i] - emb[j]||
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(1) - 2 * dot_product + square_norm.unsqueeze(0)
        distances = torch.clamp(distances, min=0.0)
        distances = torch.sqrt(distances + 1e-12) # Add epsilon to avoid NaN gradient at 0

        # Mask for valid triplets
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels).float()
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels).float()

        # Hardest positive
        # We want max distance among positives.
        # Filter out non-positives by multiplying by mask.
        anchor_positive_dist = distances * mask_anchor_positive
        hardest_positive_dist = anchor_positive_dist.max(1)[0]

        # Hardest negative
        # We want min distance among negatives.
        # Add max_dist to non-negatives so they are not selected as min.
        max_dist = distances.max().detach()
        anchor_negative_dist = distances + max_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist = anchor_negative_dist.min(1)[0]

        # Loss
        triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0)
        loss = triplet_loss.mean()

        return loss

    def _get_anchor_positive_triplet_mask(self, labels):
        """
        Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        """
        indices_equal = torch.eye(labels.size(0)).bool().to(labels.device)
        indices_not_equal = ~indices_equal

        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        return labels_equal & indices_not_equal

    def _get_anchor_negative_triplet_mask(self, labels):
        """
        Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        """
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        return ~labels_equal

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss
