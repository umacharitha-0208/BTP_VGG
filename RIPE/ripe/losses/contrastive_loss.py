import torch
import torch.nn as nn
import torch.nn.functional as F


def second_nearest_neighbor(desc1, desc2):
    if desc2.shape[0] < 2:
        raise ValueError("desc2 should have at least 2 descriptors")
    dist = torch.cdist(desc1, desc2, p=2)
    vals, idxs = torch.topk(dist, 2, dim=1, largest=False)
    return vals[:, 1].view(-1, 1)  # second nearest distances


def contrastive_loss(
    desc1, desc2, matches, inliers, label,
    logits_1=None, logits_2=None,
    pos_margin=1.0, neg_margin=1.0,
):
    # Guard against empty matches — use self-similarity diversity fallback
    # instead of returning tensor(0.0) which has no grad_fn and causes
    # chaotic zero-spikes in descriptor loss.
    if len(matches) == 0 or len(inliers) == 0 or len(matches) != len(inliers):
        if desc1.shape[0] >= 2:
            # Diversity regularization: push random descriptor pairs apart
            n = min(desc1.shape[0], 64)
            idx = torch.randperm(desc1.shape[0], device=desc1.device)[:n]
            sampled = desc1[idx]
            dists = torch.cdist(sampled, sampled, p=2)
            # Penalize descriptors that are too close (off-diagonal only)
            mask = ~torch.eye(n, dtype=torch.bool, device=desc1.device)
            fallback_loss = torch.clamp(neg_margin - dists[mask], min=0.0).mean()
            return fallback_loss * 0.1  # Scale down so it doesn't dominate
        # Last resort: return a tensor that still has requires_grad=True
        return torch.tensor(0.0, device=desc1.device, requires_grad=True)

    if inliers.sum() < 8:
        inliers = torch.ones_like(inliers)

    matched_inliers_descs1 = desc1[matches[:, 0][inliers]]
    matched_inliers_descs2 = desc2[matches[:, 1][inliers]]

    if logits_1 is not None and logits_2 is not None:
        logits = torch.minimum(logits_1[matches[:, 0][inliers]],
                               logits_2[matches[:, 1][inliers]])
    else:
        logits = torch.ones(matched_inliers_descs1.shape[0], device=desc1.device)

    # === Shaped reward + margin annealing (from image 3) ===
    if label:  # positive pair
        snn_dists_1 = second_nearest_neighbor(matched_inliers_descs1, desc2)
        snn_dists_2 = second_nearest_neighbor(matched_inliers_descs2, desc1)
        dists_hard = torch.min(torch.hstack((snn_dists_1, snn_dists_2)), dim=1).values
        dists_pos = F.pairwise_distance(matched_inliers_descs1, matched_inliers_descs2)
        loss = torch.clamp(pos_margin + dists_pos - dists_hard, min=0.0)
    else:  # negative pair
        dists = F.pairwise_distance(matched_inliers_descs1, matched_inliers_descs2)
        loss = torch.clamp(neg_margin - dists, min=0.0)

    loss = loss * logits
    return loss.sum() / (logits.sum() + 1e-8)


class ContrastiveLoss(nn.Module):
    def __init__(self, pos_margin=1.0, neg_margin=1.0):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def forward(self, desc1, desc2, matches, inliers, label,
                logits_1=None, logits_2=None):
        return contrastive_loss(
            desc1, desc2, matches, inliers, label,
            logits_1, logits_2,
            self.pos_margin, self.neg_margin
        )