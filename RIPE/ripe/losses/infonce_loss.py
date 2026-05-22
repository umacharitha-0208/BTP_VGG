"""InfoNCE descriptor loss.

Works even when RANSAC finds 0 matches — uses all selected keypoints as a
queue. Provides a non-zero gradient on every batch, unlike the contrastive
hinge loss which returns 0 when no matches exist.

For positive pairs: matched keypoints (or all if no matches, treated as soft
positives) are pulled together; all others in the batch act as negatives.

For negative pairs: no positive pull — only the uniformity (spread) term.
"""

import torch
import torch.nn.functional as F


def infonce_loss(
    desc1: torch.Tensor,          # (N, D) L2-normalised
    desc2: torch.Tensor,          # (M, D) L2-normalised
    matches: torch.Tensor,        # (K, 2) absolute match indices, may be empty
    inliers: torch.Tensor,        # (K,) bool inlier mask, may be empty
    label: float,                 # 1.0 = positive pair, 0.0 = negative pair
    temperature: float = 0.07,
    uniformity_weight: float = 0.1,
) -> torch.Tensor:
    device = desc1.device

    # ── Uniformity regularizer (always active) ───────────────────────────────
    # Encourages descriptors to spread over the unit sphere.
    # Works on both positive and negative pairs, and requires no matches.
    def _uniformity(d):
        sq = torch.sum(d ** 2, dim=1, keepdim=True)
        sq2 = sq + sq.T - 2 * (d @ d.T)
        # Exclude diagonal (self-pairs give sq2=0 → exp=1.0, inflating the mean)
        mask = ~torch.eye(d.shape[0], dtype=torch.bool, device=d.device)
        return torch.log(torch.exp(-2 * sq2[mask]).mean() + 1e-8)

    unif = uniformity_weight * (_uniformity(desc1) + _uniformity(desc2)) * 0.5

    if not label:
        # Negative pair: only spread descriptors
        return unif

    # ── InfoNCE on matched / pseudo-matched pairs ─────────────────────────────
    if matches is not None and len(matches) > 0 and len(inliers) > 0:
        valid = inliers.bool()
        if valid.sum() >= 2:
            idx1 = matches[:, 0][valid]
            idx2 = matches[:, 1][valid]
            anc = desc1[idx1]       # (K, D)
            pos = desc2[idx2]       # (K, D)
        else:
            # Fall back to all matches as pseudo-positives
            anc = desc1[matches[:, 0]]
            pos = desc2[matches[:, 1]]
    else:
        # No matches at all — use top-k mutual nearest pairs as pseudo-positives
        # (works even with fully random descriptors, providing weak gradient)
        k = min(16, desc1.shape[0], desc2.shape[0])
        with torch.no_grad():
            sim = desc1[:k] @ desc2[:k].T          # (k, k)
            nn1 = sim.argmax(dim=1)                # (k,) nearest in desc2
            nn2 = sim.argmax(dim=0)                # (k,) nearest in desc1
            mutual = (nn2[nn1] == torch.arange(k, device=device))
            if mutual.sum() == 0:
                return unif                        # nothing to pull together
            sel = mutual.nonzero(as_tuple=False).squeeze(1)
        anc = desc1[sel]
        pos = desc2[nn1[sel]]

    # Symmetric InfoNCE
    all_desc2 = desc2[:min(256, desc2.shape[0])]  # cap queue for memory on CPU
    logits_anc = anc @ all_desc2.T / temperature   # (K, Q)
    targets = torch.zeros(anc.shape[0], dtype=torch.long, device=device)
    # find where positive lives in all_desc2
    for i, p in enumerate(pos):
        sims = (all_desc2 * p).sum(-1)
        targets[i] = sims.argmax()
    nce = F.cross_entropy(logits_anc, targets)

    return nce + unif


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07, uniformity_weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.uniformity_weight = uniformity_weight

    def forward(self, desc1, desc2, matches, inliers, label,
                logits_1=None, logits_2=None):
        label_bool = float(label) > 0.5 if not isinstance(label, bool) else label
        return infonce_loss(
            desc1, desc2, matches, inliers, label_bool,
            self.temperature, self.uniformity_weight,
        )
