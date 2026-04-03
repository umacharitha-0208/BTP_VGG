import concurrent.futures
import torch
from ripe.losses.contrastive_loss import second_nearest_neighbor   # reuse existing function


class ConcurrentMatcher:
    def __init__(self, matcher, robust_estimator, min_num_matches=8, max_workers=12):
        self.matcher = matcher
        self.robust_estimator = robust_estimator
        self.min_num_matches = min_num_matches
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    @torch.no_grad()
    def __call__(self, kpts1, kpts2, pdesc1, pdesc2,
                 selected_mask1, selected_mask2, inl_th, label=None):

        B = pdesc1.shape[0]
        dev = pdesc1.device

        batch_rel_idx_matches = [None] * B
        batch_idx_matches = [None] * B
        batch_ransac_inliers = [None] * B
        batch_Fm = [None] * B

        for b in range(B):
            if selected_mask1[b].sum() < 16 or selected_mask2[b].sum() < 16:
                continue

            # Get selected descriptors
            desc1_selected = pdesc1[b][selected_mask1[b]]
            desc2_selected = pdesc2[b][selected_mask2[b]]
            
            # Get matches from matcher
            matches = self.matcher(desc1_selected, desc2_selected)
            
            if matches is None or matches[1] is None or len(matches[1]) == 0:
                batch_ransac_inliers[b] = torch.tensor([], dtype=torch.bool, device=dev)
                continue
                
            rel_idx = matches[1]  # indices of matches
            
            # Compute pairwise distances for Lowe's ratio test
            dists_full = torch.cdist(desc1_selected, desc2_selected, p=2)
            
            # Get distances for matched pairs (first and second nearest)
            top2_dists, _ = torch.topk(dists_full, k=min(2, dists_full.shape[1]), dim=1, largest=False)
            
            # Matched distances
            matched_first_dists = top2_dists[rel_idx[:, 0], 0]
            if top2_dists.shape[1] > 1:
                matched_second_dists = top2_dists[rel_idx[:, 0], 1]
            else:
                matched_second_dists = torch.full_like(matched_first_dists, float('inf'))
            
            # Lowe's ratio test: first_dist / second_dist < threshold
            lowe_mask = (matched_first_dists / (matched_second_dists + 1e-8)) < 0.8
            rel_idx = rel_idx[lowe_mask]

            if rel_idx.shape[0] < self.min_num_matches:
                batch_ransac_inliers[b] = torch.zeros(rel_idx.shape[0], device=dev, dtype=torch.bool)
                continue

            # Convert to absolute indices
            abs_idx0 = torch.nonzero(selected_mask1[b], as_tuple=False)[rel_idx[:, 0]]
            abs_idx1 = torch.nonzero(selected_mask2[b], as_tuple=False)[rel_idx[:, 1]]
            
            if abs_idx0.dim() > 1:
                abs_idx0 = abs_idx0.squeeze(1)
            if abs_idx1.dim() > 1:
                abs_idx1 = abs_idx1.squeeze(1)
                
            idx_matches = torch.stack([abs_idx0, abs_idx1], dim=1)

            batch_idx_matches[b] = idx_matches
            batch_rel_idx_matches[b] = rel_idx

            if label is not None and label[b] == 0:
                # Hard negative: still run full RANSAC (image 5 recommendation)
                mkpts1 = kpts1[b][idx_matches[:, 0]]
                mkpts2 = kpts2[b][idx_matches[:, 1]]
                future = self.executor.submit(self.robust_estimator, mkpts1, mkpts2, inl_th)
                Fm, ransac_inliers = future.result()
            else:
                mkpts1 = kpts1[b][idx_matches[:, 0]]
                mkpts2 = kpts2[b][idx_matches[:, 1]]
                future = self.executor.submit(self.robust_estimator, mkpts1, mkpts2, inl_th)
                Fm, ransac_inliers = future.result()

            batch_ransac_inliers[b] = ransac_inliers
            batch_Fm[b] = Fm

        return batch_rel_idx_matches, batch_idx_matches, batch_ransac_inliers, batch_Fm