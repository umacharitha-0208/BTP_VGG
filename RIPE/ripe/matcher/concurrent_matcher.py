import concurrent.futures
import torch


class ConcurrentMatcher:
    def __init__(self, matcher, robust_estimator, min_num_matches=2, max_workers=12, top_k=512):
        self.matcher = matcher
        self.robust_estimator = robust_estimator
        self.min_num_matches = min_num_matches
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.top_k = top_k  # max keypoints per image fed to MNN

    @torch.no_grad()
    def __call__(self, kpts1, kpts2, pdesc1, pdesc2,
                 selected_mask1, selected_mask2, inl_th, label=None,
                 scores1=None, scores2=None):
        """
        scores1/scores2: (B, N) detector confidence per keypoint cell.
        When provided, only the top-K scoring keypoints enter MNN.
        Matches are then sorted by descriptor similarity so poselib's
        progressive_sampling (PROSAC) draws the best candidates first.
        """
        B = pdesc1.shape[0]
        dev = pdesc1.device

        batch_rel_idx_matches = [torch.empty((0, 2), dtype=torch.long, device=dev)] * B
        batch_idx_matches = [torch.empty((0, 2), dtype=torch.long, device=dev)] * B
        batch_ransac_inliers = [torch.empty(0, dtype=torch.bool, device=dev)] * B
        batch_Fm = [None] * B

        for b in range(B):
            # Absolute indices of PPO-accepted keypoints
            sel_idx1 = torch.nonzero(selected_mask1[b], as_tuple=False).squeeze(1)
            sel_idx2 = torch.nonzero(selected_mask2[b], as_tuple=False).squeeze(1)

            if len(sel_idx1) < self.min_num_matches or len(sel_idx2) < self.min_num_matches:
                continue

            # ── PROSAC pre-filter: top-K by detector score ───────────────────────
            # Keeps only the most confident keypoints entering MNN, reducing the
            # chance that low-quality keypoints produce spurious mutual matches.
            if scores1 is not None and len(sel_idx1) > self.top_k:
                sc1 = scores1[b][sel_idx1]
                topk_idx1 = torch.topk(sc1, self.top_k).indices
                # Sort within selection by descending score → PROSAC ordering
                topk_idx1 = topk_idx1[torch.argsort(-sc1[topk_idx1])]
                sel_idx1 = sel_idx1[topk_idx1]

            if scores2 is not None and len(sel_idx2) > self.top_k:
                sc2 = scores2[b][sel_idx2]
                topk_idx2 = torch.topk(sc2, self.top_k).indices
                topk_idx2 = topk_idx2[torch.argsort(-sc2[topk_idx2])]
                sel_idx2 = sel_idx2[topk_idx2]

            desc1_selected = pdesc1[b][sel_idx1]   # (K1, D)
            desc2_selected = pdesc2[b][sel_idx2]   # (K2, D)

            # Pure MNN — LO-RANSAC handles geometric outlier rejection
            matches = self.matcher(desc1_selected, desc2_selected)

            if matches is None or matches[1] is None or len(matches[1]) == 0:
                continue

            rel_idx = matches[1]  # (M, 2) within-selection indices

            if rel_idx.shape[0] < self.min_num_matches:
                continue

            # ── Sort by descriptor cosine similarity (descending) ────────────────
            # Passes matches to poselib in quality order so progressive_sampling
            # (PROSAC) draws the highest-similarity correspondences first.
            sim = (desc1_selected[rel_idx[:, 0]] * desc2_selected[rel_idx[:, 1]]).sum(dim=1)
            sort_order = torch.argsort(-sim)
            rel_idx = rel_idx[sort_order]

            # Map within-selection indices → absolute keypoint indices
            abs_idx0 = sel_idx1[rel_idx[:, 0]]
            abs_idx1 = sel_idx2[rel_idx[:, 1]]

            idx_matches = torch.stack([abs_idx0, abs_idx1], dim=1)
            batch_idx_matches[b] = idx_matches
            # Use absolute indices for rel too (consistent with synthetic pairs path)
            batch_rel_idx_matches[b] = idx_matches

            mkpts1 = kpts1[b][abs_idx0]
            mkpts2 = kpts2[b][abs_idx1]
            future = self.executor.submit(self.robust_estimator, mkpts1, mkpts2, inl_th)
            Fm, ransac_inliers = future.result()

            batch_ransac_inliers[b] = ransac_inliers
            batch_Fm[b] = Fm

        return batch_rel_idx_matches, batch_idx_matches, batch_ransac_inliers, batch_Fm