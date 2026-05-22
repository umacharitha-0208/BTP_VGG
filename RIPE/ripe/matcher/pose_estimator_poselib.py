import poselib
import torch


class PoseLibRelativePoseEstimator:
    def __init__(self):
        pass

    def __call__(self, pts0, pts1, inl_th):
        pts0_np = pts0.cpu().numpy()
        pts1_np = pts1.cpu().numpy()

        # Adaptive threshold: 1.5× median inter-keypoint scale, clamped to [0.5, 3.0]
        scale = torch.median(torch.cdist(pts0, pts0, p=2)) + 1e-6
        adaptive_th = float(max(0.5, min(3.0, scale * 1.5)))

        # ── LO-RANSAC via PoseLib ────────────────────────────────────────────
        # lo_iterations: local-optimization refinement steps inside RANSAC.
        # progressive_sampling: PROSAC-style ordering (quality-guided sampling).
        lo_ransac_opts = {
            "max_epipolar_error": adaptive_th,
            "max_iterations": 1000,
            "min_iterations": 50,
            "lo_iterations": 25,          # key LO-RANSAC parameter
            "progressive_sampling": True,  # PROSAC-style guided sampling
        }

        F, info = poselib.estimate_fundamental(pts0_np, pts1_np, lo_ransac_opts)

        if F is not None:
            inliers = torch.tensor(info.pop("inliers"), dtype=torch.bool, device=pts0.device)
            # If fundamental matrix finds too few inliers, try homography (planar scenes)
            if inliers.sum() < 4 and len(pts0_np) >= 4:
                H_mat, h_info = poselib.estimate_homography(
                    pts0_np, pts1_np,
                    {"max_reproj_error": adaptive_th * 2, "max_iterations": 500,
                     "min_iterations": 20, "lo_iterations": 10},
                )
                if H_mat is not None:
                    h_inliers = torch.tensor(h_info.pop("inliers"), dtype=torch.bool, device=pts0.device)
                    if h_inliers.sum() > inliers.sum():
                        return H_mat, h_inliers
            return F, inliers

        # Fundamental matrix failed — try homography (works well for planar or narrow-baseline scenes)
        if len(pts0_np) >= 4:
            H_mat, h_info = poselib.estimate_homography(
                pts0_np, pts1_np,
                {"max_reproj_error": adaptive_th * 2, "max_iterations": 500,
                 "min_iterations": 20, "lo_iterations": 10},
            )
            if H_mat is not None:
                inliers = torch.tensor(h_info.pop("inliers"), dtype=torch.bool, device=pts0.device)
                return H_mat, inliers

        inliers = torch.zeros(pts0.shape[0], dtype=torch.bool, device=pts0.device)
        return None, inliers
