import poselib
import torch


class PoseLibRelativePoseEstimator:
    def __init__(self):
        pass

    def __call__(self, pts0, pts1, inl_th):
        # Adaptive threshold (image 5)
        scale = torch.median(torch.cdist(pts0, pts0, p=2)) + 1e-6
        adaptive_th = max(0.5, min(2.0, scale * 1.5))   # 1.5× median keypoint scale

        F, info = poselib.estimate_fundamental(
            pts0.cpu().numpy(),
            pts1.cpu().numpy(),
            {"max_epipolar_error": adaptive_th, "max_iterations": 1000, "min_iterations": 100},
        )

        success = F is not None
        if success:
            inliers = torch.tensor(info.pop("inliers"), dtype=torch.bool, device=pts0.device)
        else:
            inliers = torch.zeros(pts0.shape[0], dtype=torch.bool, device=pts0.device)

        return F, inliers