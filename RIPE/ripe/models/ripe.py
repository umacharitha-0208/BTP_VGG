import sys
from pathlib import Path

# Fix module path conflicts when running this file directly  
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    models_dir = str(Path(__file__).resolve().parent)
    sys.path = [p for p in sys.path if Path(p).resolve() != Path(models_dir).resolve()]
    sys.path.insert(0, str(project_root))
    
    if 'ripe' in sys.modules:
        del sys.modules['ripe']

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ripe.utils import get_pylogger
from ripe.utils.utils import gridify

log = get_pylogger(__name__)


###############################################################
# PPO POLICY NETWORK (Improved for stability)
###############################################################

class PPOPolicy(nn.Module):
    """PPO Policy with actor (Bernoulli) and critic (value head)"""

    def __init__(self, input_dim=1, hidden=64):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

        # Value head for GAE / advantage estimation
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        prob = self.actor(x)          # acceptance probability
        value = self.critic(x)        # state value for advantage
        return prob, value


###############################################################
# KEYPOINT SAMPLER
###############################################################

class KeypointSampler(nn.Module):
    """
    Improved keypoint sampler with PPO policy for acceptance decision.
    """

    def __init__(self, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.idx_cells = None
        self.policy = PPOPolicy()

    def sample(self, grid):
        chooser = torch.distributions.Categorical(logits=grid)
        choices = chooser.sample()

        logits_selected = torch.gather(grid, -1, choices.unsqueeze(-1)).squeeze(-1)

        # PPO decision
        policy_input = logits_selected.unsqueeze(-1)
        probs, values = self.policy(policy_input)

        probs = probs.squeeze(-1)
        values = values.squeeze(-1)

        flipper = torch.distributions.Bernoulli(probs)
        accepted_choices = flipper.sample()

        log_probs = chooser.log_prob(choices) + flipper.log_prob(accepted_choices)
        entropy = chooser.entropy() + flipper.entropy()

        accept_mask = accepted_choices.gt(0)

        return (
            log_probs.squeeze(1),
            choices,
            accept_mask.squeeze(1),
            logits_selected.squeeze(1),
            values.squeeze(1),   # for GAE
            entropy.squeeze(1),
            accepted_choices,
        )

    def precompute_idx_cells(self, H, W, device):
        idx_cells = gridify(
            torch.dstack(
                torch.meshgrid(
                    torch.arange(H, dtype=torch.float32, device=device),
                    torch.arange(W, dtype=torch.float32, device=device),
                )
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(1, -1, -1, -1),
            window_size=self.window_size,
        )
        return idx_cells

    def forward(self, x, mask_padding=None):
        B, C, H, W = x.shape

        keypoint_cells = gridify(x, self.window_size)

        if mask_padding is not None:
            mask_padding = torch.min(gridify(mask_padding, self.window_size), dim=4).values

        if self.idx_cells is None or self.idx_cells.shape[2:4] != (H // self.window_size, W // self.window_size):
            self.idx_cells = self.precompute_idx_cells(H, W, x.device)

        log_probs, choices, mask, logits_selected, values, entropy, accepted_choices = self.sample(keypoint_cells)

        keypoints = (
            torch.gather(
                self.idx_cells.expand(B, -1, -1, -1, -1),
                -1,
                choices.repeat(1, 2, 1, 1).unsqueeze(-1),
            )
            .squeeze(-1)
            .permute(0, 2, 3, 1)
        )

        return (
            keypoints.flip(-1),
            log_probs,
            mask,
            mask_padding,
            logits_selected,
            values,
            entropy,
            choices.view(B, -1),
            accepted_choices.view(B, -1),
        )


###############################################################
# RIPE MODEL (Main Model)
###############################################################

class RIPE(nn.Module):
    def __init__(
        self,
        net,
        upsampler,
        window_size: int = 8,
        non_linearity_dect=None,
        desc_shares: Optional[List[int]] = None,
        descriptor_dim: int = 256,
        device=None,
    ):
        super().__init__()

        self.net = net
        self.detector = KeypointSampler(window_size)
        self.upsampler = upsampler

        self.window_size = window_size
        self.non_linearity_dect = non_linearity_dect if non_linearity_dect else nn.Identity()

        log.info(f"Training with window size {window_size}.")

        dim_coarse_desc = self.get_dim_raw_desc()

        if desc_shares is not None:
            self.conv_dim_reduction_coarse_desc = nn.ModuleList()
            for dim_in, dim_out in zip(dim_coarse_desc, desc_shares):
                self.conv_dim_reduction_coarse_desc.append(
                    nn.Conv1d(dim_in, dim_out, kernel_size=1)
                )
        else:
            self.conv_dim_reduction_coarse_desc = nn.Conv1d(
                sum(dim_coarse_desc), descriptor_dim, kernel_size=1
            )

    def get_dim_raw_desc(self):
        layers_dims_encoder = self.net.get_dim_layers_encoder()

        if self.upsampler.name == "InterpolateSparse2d":
            return [layers_dims_encoder[-1]]
        elif self.upsampler.name == "HyperColumnFeatures":
            return layers_dims_encoder
        else:
            raise ValueError(f"Unknown interpolator {self.upsampler.name}")

    ###############################################################
    # DYNAMIC KEYPOINT SELECTION
    ###############################################################

    def dynamic_topk(self, scores, max_k=2048, quality_threshold=0.3, verbose=False):
        num_available = scores.shape[0]
        high_quality_mask = scores >= quality_threshold
        num_high_quality = high_quality_mask.sum().item()
        mean_score = scores.mean().item()

        if mean_score > 0.6:
            target_k = int(max_k * 0.4)
        elif mean_score > 0.4:
            target_k = int(max_k * 0.7)
        elif mean_score > 0.25:
            target_k = int(max_k * 0.9)
        else:
            target_k = max_k

        k = min(target_k, num_high_quality, num_available)
        k = max(k, 32)  # minimum for stable matching

        if verbose:
            print(f"[Dynamic TopK] Available: {num_available}, High Quality: {num_high_quality}, "
                  f"Mean: {mean_score:.3f} → Selected: {k}")

        return k

    ###############################################################

    @torch.inference_mode()
    def detectAndCompute(self, img, threshold=0.5, top_k=2048, output_aux=False):
        self.eval()

        if img.dim() == 3:
            img = img.unsqueeze(0)

        out = self(img, training=False)

        B, K, H, W = out["heatmap"].shape
        assert B == 1

        kpts = [{"xy": self.NMS(out["heatmap"][b], threshold)} for b in range(B)]

        score_map = out["heatmap"][0].squeeze(0)
        scores = score_map[
            kpts[0]["xy"][:, 1].long(),
            kpts[0]["xy"][:, 0].long()
        ]
        scores = scores / score_map.max()

        # Use fixed top_k for speed (dynamic_topk can be re-enabled if needed)
        sorted_idx = torch.argsort(-scores)
        kpts = kpts[0]["xy"][sorted_idx[:min(top_k, len(sorted_idx))]]

        # Spatial sanity filter
        spatial_mask = (
            (kpts[:, 0] > 16) & (kpts[:, 1] > 16) &
            (kpts[:, 0] < W - 16) & (kpts[:, 1] < H - 16)
        )
        kpts = kpts[spatial_mask]

        if kpts.shape[0] == 0:
            raise RuntimeError("No keypoints detected")

        descs = self.get_descs(
            out["coarse_descs"], img, kpts.unsqueeze(0), H, W
        )
        descs = descs.squeeze(0)

        scores = score_map[kpts[:, 1], kpts[:, 0]]
        sort_idx = torch.argsort(-scores)
        kpts, descs, scores = kpts[sort_idx], descs[sort_idx], scores[sort_idx]

        if output_aux:
            return kpts.float(), descs, scores, {"heatmap": out["heatmap"], "coarse_descs": out["coarse_descs"]}

        return kpts.float(), descs, scores

    def NMS(self, x, threshold=3.0, kernel_size=3):
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        return pos.nonzero()[..., 1:].flip(-1)

    def get_descs(self, feature_map, guidance, kpts, H, W):
        descs = self.upsampler(feature_map, kpts, H, W)
        desc = torch.cat(descs, dim=-1)

        desc = self.conv_dim_reduction_coarse_desc(
            desc.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # Descriptor L2 normalization (important fix from image 4)
        desc = F.normalize(desc, dim=2)

        return desc

    ###############################################################
    # FORWARD PASS
    ###############################################################

    def forward(self, x, mask_padding=None, training=False):
        B, C, H, W = x.shape

        out = self.net(x)
        out["heatmap"] = self.non_linearity_dect(out["heatmap"])

        if training:
            kpts, log_probs, mask, mask_padding, logits_selected, values, entropy, choices, accepted_choices = self.detector(
                out["heatmap"], mask_padding
            )

            # Spatial filtering
            filter_A = kpts[:, :, :, 0] >= 16
            filter_B = kpts[:, :, :, 1] >= 16
            filter_C = kpts[:, :, :, 0] < W - 16
            filter_D = kpts[:, :, :, 1] < H - 16
            filter_all = filter_A * filter_B * filter_C * filter_D

            mask = mask * filter_all

            return (
                kpts.view(B, -1, 2),
                log_probs.view(B, -1),
                mask.view(B, -1),
                mask_padding.view(B, -1) if mask_padding is not None else None,
                logits_selected.view(B, -1),
                values.view(B, -1),
                entropy.view(B, -1),
                choices,
                accepted_choices,
                out,
            )
        else:
            return out

    ###############################################################
    # PPO EVALUATION (for old policy re-evaluation)
    ###############################################################

    def evaluate_actions(self, x, choices, accepted_choices, mask_padding=None):
        """Evaluate log probs and values for given actions (for PPO)"""
        B, C, H, W = x.shape

        out = self.net(x)
        out["heatmap"] = self.non_linearity_dect(out["heatmap"])

        # Get the grid for evaluation
        keypoint_cells = gridify(out["heatmap"], self.window_size)

        if mask_padding is not None:
            mask_padding_grid = torch.min(gridify(mask_padding, self.window_size), dim=4).values
        else:
            mask_padding_grid = None

        if self.detector.idx_cells is None or self.detector.idx_cells.shape[2:4] != (H // self.window_size, W // self.window_size):
            self.detector.idx_cells = self.detector.precompute_idx_cells(H, W, x.device)

        # Flatten keypoint_cells so each spatial location has a category vector
        # keypoint_cells has shape [B, C, grid_h, grid_w, num_cells] from gridify
        B_size, C, grid_h, grid_w, num_cells = keypoint_cells.shape
        # Squeeze C dimension (should be 1 for heatmap) and reshape to [B, grid_h*grid_w, num_cells]
        keypoint_cells = keypoint_cells.squeeze(1)  # [B, grid_h, grid_w, num_cells]
        keypoint_cells_flat = keypoint_cells.reshape(B_size, grid_h * grid_w, num_cells)

        # choices comes in shape [B, grid_h * grid_w]
        choices_flat = choices.view(B_size, -1).unsqueeze(-1)

        logits_selected = keypoint_cells_flat.gather(1, choices_flat).squeeze(-1)

        # PPO decision evaluation
        policy_input = logits_selected.unsqueeze(-1)
        probs, values = self.detector.policy(policy_input)

        probs = probs.squeeze(-1)
        values = values.squeeze(-1)

        # Compute log probs for the given actions
        chooser = torch.distributions.Categorical(logits=keypoint_cells_flat)
        flipper = torch.distributions.Bernoulli(probs)

        log_probs = chooser.log_prob(choices)
        entropy = chooser.entropy() + flipper.entropy()

        return (
            log_probs.squeeze(1),
            values.squeeze(1),
            entropy.squeeze(1),
            out,
        )


###############################################################
# UTILITY
###############################################################

def output_number_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of trainable parameters: {nb_params:d}")