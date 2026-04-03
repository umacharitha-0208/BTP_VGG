import pyrootutils
import sys
from pathlib import Path

try:
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=["ripe", ".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )
except FileNotFoundError:
    # Fallback for environments without project markers like .git/pyproject.toml.
    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root))

SEED = 32000

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import collections
import os

import hydra
from hydra.utils import instantiate
from lightning.fabric import Fabric

# print(SEED)
import random

os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import wandb
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from ripe.utils import get_pylogger
from ripe.utils.utils import get_rewards
from ripe.utils.wandb_utils import get_flattened_wandb_cfg

log = get_pylogger(__name__)

torch.set_float32_matmul_precision('high')
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def unpack_batch(batch):
    # This function expects your custom dataset to return these keys
    # Adjust key names if your imw2022.py returns different ones
    src_image = batch["image0"]     # ← changed to match common naming
    trg_image = batch["image1"]
    # If your dataset does NOT return masks / homography / label, set them to None or remove
    trg_mask = batch.get("trg_mask")
    src_mask = batch.get("src_mask")
    if trg_mask is None:
        trg_mask = torch.ones((trg_image.shape[0], 1, trg_image.shape[2], trg_image.shape[3]), device=trg_image.device)
    if src_mask is None:
        src_mask = torch.ones((src_image.shape[0], 1, src_image.shape[2], src_image.shape[3]), device=src_image.device)
    label = batch.get("label", None)
    H = batch.get("homography", None)

    return src_image, trg_image, src_mask, trg_mask, H, label


@hydra.main(config_path="../conf/", config_name="train", version_base=None)
def train(cfg):
    """Main training function for the RIPE model - using only custom dataset."""

    strategy = "ddp" if cfg.num_gpus > 1 else "auto"
    fabric = Fabric(
        accelerator="cpu",          # Fallback to CPU for this quick 5-step test
        devices=1,
        precision=cfg.precision,
        strategy=strategy,
    )
    fabric.launch()

    output_dir = Path(cfg.output_dir)
    experiment_name = output_dir.parent.parent.parent.name
    run_id = output_dir.parent.parent.name
    timestamp = output_dir.parent.name + "_" + output_dir.name

    experiment_name = run_id + " " + timestamp + " " + experiment_name

    # setup logger
    wandb_logger = wandb.init(
        project=cfg.project_name,
        name=experiment_name,
        config=get_flattened_wandb_cfg(cfg),
        dir=cfg.output_dir,
        mode=cfg.wandb_mode,
    )

    batch_size = cfg.batch_size
    steps = cfg.num_steps
    lr = cfg.lr

    num_grad_accs = cfg.num_grad_accs  # gradient accumulation steps

    # instantiate your custom dataset (from cfg.data = imw2022)
    ds = instantiate(cfg.data)

    # prepare dataloader
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
        num_workers=cfg.num_workers,
    )
    dl = fabric.setup_dataloaders(dl)
    i_dl = iter(dl)

    # create matcher
    matcher = instantiate(cfg.matcher)

    if cfg.desc_loss_weight != 0.0:
        descriptor_loss = instantiate(cfg.descriptor_loss)
    else:
        log.warning(
            "Descriptor loss weight is 0.0 → descriptor loss disabled. "
            "1x1 conv for descriptors will be deactivated!"
        )
        descriptor_loss = None

    upsampler = instantiate(cfg.upsampler) if "upsampler" in cfg else None

    # create network
    net = instantiate(cfg.network)(
        net=instantiate(cfg.backbones),
        upsampler=upsampler,
        descriptor_dim=cfg.descriptor_dim if descriptor_loss is not None else None,
        device=fabric.device,
    ).train()

    # log number of parameters
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f"Number of trainable parameters: {num_params}")

    fp_penalty = cfg.fp_penalty
    kp_penalty = cfg.kp_penalty

    opt_pi = AdamW(filter(lambda x: x.requires_grad, net.parameters()), lr=lr, weight_decay=1e-5)
    net, opt_pi = fabric.setup(net, opt_pi)

    if cfg.lr_scheduler:
        scheduler = instantiate(cfg.lr_scheduler)(optimizer=opt_pi, steps_init=0)
    else:
        scheduler = None

    # mean average of skipped batches (monitor if model learns anything)
    ma_skipped_batches = collections.deque(maxlen=100)

    opt_pi.zero_grad()

    # schedulers for alpha, beta, inlier threshold
    alpha_scheduler = instantiate(cfg.alpha_scheduler)
    beta_scheduler = instantiate(cfg.beta_scheduler)
    inl_th_scheduler = instantiate(cfg.inl_th)

    # Open result file for logging
    result_file = Path(__file__).resolve().parent.parent / "results2.txt"
    result_fh = open(result_file, "w", buffering=1)  # Line buffered

    def write_result(msg):
        print(msg, flush=True)
        result_fh.write(msg + "\n")
        result_fh.flush()
        os.fsync(result_fh.fileno())

    # ======  Training Loop  ======
    net.train()

    with tqdm.tqdm(total=steps) as pbar:
        for i_step in range(steps):
            write_result(f"[Step {i_step+1}/{steps}] Started")

            alpha = alpha_scheduler(i_step)
            beta = beta_scheduler(i_step)
            inl_th = inl_th_scheduler(i_step)

            if scheduler:
                scheduler.step()

            sum_reward_batch = 0
            sum_num_keypoints_1 = 0
            sum_num_keypoints_2 = 0
            loss = None
            loss_policy_stack = None
            loss_desc_stack = None
            loss_kp_stack = None
            current_value_loss = None

            try:
                batch = next(i_dl)
            except StopIteration:
                i_dl = iter(dl)
                batch = next(i_dl)

            p1, p2, mask_padding_1, mask_padding_2, Hs, label = unpack_batch(batch)

            # forward pass - extract keypoints & descriptors (PPO: also returns value estimates)
            (kpts1, logprobs1, selected_mask1, mask_padding_grid_1, logits_selected_1, values1, out1) = net(
                p1, mask_padding_1, training=True
            )
            (kpts2, logprobs2, selected_mask2, mask_padding_grid_2, logits_selected_2, values2, out2) = net(
                p2, mask_padding_2, training=True
            )

            desc_1 = net.get_descs(out1["coarse_descs"], p1, kpts1, p1.shape[2], p1.shape[3])
            desc_2 = net.get_descs(out2["coarse_descs"], p2, kpts2, p2.shape[2], p2.shape[3])

            if cfg.padding_filter_mode == "ignore":
                # Ensure they are bool for the & operator
                batch_mask_selection_for_matching_1 = selected_mask1.bool() & mask_padding_grid_1.bool()
                batch_mask_selection_for_matching_2 = selected_mask2.bool() & mask_padding_grid_2.bool()
            elif cfg.padding_filter_mode == "punish":
                batch_mask_selection_for_matching_1 = selected_mask1
                batch_mask_selection_for_matching_2 = selected_mask2
            else:
                raise ValueError(f"Unknown padding filter mode: {cfg.padding_filter_mode}")

            # matching
            (
                batch_rel_idx_matches,
                batch_abs_idx_matches,
                batch_ransac_inliers,
                batch_Fm,
            ) = matcher(
                kpts1,
                kpts2,
                desc_1,
                desc_2,
                batch_mask_selection_for_matching_1,
                batch_mask_selection_for_matching_2,
                inl_th,
                label if cfg.no_filtering_negatives else None,
            )

            for b in range(batch_size):
                if batch_rel_idx_matches[b] is None:
                    ma_skipped_batches.append(1)
                    continue
                else:
                    ma_skipped_batches.append(0)

                mask_selection_for_matching_1 = batch_mask_selection_for_matching_1[b]
                mask_selection_for_matching_2 = batch_mask_selection_for_matching_2[b]

                rel_idx_matches = batch_rel_idx_matches[b]
                abs_idx_matches = batch_abs_idx_matches[b]
                ransac_inliers = batch_ransac_inliers[b]

                if cfg.selected_only:
                    dense_logprobs = (
                        logprobs1[b][mask_selection_for_matching_1.bool()].view(-1, 1)
                        + logprobs2[b][mask_selection_for_matching_2.bool()].view(1, -1)
                    )
                else:
                    if cfg.padding_filter_mode == "ignore":
                        dense_logprobs = (
                            logprobs1[b][mask_padding_grid_1[b].bool()].view(-1, 1)
                            + logprobs2[b][mask_padding_grid_2[b].bool()].view(1, -1)
                        )
                    elif cfg.padding_filter_mode == "punish":
                        dense_logprobs = logprobs1[b].view(-1, 1) + logprobs2[b].view(1, -1)
                    else:
                        raise ValueError(f"Unknown padding filter mode: {cfg.padding_filter_mode}")

                # reward calculation
                if cfg.reward_type == "inlier":
                    reward = 0.5 if cfg.no_filtering_negatives and not label[b] else 1.0
                elif cfg.reward_type == "inlier_ratio":
                    ratio_inlier = ransac_inliers.sum() / len(abs_idx_matches) if len(abs_idx_matches) > 0 else 0.0
                    reward = ratio_inlier
                elif cfg.reward_type == "inlier+inlier_ratio":
                    ratio_inlier = ransac_inliers.sum() / len(abs_idx_matches) if len(abs_idx_matches) > 0 else 0.0
                    reward = (1.0 - beta) * 1.0 + beta * ratio_inlier
                else:
                    raise ValueError(f"Unknown reward type: {cfg.reward_type}")

                dense_rewards = get_rewards(
                    reward,
                    kpts1[b],
                    kpts2[b],
                    mask_selection_for_matching_1,
                    mask_selection_for_matching_2,
                    mask_padding_grid_1[b],
                    mask_padding_grid_2[b],
                    rel_idx_matches,
                    abs_idx_matches,
                    ransac_inliers,
                    label[b] if label is not None else None,
                    fp_penalty * alpha,
                    use_whitening=cfg.use_whitening,
                    selected_only=cfg.selected_only,
                    filter_mode=cfg.padding_filter_mode,
                )

                if descriptor_loss is not None:
                    hard_loss = descriptor_loss(
                        desc1=desc_1[b],
                        desc2=desc_2[b],
                        matches=abs_idx_matches,
                        inliers=ransac_inliers,
                        label=label[b] if label is not None else None,
                        logits_1=None,
                        logits_2=None,
                    )
                    loss_desc_stack = (
                        hard_loss if loss_desc_stack is None else torch.hstack((loss_desc_stack, hard_loss))
                    )

                sum_reward_batch += dense_rewards.sum()

                # PPO: Compute value baseline for advantage estimation
                if cfg.selected_only:
                    dense_values = (
                        values1[b][mask_selection_for_matching_1.bool()].view(-1, 1)
                        + values2[b][mask_selection_for_matching_2.bool()].view(1, -1)
                    )
                else:
                    if cfg.padding_filter_mode == "ignore":
                        dense_values = (
                            values1[b][mask_padding_grid_1[b].bool()].view(-1, 1)
                            + values2[b][mask_padding_grid_2[b].bool()].view(1, -1)
                        )
                    elif cfg.padding_filter_mode == "punish":
                        dense_values = values1[b].view(-1, 1) + values2[b].view(1, -1)
                    else:
                        raise ValueError(f"Unknown padding filter mode: {cfg.padding_filter_mode}")

                # PPO advantage = reward - value baseline (reduces variance vs REINFORCE)
                advantages = dense_rewards - dense_values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                current_loss_policy = (advantages * dense_logprobs).view(-1)
                
                # PPO value loss: train critic to predict returns
                current_value_loss = F.mse_loss(dense_values, dense_rewards.detach())
                loss_policy_stack = (
                    current_loss_policy
                    if loss_policy_stack is None
                    else torch.hstack((loss_policy_stack, current_loss_policy))
                )

                if kp_penalty != 0.0:
                    loss_kp = (
                        logprobs1[b][mask_selection_for_matching_1]
                        * torch.full_like(logprobs1[b][mask_selection_for_matching_1], kp_penalty * alpha)
                    ).mean() + (
                        logprobs2[b][mask_selection_for_matching_2]
                        * torch.full_like(logprobs2[b][mask_selection_for_matching_2], kp_penalty * alpha)
                    ).mean()
                    loss_kp_stack = loss_kp if loss_kp_stack is None else torch.hstack((loss_kp_stack, loss_kp))

                sum_num_keypoints_1 += mask_selection_for_matching_1.sum()
                sum_num_keypoints_2 += mask_selection_for_matching_2.sum()

            # Skip step if no loss computed (no matches found)
            if loss_policy_stack is None:
                pbar.update()
                msg = f"[Step {i_step+1}/{steps}] Skipped (no matches found)"
                write_result(msg)
                continue

            loss = loss_policy_stack.mean()
            if loss_kp_stack is not None:
                loss += loss_kp_stack.mean()

            loss = -loss

            # PPO value loss (train critic)
            if current_value_loss is not None:
                loss += 0.5 * current_value_loss

            if descriptor_loss is not None and loss_desc_stack is not None:
                loss += cfg.desc_loss_weight * loss_desc_stack.mean()

            # Calculate metrics
            mean_reward = sum_reward_batch / batch_size
            avg_det_1 = sum_num_keypoints_1 / batch_size
            avg_det_2 = sum_num_keypoints_2 / batch_size
            
            # Print clean progress line for every step
            msg = f"[Step {i_step+1}/{steps}] Loss: {loss.item():.4f}, Reward: {mean_reward:.1f}, Det: ({avg_det_1:.0f}, {avg_det_2:.0f})"
            write_result(msg)

            pbar.update()

            # backward + optimization
            loss /= num_grad_accs
            fabric.backward(loss)

            if i_step % num_grad_accs == 0:
                opt_pi.step()
                opt_pi.zero_grad()

            # logging
            if i_step % cfg.log_interval == 0:
                wandb_logger.log(
                    {
                        "loss": loss.item(),
                        "loss_policy": -loss_policy_stack.mean().item(),
                        "loss_kp": loss_kp_stack.mean().item() if loss_kp_stack is not None else 0.0,
                        "loss_hard": (loss_desc_stack.mean().item() if loss_desc_stack is not None else 0.0),
                        "mean_num_det_kpts1": sum_num_keypoints_1 / batch_size,
                        "mean_num_det_kpts2": sum_num_keypoints_2 / batch_size,
                        "mean_reward": sum_reward_batch / batch_size,
                        "lr": opt_pi.param_groups[0]["lr"],
                        "ma_skipped_batches": sum(ma_skipped_batches) / len(ma_skipped_batches),
                        "inl_th": inl_th,
                    },
                    step=i_step,
                )

    # save final model (strip Fabric wrapper prefix from keys)
    raw_state = net.state_dict()
    clean_state = {k.replace("_forward_module.", ""): v for k, v in raw_state.items()}
    save_path = output_dir / "model_weights.pt"
    torch.save(clean_state, save_path)
    log.info(f"Saved trained model to {save_path}")
    
    # Write final summary
    result_fh.write(f"\n=== Training Complete ===\n")
    result_fh.write(f"Total steps: {steps}\n")
    result_fh.write(f"Model saved to: {save_path}\n")
    result_fh.write(f"Parameters: {num_params}\n")
    result_fh.close()
    
    print(f"\n=== Training Complete ===", flush=True)
    print(f"Total steps: {steps}", flush=True)
    print(f"Model saved to: {save_path}", flush=True)
    print(f"Parameters: {num_params}", flush=True)


if __name__ == "__main__":
    train()
    