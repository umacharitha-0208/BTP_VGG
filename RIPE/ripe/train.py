import multiprocessing
import pyrootutils
import sys
from pathlib import Path

# Windows requires spawn context for DataLoader workers with CUDA
if sys.platform == "win32":
    multiprocessing.set_start_method("spawn", force=True)

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
import gc
import math
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


def _synthetic_inliers(kpts1, kpts2, H_matrix, threshold=8.0):
    """Compute inlier matches directly from a known homography (no RANSAC needed).

    Projects every keypoint in kpts1 through H, then finds the nearest keypoint
    in kpts2. Pairs within `threshold` pixels are inliers.

    Args:
        kpts1:    (N, 2) float tensor of [x, y] pixel coords in image 1
        kpts2:    (M, 2) float tensor of [x, y] pixel coords in image 2
        H_matrix: (3, 3) float tensor — pixel-space homography from image1→image2
        threshold: inlier pixel distance threshold
    Returns:
        abs_idx_matches: (K, 2) long tensor of matched (i, j) index pairs
        ransac_inliers:  (K,) bool tensor — all True (every returned pair is an inlier)
    """
    if kpts1.shape[0] == 0 or kpts2.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.long, device=kpts1.device), \
               torch.empty(0, dtype=torch.bool, device=kpts1.device)

    H = H_matrix.to(kpts1.device).float()
    ones = torch.ones(kpts1.shape[0], 1, device=kpts1.device)
    kpts1_h = torch.cat([kpts1.float(), ones], dim=1)          # (N, 3)
    proj = (H @ kpts1_h.T).T                                   # (N, 3)
    proj = proj[:, :2] / proj[:, 2:3].clamp(min=1e-6)          # (N, 2)

    dists = torch.cdist(proj, kpts2.float())                    # (N, M)
    min_dist, nn_idx = dists.min(dim=1)

    inlier_mask = min_dist < threshold                          # (N,)
    src_idx = torch.where(inlier_mask)[0]
    dst_idx = nn_idx[inlier_mask]

    abs_idx_matches = torch.stack([src_idx, dst_idx], dim=1)
    ransac_inliers  = torch.ones(abs_idx_matches.shape[0], dtype=torch.bool,
                                 device=kpts1.device)
    return abs_idx_matches, ransac_inliers

log = get_pylogger(__name__)

torch.set_float32_matmul_precision('high')
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def collate_fn(batch):
    """Custom collate that handles per-sample homography=None (real pairs)."""
    from torch.utils.data.dataloader import default_collate
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in batch]
        if k == "homography":
            # Keep as list — may contain None (real) or Tensor (synthetic)
            out[k] = vals
        else:
            out[k] = default_collate(vals)
    return out


def unpack_batch(batch):
    src_image = batch["image0"]
    trg_image = batch["image1"]
    src_mask = batch.get("src_mask")
    trg_mask = batch.get("trg_mask")
    if trg_mask is None:
        trg_mask = torch.ones((trg_image.shape[0], 1, trg_image.shape[2], trg_image.shape[3]))
    if src_mask is None:
        src_mask = torch.ones((src_image.shape[0], 1, src_image.shape[2], src_image.shape[3]))
    label = batch.get("label", None)
    H = batch.get("homography", None)   # list of (3,3) tensors or None per sample
    return src_image, trg_image, src_mask, trg_mask, H, label


@hydra.main(config_path="../conf/", config_name="train", version_base=None)
def train(cfg):
    """Main training function for the RIPE model - using only custom dataset."""

    # ── Device setup — resolve GPU explicitly, don't rely on fabric.device ──────
    _cuda_available = torch.cuda.is_available()
    print(f"[DEVICE] Python: {sys.executable}", flush=True)
    print(f"[DEVICE] PyTorch: {torch.__version__}", flush=True)
    print(f"[DEVICE] CUDA available: {_cuda_available}", flush=True)
    if not _cuda_available:
        import os
        print(f"[DEVICE] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}", flush=True)
        raise RuntimeError(
            "\n\n*** CUDA NOT AVAILABLE ***\n"
            f"  Python used: {sys.executable}\n"
            f"  PyTorch: {torch.__version__}\n"
            "  You are running the wrong Python. Use:\n"
            '  "C:\\Program Files\\Python310\\python.exe" -m ripe.train\n'
        )
    DEVICE = torch.device("cuda:0")
    print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB", flush=True)
    print(f"[DEVICE] Training on: {DEVICE}", flush=True)

    strategy = "ddp" if cfg.num_gpus > 1 else "auto"
    fabric = Fabric(
        accelerator="auto",
        devices=cfg.num_gpus,
        precision=cfg.precision,
        strategy=strategy,
    )
    fabric.launch()
    print(f"[DEVICE] fabric.device={fabric.device}  (using explicit DEVICE={DEVICE})", flush=True)

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

    # instantiate your custom dataset (gitrom cfg.data = imw2022)
    ds = instantiate(cfg.data)
    log.info(f"Dataset initialized with {len(ds)} samples")

    # prepare dataloader
    log.info(f"Creating DataLoader with num_workers={cfg.num_workers}, batch_size={batch_size}")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,   # keep workers alive between batches
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),      # fast CPU→GPU pin only when GPU present
        collate_fn=collate_fn,
    )
    log.info("DataLoader created, skipping fabric.setup_dataloaders for Windows compatibility")
    # dl = fabric.setup_dataloaders(dl)  # Skip on Windows to avoid hanging
    i_dl = iter(dl)
    log.info("DataLoader iterator created")

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
        device=DEVICE,
    ).train()

    # log number of parameters
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f"Number of trainable parameters: {num_params}")

    fp_penalty = cfg.fp_penalty
    kp_penalty = cfg.kp_penalty

    opt_pi = AdamW(filter(lambda x: x.requires_grad, net.parameters()), lr=lr, weight_decay=1e-6)
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

    # running reward normalization for PPO stability (EMA using reward_norm_momentum)
    reward_norm_mean_ema = 0.0
    reward_norm_var_ema  = 1.0
    reward_norm_ema_init = False   # warm up EMA with first batch value

    # PPO rollout buffer
    rollout_buffer = []

    # Open result file inside the dated output folder — never overwritten between runs
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "results.txt"
    result_fh = open(result_file, "w", buffering=1)  # Line buffered

    def write_result(msg):
        print(msg, flush=True)
        result_fh.write(msg + "\n")
        result_fh.flush()
        os.fsync(result_fh.fileno())

    # ======  Training Loop  ======
    net.train()

    import time as _time
    _DEBUG = True   # set False after first successful run

    def _dbg(msg):
        if _DEBUG:
            print(f"[DEBUG {_time.strftime('%H:%M:%S')}] {msg}", flush=True)

    _dbg("Entering training loop")

    with tqdm.tqdm(total=steps) as pbar:
        for i_step in range(steps):
            _dbg(f"---- Step {i_step+1} start ----")

            alpha = alpha_scheduler(i_step)
            beta = beta_scheduler(i_step)
            inl_th = inl_th_scheduler(i_step)

            if scheduler:
                scheduler.step()

            if cfg.lr_warmup_steps > 0:
                lr_factor = min(1.0, (i_step + 1) / cfg.lr_warmup_steps)
                for param_group in opt_pi.param_groups:
                    param_group["lr"] = cfg.lr * lr_factor

            sum_reward_batch = 0
            sum_raw_reward_batch = 0
            sum_num_keypoints_1 = 0
            sum_num_keypoints_2 = 0
            loss = None
            loss_policy_stack = None
            loss_desc_stack = None
            loss_kp_stack = None
            loss_value_stack = None
            loss_return_stack = None
            loss_advantage_stack = None
            current_value_loss = None
            sum_inliers = 0

            _dbg("Loading batch from DataLoader...")
            try:
                batch = next(i_dl)
            except StopIteration:
                i_dl = iter(dl)
                batch = next(i_dl)
            _dbg(f"Batch loaded — label={[float(l) for l in batch['label']]}")

            p1, p2, mask_padding_1, mask_padding_2, Hs, label = unpack_batch(batch)
            _dbg(f"Batch unpacked — p1={tuple(p1.shape)} dtype={p1.dtype} device={p1.device}")

            # Move all data explicitly to GPU (Fabric device is not reliable when setup_dataloaders is skipped)
            p1 = p1.to(DEVICE)
            p2 = p2.to(DEVICE)
            mask_padding_1 = mask_padding_1.to(DEVICE)
            mask_padding_2 = mask_padding_2.to(DEVICE)
            _dbg(f"Data moved to {DEVICE} — p1.device={p1.device}")

            # forward pass - extract keypoints & descriptors (PPO: also returns value estimates)
            try:
                _dbg("Forward pass 1 (image1)...")
                (kpts1, logprobs1, selected_mask1, mask_padding_grid_1, logits_selected_1, values1, entropy1, choices1, accepted_choices1, out1) = net(
                    p1, mask_padding_1, training=True
                )
                _dbg(f"Forward 1 done — kpts1={tuple(kpts1.shape)} device={kpts1.device}")

                _dbg("Forward pass 2 (image2)...")
                (kpts2, logprobs2, selected_mask2, mask_padding_grid_2, logits_selected_2, values2, entropy2, choices2, accepted_choices2, out2) = net(
                    p2, mask_padding_2, training=True
                )
                _dbg(f"Forward 2 done — kpts2={tuple(kpts2.shape)}")

                _dbg("Computing descriptors...")
                desc_1 = net.get_descs(out1["coarse_descs"], p1, kpts1, p1.shape[2], p1.shape[3])
                desc_2 = net.get_descs(out2["coarse_descs"], p2, kpts2, p2.shape[2], p2.shape[3])
                _dbg(f"Descriptors done — desc_1={tuple(desc_1.shape)}")
            except Exception as _fwd_err:
                print(f"\n[FORWARD PASS ERROR] {type(_fwd_err).__name__}: {_fwd_err}", flush=True)
                import traceback; traceback.print_exc()
                raise

            if cfg.padding_filter_mode == "ignore":
                # Ensure they are bool for the & operator
                batch_mask_selection_for_matching_1 = selected_mask1.bool() & mask_padding_grid_1.bool()
                batch_mask_selection_for_matching_2 = selected_mask2.bool() & mask_padding_grid_2.bool()
            elif cfg.padding_filter_mode == "punish":
                batch_mask_selection_for_matching_1 = selected_mask1
                batch_mask_selection_for_matching_2 = selected_mask2
            else:
                raise ValueError(f"Unknown padding filter mode: {cfg.padding_filter_mode}")

            # ── Matching ───────────────────────────────────────────────────────
            _dbg("Starting matching...")
            Hs_list = Hs if Hs is not None else [None] * batch_size

            use_synthetic = [
                (Hs_list[b] is not None and not (isinstance(Hs_list[b], torch.Tensor) and Hs_list[b].numel() == 0))
                for b in range(batch_size)
            ]
            _dbg(f"use_synthetic={use_synthetic}")

            if any(use_synthetic):
                # Mixed batch: some synthetic, some real — handle per-sample
                batch_rel_idx_matches = [torch.empty((0, 2), dtype=torch.long, device=DEVICE)] * batch_size
                batch_abs_idx_matches = [torch.empty((0, 2), dtype=torch.long, device=DEVICE)] * batch_size
                batch_ransac_inliers  = [torch.empty(0, dtype=torch.bool, device=DEVICE)] * batch_size
                batch_Fm              = [None] * batch_size

                real_indices = [b for b in range(batch_size) if not use_synthetic[b]]

                # Process real samples through MNN+LO-RANSAC
                if real_indices:
                    r_kpts1   = torch.stack([kpts1[b] for b in real_indices])
                    r_kpts2   = torch.stack([kpts2[b] for b in real_indices])
                    r_desc1   = torch.stack([desc_1[b] for b in real_indices])
                    r_desc2   = torch.stack([desc_2[b] for b in real_indices])
                    r_m1      = torch.stack([batch_mask_selection_for_matching_1[b] for b in real_indices])
                    r_m2      = torch.stack([batch_mask_selection_for_matching_2[b] for b in real_indices])
                    r_lbl     = [label[b] for b in real_indices] if label is not None else None
                    r_scores1 = torch.stack([logits_selected_1[b] for b in real_indices])
                    r_scores2 = torch.stack([logits_selected_2[b] for b in real_indices])
                    r_rel, r_abs, r_inl, r_fm = matcher(
                        r_kpts1, r_kpts2, r_desc1, r_desc2, r_m1, r_m2, inl_th, r_lbl,
                        scores1=r_scores1, scores2=r_scores2,
                    )
                    for i, b in enumerate(real_indices):
                        batch_rel_idx_matches[b] = r_rel[i]
                        batch_abs_idx_matches[b]  = r_abs[i]
                        batch_ransac_inliers[b]   = r_inl[i]
                        batch_Fm[b]               = r_fm[i]

                # Process synthetic samples via H-based inlier computation
                for b in range(batch_size):
                    if not use_synthetic[b]:
                        continue
                    sel1 = batch_mask_selection_for_matching_1[b]
                    sel2 = batch_mask_selection_for_matching_2[b]
                    kpts1_sel = kpts1[b][sel1.bool()]
                    kpts2_sel = kpts2[b][sel2.bool()]
                    abs_m, inl = _synthetic_inliers(kpts1_sel, kpts2_sel, Hs_list[b])
                    # remap from selected-space back to full keypoint indices
                    full_idx1 = torch.where(sel1.bool())[0]
                    full_idx2 = torch.where(sel2.bool())[0]
                    if abs_m.shape[0] > 0:
                        abs_m_full = torch.stack([full_idx1[abs_m[:, 0]], full_idx2[abs_m[:, 1]]], dim=1)
                    else:
                        abs_m_full = abs_m
                    batch_abs_idx_matches[b]  = abs_m_full
                    batch_ransac_inliers[b]   = inl
                    # rel_idx not needed for synthetic (no desc loss indexing via rel_idx)
                    batch_rel_idx_matches[b]  = abs_m_full   # reuse abs idx as rel idx
            else:
                _dbg("Calling MNN+RANSAC matcher (all-real batch)...")
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
                    scores1=logits_selected_1,
                    scores2=logits_selected_2,
                )
                _dbg("Matcher done")

            _dbg("Matching complete — computing rewards...")
            sample_mask_selection_for_matching_1 = []
            sample_mask_selection_for_matching_2 = []
            sample_mask_padding_grid_1 = []
            sample_mask_padding_grid_2 = []

            for b in range(batch_size):
                if batch_rel_idx_matches[b] is None or len(batch_rel_idx_matches[b]) == 0:
                    ma_skipped_batches.append(1)
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

                if cfg.reward_type == "inlier":
                    reward = 0.5 if cfg.no_filtering_negatives and not label[b] else 1.0
                elif cfg.reward_type == "inlier_ratio":
                    ratio_inlier = float(ransac_inliers.sum()) / len(abs_idx_matches) if len(abs_idx_matches) > 0 else 0.0
                    reward = ratio_inlier
                elif cfg.reward_type == "inlier+inlier_ratio":
                    ratio_inlier = float(ransac_inliers.sum()) / len(abs_idx_matches) if len(abs_idx_matches) > 0 else 0.0
                    reward = (1.0 - beta) * 1.0 + beta * ratio_inlier
                elif cfg.reward_type == "balanced":
                    # Balanced reward: inlier rate + log-scaled inlier count only.
                    # kp_density is intentionally excluded: get_rewards() negates reward
                    # for negative pairs, so including kp_density would penalise keypoints
                    # on 50% of batches → directly causes keypoint suppression.
                    # The kp_count_floor regulariser handles anti-suppression separately.
                    num_inliers = float(ransac_inliers.sum()) if ransac_inliers is not None else 0.0
                    num_matches = len(abs_idx_matches) if abs_idx_matches is not None else 0
                    inlier_rate = num_inliers / max(num_matches, 1)
                    inlier_count_score = math.log1p(num_inliers) / math.log1p(50)  # 50 inliers = 1.0
                    reward = 0.5 * inlier_rate + 0.5 * inlier_count_score
                else:
                    raise ValueError(f"Unknown reward type: {cfg.reward_type}")

                sum_raw_reward_batch += float(reward)

                effective_reward_scale = cfg.reward_scale
                if cfg.reward_scale_warmup_steps > 0:
                    effective_reward_scale *= min(1.0, (i_step + 1) / cfg.reward_scale_warmup_steps)

                effective_fp_penalty = fp_penalty
                if cfg.fp_penalty_warmup_steps > 0:
                    effective_fp_penalty *= min(1.0, (i_step + 1) / cfg.fp_penalty_warmup_steps)

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
                    effective_fp_penalty * alpha,
                    use_whitening=cfg.use_whitening,
                    selected_only=cfg.selected_only,
                    filter_mode=cfg.padding_filter_mode,
                ) * effective_reward_scale

                batch_mean = dense_rewards.mean().item()
                batch_var  = dense_rewards.var(unbiased=False).item()

                # EMA normalizer — uses reward_norm_momentum from config (was unused before)
                m = cfg.reward_norm_momentum
                if not reward_norm_ema_init:
                    reward_norm_mean_ema = batch_mean
                    reward_norm_var_ema  = max(batch_var, cfg.reward_norm_eps)
                    reward_norm_ema_init = True
                else:
                    reward_norm_mean_ema = m * reward_norm_mean_ema + (1 - m) * batch_mean
                    reward_norm_var_ema  = m * reward_norm_var_ema  + (1 - m) * batch_var

                reward_norm_std = math.sqrt(reward_norm_var_ema + cfg.reward_norm_eps)
                dense_rewards = (dense_rewards - reward_norm_mean_ema) / reward_norm_std

                if descriptor_loss is not None:
                    _anneal_steps = getattr(cfg, 'desc_margin_anneal_steps', 1000)
                    if hasattr(descriptor_loss, 'pos_margin'):
                        descriptor_loss.pos_margin = 2.0 - 1.0 * min(1.0, i_step / max(_anneal_steps, 1))
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
                        hard_loss.unsqueeze(0) if loss_desc_stack is None else torch.cat((loss_desc_stack, hard_loss.unsqueeze(0)))
                    )

                sum_reward_batch += dense_rewards.sum()
                sum_inliers += ransac_inliers.sum().item() if ransac_inliers is not None else 0

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

                loss_policy_stack = (
                    dense_logprobs.detach().view(-1)
                    if loss_policy_stack is None
                    else torch.cat((loss_policy_stack, dense_logprobs.detach().view(-1)))
                )
                loss_value_stack = (
                    dense_values.detach().view(-1)
                    if loss_value_stack is None
                    else torch.cat((loss_value_stack, dense_values.detach().view(-1)))
                )
                loss_return_stack = (
                    dense_rewards.detach().view(-1)
                    if loss_return_stack is None
                    else torch.cat((loss_return_stack, dense_rewards.detach().view(-1)))
                )

                sample_mask_selection_for_matching_1.append(mask_selection_for_matching_1)
                sample_mask_selection_for_matching_2.append(mask_selection_for_matching_2)
                sample_mask_padding_grid_1.append(mask_padding_grid_1[b] if mask_padding_grid_1 is not None else None)
                sample_mask_padding_grid_2.append(mask_padding_grid_2[b] if mask_padding_grid_2 is not None else None)

                effective_kp_penalty = kp_penalty
                if cfg.kp_penalty_warmup_steps > 0:
                    effective_kp_penalty *= min(1.0, (i_step + 1) / cfg.kp_penalty_warmup_steps)

                if effective_kp_penalty != 0.0:
                    loss_kp = (
                        logprobs1[b][mask_selection_for_matching_1]
                        * torch.full_like(logprobs1[b][mask_selection_for_matching_1], effective_kp_penalty * alpha)
                    ).mean() + (
                        logprobs2[b][mask_selection_for_matching_2]
                        * torch.full_like(logprobs2[b][mask_selection_for_matching_2], effective_kp_penalty * alpha)
                    ).mean()
                    loss_kp_stack = loss_kp.unsqueeze(0) if loss_kp_stack is None else torch.cat((loss_kp_stack, loss_kp.unsqueeze(0)))

                sum_num_keypoints_1 += mask_selection_for_matching_1.sum()
                sum_num_keypoints_2 += mask_selection_for_matching_2.sum()

            # ====== Keypoint Count Floor Regularizer (Fix 2) ======
            # Penalize the policy when detected keypoints fall below a minimum
            # threshold — directly counteracts the suppression bias.
            loss_kp_count = torch.tensor(0.0, device=DEVICE)
            kp_count_floor = getattr(cfg, 'kp_count_floor', 380)
            kp_count_penalty = getattr(cfg, 'kp_count_penalty', 0.5)
            kp_count_warmup = getattr(cfg, 'kp_count_warmup_steps', 500)
            if kp_count_floor > 0 and i_step >= kp_count_warmup:
                avg_kp_1 = sum_num_keypoints_1.float() / batch_size
                avg_kp_2 = sum_num_keypoints_2.float() / batch_size
                # Asymmetric quadratic penalty: only when BELOW floor
                if avg_kp_1 < kp_count_floor:
                    deficit_1 = (kp_count_floor - avg_kp_1) / kp_count_floor
                    loss_kp_count = loss_kp_count + kp_count_penalty * (deficit_1 ** 2)
                if avg_kp_2 < kp_count_floor:
                    deficit_2 = (kp_count_floor - avg_kp_2) / kp_count_floor
                    loss_kp_count = loss_kp_count + kp_count_penalty * (deficit_2 ** 2)

            # ====== Descriptor Warmup Phase (Fix 4) ======
            # During warmup, only train the descriptor head. PPO is skipped.
            descriptor_warmup_steps = getattr(cfg, 'descriptor_warmup_steps', 0)
            is_warmup = i_step < descriptor_warmup_steps

            if is_warmup:
                # Warmup: train descriptor head only, skip PPO entirely
                warmup_loss = torch.tensor(0.0, device=DEVICE)
                if loss_desc_stack is not None:
                    warmup_loss = warmup_loss + cfg.desc_loss_weight * loss_desc_stack.mean()
                # Also apply kp count regularizer during warmup
                if loss_kp_count.item() > 0:
                    warmup_loss = warmup_loss + loss_kp_count
                if isinstance(warmup_loss, torch.Tensor) and warmup_loss.requires_grad:
                    fabric.backward(warmup_loss)
                    fabric.clip_gradients(net, optimizer=opt_pi, max_norm=cfg.gradient_clip_norm)
                    opt_pi.step()
                    opt_pi.zero_grad()
                loss = warmup_loss.detach() if isinstance(warmup_loss, torch.Tensor) else torch.tensor(0.0, device=DEVICE)
                # Skip rollout buffer and PPO update
            else:
                # ====== Normal Training: Aux losses + PPO ======
                _dbg("Computing aux loss (desc + kp)...")
                aux_loss = torch.tensor(0.0, device=DEVICE)
                if loss_desc_stack is not None:
                    aux_loss = aux_loss + cfg.desc_loss_weight * loss_desc_stack.mean()
                if loss_kp_stack is not None:
                    aux_loss = aux_loss + loss_kp_stack.mean()
                if loss_kp_count.item() > 0:
                    aux_loss = aux_loss + loss_kp_count

                if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad:
                    _dbg(f"Backward aux_loss={aux_loss.item():.4f}...")
                    fabric.backward(aux_loss)
                    fabric.clip_gradients(net, optimizer=opt_pi, max_norm=cfg.gradient_clip_norm)
                    opt_pi.step()
                    opt_pi.zero_grad()
                _dbg("Aux loss step done")

                if loss_return_stack is not None:
                    loss_advantage_stack = loss_return_stack - loss_value_stack.detach()
                    loss_advantage_stack = (loss_advantage_stack - loss_advantage_stack.mean()) / (loss_advantage_stack.std() + 1e-8)
                else:
                    loss_advantage_stack = None

                # Store experience in rollout buffer only if matches found
                if loss_policy_stack is not None:
                    rollout_buffer.append({
                        'logprobs': loss_policy_stack,
                        'values': loss_value_stack,
                        'rewards': loss_return_stack,
                        'advantages': loss_advantage_stack,
                        'desc_loss': None,
                        'kp_loss': None,
                        'p1': p1.cpu() if p1 is not None else None,
                        'p2': p2.cpu() if p2 is not None else None,
                        'choices1': choices1,
                        'choices2': choices2,
                        'accepted_choices1': accepted_choices1,
                        'accepted_choices2': accepted_choices2,
                        'mask_padding_1': mask_padding_1,
                        'mask_padding_2': mask_padding_2,
                        'sample_mask_selection_for_matching_1': sample_mask_selection_for_matching_1,
                        'sample_mask_selection_for_matching_2': sample_mask_selection_for_matching_2,
                        'sample_mask_padding_grid_1': sample_mask_padding_grid_1,
                        'sample_mask_padding_grid_2': sample_mask_padding_grid_2,
                    })

                _dbg(f"Rollout buffer size={len(rollout_buffer)}/{cfg.rollout_steps}")
                # If rollout buffer is full, compute returns and advantages, then update
                if len(rollout_buffer) >= cfg.rollout_steps:
                    _dbg("PPO update starting...")
                    # Compute discounted returns and GAE (backwards)
                    for i in reversed(range(len(rollout_buffer))):
                        rewards = rollout_buffer[i]['rewards']
                        values = rollout_buffer[i]['values']
                        
                        if i == len(rollout_buffer) - 1:
                            next_values = torch.zeros_like(values)
                            next_advantages = torch.zeros_like(rewards)
                        else:
                            next_values = rollout_buffer[i+1]['values']
                            next_advantages = rollout_buffer[i+1]['advantages']
                        
                        # TD error
                        delta = rewards + cfg.gamma * next_values - values
                        
                        # GAE
                        advantages = delta + cfg.gae_lambda * cfg.gamma * next_advantages
                        
                        # Returns for value function
                        returns = advantages + values
                        
                        rollout_buffer[i]['advantages'] = advantages
                        rollout_buffer[i]['returns'] = returns

                    # PPO update
                    old_logprobs_all = torch.cat([exp['logprobs'] for exp in rollout_buffer]).detach()
                    old_advantages_all = torch.cat([exp['advantages'] for exp in rollout_buffer]).detach()
                    old_returns_all = torch.cat([exp['returns'] for exp in rollout_buffer]).detach()

                    opt_pi.zero_grad()
                    last_loss = None

                    for epoch in range(cfg.ppo_epochs):
                        new_logprobs_all = []
                        new_values_all = []
                        entropy_terms = []

                        for exp in rollout_buffer:
                            p1_dev = exp['p1'].to(DEVICE)
                            p2_dev = exp['p2'].to(DEVICE)
                            
                            logprobs1_new, values1_new, entropy1_new, _ = net.evaluate_actions(
                                p1_dev, exp['choices1'], exp['accepted_choices1'], exp['mask_padding_1']
                            )
                            logprobs2_new, values2_new, entropy2_new, _ = net.evaluate_actions(
                                p2_dev, exp['choices2'], exp['accepted_choices2'], exp['mask_padding_2']
                            )

                            new_logprobs = []
                            new_values = []

                            for b in range(batch_size):
                                if exp['sample_mask_selection_for_matching_1'][b] is None:
                                    continue

                                if cfg.selected_only:
                                    dense_logprobs_new = (
                                        logprobs1_new[b][exp['sample_mask_selection_for_matching_1'][b].bool()].view(-1, 1)
                                        + logprobs2_new[b][exp['sample_mask_selection_for_matching_2'][b].bool()].view(1, -1)
                                    )
                                    dense_values_new = (
                                        values1_new[b][exp['sample_mask_selection_for_matching_1'][b].bool()].view(-1, 1)
                                        + values2_new[b][exp['sample_mask_selection_for_matching_2'][b].bool()].view(1, -1)
                                    )
                                else:
                                    if cfg.padding_filter_mode == "ignore":
                                        dense_logprobs_new = (
                                            logprobs1_new[b][exp['sample_mask_padding_grid_1'][b].bool()].view(-1, 1)
                                            + logprobs2_new[b][exp['sample_mask_padding_grid_2'][b].bool()].view(1, -1)
                                        )
                                        dense_values_new = (
                                            values1_new[b][exp['sample_mask_padding_grid_1'][b].bool()].view(-1, 1)
                                            + values2_new[b][exp['sample_mask_padding_grid_2'][b].bool()].view(1, -1)
                                        )
                                    elif cfg.padding_filter_mode == "punish":
                                        dense_logprobs_new = logprobs1_new[b].view(-1, 1) + logprobs2_new[b].view(1, -1)
                                        dense_values_new = values1_new[b].view(-1, 1) + values2_new[b].view(1, -1)
                                    else:
                                        raise ValueError(f"Unknown padding filter mode: {cfg.padding_filter_mode}")

                                new_logprobs.append(dense_logprobs_new.view(-1))
                                new_values.append(dense_values_new.view(-1))
                                entropy_terms.append((entropy1_new[b].mean() + entropy2_new[b].mean()) * 0.5)

                            if len(new_logprobs) > 0:
                                new_logprobs_all.extend(new_logprobs)
                                new_values_all.extend(new_values)

                        if len(new_logprobs_all) == 0:
                            break

                        new_logprobs_all = torch.cat(new_logprobs_all)
                        new_values_all = torch.cat(new_values_all)
                        entropy_bonus = torch.stack(entropy_terms).mean()

                        ratio = torch.exp(new_logprobs_all - old_logprobs_all)
                        # Tighten clip after first 10% of training (8k/80k) for stability
                        current_clip_eps = 0.05 if i_step >= (cfg.num_steps // 10) else cfg.ppo_clip_eps
                        clipped_ratio = torch.clamp(ratio, 1.0 - current_clip_eps, 1.0 + current_clip_eps)
                        policy_loss = -torch.min(ratio * old_advantages_all, clipped_ratio * old_advantages_all).mean()
                        value_loss = F.mse_loss(new_values_all, old_returns_all)

                        effective_entropy_coef = cfg.entropy_coef
                        if cfg.entropy_coef_warmup_steps > 0:
                            effective_entropy_coef *= min(1.0, (i_step + 1) / cfg.entropy_coef_warmup_steps)

                        effective_value_loss_coef = cfg.value_loss_coef
                        if cfg.value_loss_coef_warmup_steps > 0:
                            effective_value_loss_coef *= min(1.0, (i_step + 1) / cfg.value_loss_coef_warmup_steps)

                        epoch_loss = policy_loss + effective_value_loss_coef * value_loss - effective_entropy_coef * entropy_bonus

                        fabric.backward(epoch_loss)
                        fabric.clip_gradients(net, optimizer=opt_pi, max_norm=cfg.gradient_clip_norm)
                        opt_pi.step()
                        opt_pi.zero_grad()

                        last_loss = epoch_loss.detach()

                    loss = last_loss if last_loss is not None else torch.tensor(0.0, device=DEVICE)
                    # Clear rollout buffer
                    rollout_buffer = []
                    gc.collect()

                else:
                    loss = torch.tensor(0.0, device=DEVICE)  # No update yet

            # ====== Periodic Checkpoint Saving (Fix 5) ======
            checkpoint_interval = getattr(cfg, 'checkpoint_interval', 5000)
            if checkpoint_interval > 0 and (i_step + 1) % checkpoint_interval == 0:
                raw_state = net.state_dict()
                clean_state = {k.replace("_forward_module.", ""): v for k, v in raw_state.items()}
                ckpt_path = output_dir / f"checkpoint_step_{i_step+1}.pt"
                torch.save(clean_state, ckpt_path)
                write_result(f"[Checkpoint] Saved to {ckpt_path}")

            _dbg(f"---- Step {i_step+1} COMPLETE ----")
            # logging
            reward_value = sum_raw_reward_batch / batch_size
            det_1 = int((sum_num_keypoints_1.item() if hasattr(sum_num_keypoints_1, "item") else sum_num_keypoints_1) / batch_size)
            det_2 = int((sum_num_keypoints_2.item() if hasattr(sum_num_keypoints_2, "item") else sum_num_keypoints_2) / batch_size)
            inliers_avg = sum_inliers / batch_size
            desc_l = loss_desc_stack.mean().item() if loss_desc_stack is not None else 0.0
            kp_count_l = loss_kp_count.item() if isinstance(loss_kp_count, torch.Tensor) else 0.0
            phase = "WARMUP" if is_warmup else "PPO"
            
            write_result(f"[Step {i_step+1}/{steps}] [{phase}] PPO Loss: {loss.item():.4f}, Desc Loss: {desc_l:.4f}, KP Floor: {kp_count_l:.4f}, Reward: {reward_value:.4f}, Inliers: {inliers_avg:.1f}, Det: ({det_1}, {det_2})")
            pbar.update(1)
            pbar.set_postfix({
                "R": f"{reward_value:.3f}",
                "inl": f"{inliers_avg:.0f}",
                "ppo": f"{loss.item():.3f}",
                "desc": f"{desc_l:.3f}",
            })

            if i_step % cfg.log_interval == 0:
                det1_log = int(sum_num_keypoints_1.item() / batch_size) if hasattr(sum_num_keypoints_1, "item") else int(sum_num_keypoints_1 / batch_size)
                det2_log = int(sum_num_keypoints_2.item() / batch_size) if hasattr(sum_num_keypoints_2, "item") else int(sum_num_keypoints_2 / batch_size)
                reward_log = (sum_reward_batch.item() if hasattr(sum_reward_batch, "item") else float(sum_reward_batch)) / batch_size
                
                wandb_logger.log(
                    {
                        "loss": loss.item(),
                        "loss_policy": -loss_policy_stack.mean().item() if loss_policy_stack is not None else 0.0,
                        "loss_kp": loss_kp_stack.mean().item() if loss_kp_stack is not None else 0.0,
                        "loss_hard": (loss_desc_stack.mean().item() if loss_desc_stack is not None else 0.0),
                        "mean_num_det_kpts1": det1_log,
                        "mean_num_det_kpts2": det2_log,
                        "mean_reward": reward_log,
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
    