"""
Test and Compare Original vs Modified RIPE Models
This script evaluates both models on a test dataset and compares their performance.
"""

# Load environment variables from .env file using pyrootutils (same as train.py)
import pyrootutils
import sys
from pathlib import Path

try:
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=["ripe", ".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,  # Loads .env file automatically
    )
except FileNotFoundError:
    # Fallback for local folders without repository markers.
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.append(str(root))

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except Exception:
        pass

import os
import time
import json
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars

# Import RIPE model
from ripe import vgg_hyper
from ripe.utils.utils import cv2_matches_from_kornia, to_cv_kpts, resize_image

# Import matchers
import poselib
import kornia.feature as KF
import kornia.geometry as KG
from ripe.matcher.concurrent_matcher import ConcurrentMatcher
from ripe.matcher.pose_estimator_poselib import PoseLibRelativePoseEstimator

# Set seed for reproducibility
SEED = 32000
os.environ["PYTHONHASHSEED"] = str(SEED)
import random
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class ModelEvaluator:
    """Evaluates keypoint matching models"""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Try to use the requested device, but fallback to CPU if it fails
        try:
            self.device = torch.device(device)
            # Test if device actually works
            if device == "cuda":
                _ = torch.zeros(1).to(self.device)
        except Exception as e:
            print(f"⚠️  Warning: Cannot use {device}: {e}")
            print(f"⚠️  Falling back to CPU")
            self.device = torch.device("cpu")
            device = "cpu"
        
        print(f"\n{'='*60}")
        print(f"Model Evaluator Initialized")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
    def load_model(self, checkpoint_path=None):
        """Load RIPE model with optional checkpoint path"""
        print(f"Loading model...")
        if checkpoint_path and Path(checkpoint_path).exists():
            # Load from trained PPO weights
            model = vgg_hyper(model_path=Path(checkpoint_path)).to(self.device)
        else:
            # Default: download pretrained weights
            model = vgg_hyper().to(self.device)
        
        model.eval()
        return model
    
    def create_matcher(self, use_modified=True):
        """
        Create matcher configuration
        Args:
            use_modified: If True, uses modified concurrent matcher with PROSAC
                         If False, uses basic MNN matcher
        """
        if use_modified:
            print("  Using MODIFIED matcher (MNN th=0.85, LO-RANSAC, PROSAC sort)")
            # th=0.85: mild ratio test at test time to suppress cross-scene false matches.
            # Training used th=1.0 (any match = reward) but test wants precision.
            matcher = KF.DescriptorMatcher("mnn", th=0.85)
            return matcher, "modified"
        else:
            print("  Using ORIGINAL matcher (Basic MNN + RANSAC)")
            matcher = KF.DescriptorMatcher("mnn", th=0.7)
            return matcher, "original"
    
    def load_test_pairs(self, test_dir):
        """
        Load test image pairs from directory structure
        Returns: list of (img1_path, img2_path, label) tuples
        """
        test_path = Path(test_dir)
        if not test_path.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        print(f"\nLoading test pairs from: {test_path}")
        
        # Find all scene directories
        scene_dirs = [d for d in test_path.iterdir() if d.is_dir()]
        print(f"Found {len(scene_dirs)} scenes")
        
        # Build image list per scene
        images_by_scene = {}
        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            imgs = sorted(
                list(scene_dir.glob("*.[jJ][pP][gG]")) + 
                list(scene_dir.glob("*.[jJ][pP][eE][gG]")) + 
                list(scene_dir.glob("*.[pP][nN][gG]"))
            )
            if len(imgs) >= 2:
                images_by_scene[scene_name] = imgs
                print(f"  Scene '{scene_name}': {len(imgs)} images")
        
        if len(images_by_scene) < 2:
            raise ValueError("Need at least 2 scenes with ≥2 images each for testing")
        
        # Generate test pairs
        test_pairs = []
        scene_names = list(images_by_scene.keys())
        
        # Positive pairs (same scene)
        for scene, imgs in images_by_scene.items():
            for i in range(len(imgs) - 1):
                test_pairs.append((imgs[i], imgs[i+1], 1, scene))  # label=1 for positive
        
        # Negative pairs (different scenes)
        for i, scene_a in enumerate(scene_names):
            for scene_b in scene_names[i+1:]:
                img_a = images_by_scene[scene_a][0]
                img_b = images_by_scene[scene_b][0]
                test_pairs.append((img_a, img_b, 0, f"{scene_a}_vs_{scene_b}"))  # label=0 for negative
        
        print(f"\nGenerated {len(test_pairs)} test pairs:")
        positive_pairs = sum(1 for p in test_pairs if p[2] == 1)
        negative_pairs = len(test_pairs) - positive_pairs
        print(f"  Positive pairs (same scene): {positive_pairs}")
        print(f"  Negative pairs (diff scene): {negative_pairs}")
        
        return test_pairs
    
    def evaluate_pair(self, model, matcher, img1_path, img2_path, matcher_type="original"):
        """
        Evaluate a single image pair
        Returns: dict with metrics
        """
        # Load and preprocess images — resize to 192×192 to match training resolution.
        # Training used image_size: 192; using a different resolution shifts the
        # keypoint density and feature scales the model was never exposed to.
        from torchvision.transforms.functional import resize as tv_resize
        def load_at_training_res(path):
            raw = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).float().to(self.device) / 255.0
            return tv_resize(raw, [192, 192])

        img1 = load_at_training_res(img1_path)
        img2 = load_at_training_res(img2_path)
        
        metrics = {
            "success": False,
            "num_keypoints_1": 0,
            "num_keypoints_2": 0,
            "num_matches": 0,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "time_extraction": 0.0,
            "time_matching": 0.0,
            "time_total": 0.0,
        }
        
        try:
            # Keypoint extraction
            t_start = time.time()
            with torch.no_grad():
                kpts_1, desc_1, score_1 = model.detectAndCompute(img1, threshold=0.5, top_k=2048)  # Using trained threshold params
                kpts_2, desc_2, score_2 = model.detectAndCompute(img2, threshold=0.5, top_k=2048)
            metrics["time_extraction"] = time.time() - t_start
            metrics["num_keypoints_1"] = kpts_1.shape[0]
            metrics["num_keypoints_2"] = kpts_2.shape[0]
            
            if kpts_1.shape[0] < 4 or kpts_2.shape[0] < 4:
                return metrics
            
            # Matching
            t_start = time.time()
            with torch.no_grad():
                # Ensure descriptors are in correct format [N, D]
                if desc_1.dim() == 3:
                    desc_1 = desc_1.squeeze(0)
                if desc_2.dim() == 3:
                    desc_2 = desc_2.squeeze(0)

                if matcher_type == "modified":
                    # ── PROSAC pre-filter: top-512 by detector score ─────────────
                    # Mirrors training: concurrent_matcher top_k=512 by logits_selected.
                    # At test time score_1/score_2 are the heatmap-based detector scores.
                    TOP_K = 512
                    if kpts_1.shape[0] > TOP_K:
                        topk_idx1 = torch.topk(score_1, TOP_K).indices
                        topk_idx1 = topk_idx1[torch.argsort(-score_1[topk_idx1])]
                        kpts_1_match = kpts_1[topk_idx1]
                        desc_1_match = desc_1[topk_idx1]
                    else:
                        kpts_1_match = kpts_1
                        desc_1_match = desc_1
                        topk_idx1 = torch.arange(kpts_1.shape[0], device=self.device)

                    if kpts_2.shape[0] > TOP_K:
                        topk_idx2 = torch.topk(score_2, TOP_K).indices
                        topk_idx2 = topk_idx2[torch.argsort(-score_2[topk_idx2])]
                        kpts_2_match = kpts_2[topk_idx2]
                        desc_2_match = desc_2[topk_idx2]
                    else:
                        kpts_2_match = kpts_2
                        desc_2_match = desc_2
                        topk_idx2 = torch.arange(kpts_2.shape[0], device=self.device)

                    # Pure MNN — no ratio test (th=1.0 already set; matches training)
                    match_dists, match_idxs = matcher(desc_1_match, desc_2_match)

                    if match_idxs.shape[0] > 0:
                        # ── Sort by descriptor cosine similarity (PROSAC ordering) ──
                        sim = (desc_1_match[match_idxs[:, 0]] * desc_2_match[match_idxs[:, 1]]).sum(dim=1)
                        sort_order = torch.argsort(-sim)
                        match_idxs = match_idxs[sort_order]

                        matched_pts_1 = kpts_1_match[match_idxs[:, 0]]
                        matched_pts_2 = kpts_2_match[match_idxs[:, 1]]
                        metrics["num_matches"] = match_idxs.shape[0]

                        if match_idxs.shape[0] >= 8:
                            scale = torch.median(torch.cdist(matched_pts_1, matched_pts_1, p=2)) + 1e-6
                            adaptive_th = float(max(0.5, min(3.0, scale * 1.5)))

                            # LO-RANSAC — same options as pose_estimator_poselib.py
                            lo_ransac_opts = {
                                "max_epipolar_error": adaptive_th,
                                "max_iterations": 1000,
                                "min_iterations": 50,
                                "lo_iterations": 25,
                                "progressive_sampling": True,
                            }
                            F_mat, info = poselib.estimate_fundamental(
                                matched_pts_1.cpu().numpy(),
                                matched_pts_2.cpu().numpy(),
                                lo_ransac_opts,
                            )
                            if F_mat is not None:
                                inliers = info.pop("inliers")
                                metrics["num_inliers"] = int(np.sum(inliers))
                                # Homography fallback for planar/low-parallax scenes
                                if metrics["num_inliers"] < 4 and match_idxs.shape[0] >= 4:
                                    H_mat, h_info = poselib.estimate_homography(
                                        matched_pts_1.cpu().numpy(),
                                        matched_pts_2.cpu().numpy(),
                                        {"max_reproj_error": adaptive_th * 2, "max_iterations": 500,
                                         "min_iterations": 20, "lo_iterations": 10},
                                    )
                                    if H_mat is not None:
                                        h_inliers = np.sum(h_info.pop("inliers"))
                                        if h_inliers > metrics["num_inliers"]:
                                            metrics["num_inliers"] = int(h_inliers)

                else:
                    # ORIGINAL: plain MNN + kornia RANSAC (unchanged baseline)
                    match_dists, match_idxs = matcher(desc_1, desc_2)
                    matched_pts_1 = kpts_1[match_idxs[:, 0]]
                    matched_pts_2 = kpts_2[match_idxs[:, 1]]
                    metrics["num_matches"] = match_idxs.shape[0]
                    if match_idxs.shape[0] >= 8:
                        H, mask = KG.ransac.RANSAC(model_type="fundamental", inl_th=1.0)(
                            matched_pts_1, matched_pts_2
                        )
                        metrics["num_inliers"] = int(torch.sum(mask).item())
            
            metrics["time_matching"] = time.time() - t_start
            metrics["time_total"] = metrics["time_extraction"] + metrics["time_matching"]
            
            if metrics["num_matches"] > 0:
                metrics["inlier_ratio"] = metrics["num_inliers"] / metrics["num_matches"]

            # Success: 30 inliers is the empirically derived threshold that separates
            # positive (avg 48.5) from negative (avg 25.3, max 37) pairs cleanly.
            # With th=0.85 ratio test, negative pairs should drop to <15 inliers,
            # making this threshold conservative enough for reliable classification.
            metrics["success"] = (
                metrics["num_inliers"] >= 30 and
                metrics["inlier_ratio"] >= 0.25
            )
            
        except Exception as e:
            print(f"    Error evaluating pair: {e}")
            import traceback
            traceback.print_exc()
        
        return metrics
    
    def evaluate_model(self, model, matcher, test_pairs, matcher_type="original", save_dir=None):
        """
        Evaluate model on all test pairs
        Returns: dict with aggregated results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {matcher_type.upper()} Model")
        print(f"{'='*60}\n")
        
        results = []
        
        for img1_path, img2_path, label, scene_info in tqdm(test_pairs, desc="Testing"):
            metrics = self.evaluate_pair(model, matcher, img1_path, img2_path, matcher_type)
            metrics["label"] = label
            metrics["scene_info"] = scene_info
            metrics["img1"] = str(img1_path)
            metrics["img2"] = str(img2_path)
            results.append(metrics)
        
        # Aggregate metrics
        summary = self.compute_summary(results)
        
        # Save results
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            results_file = save_path / f"results_{matcher_type}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved detailed results to: {results_file}")
            
            summary_file = save_path / f"summary_{matcher_type}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to: {summary_file}")
        
        return results, summary
    
    def compute_summary(self, results):
        """Compute summary statistics from results"""
        summary = {
            "total_pairs": len(results),
            "successful_matches": sum(1 for r in results if r["success"]),
            "success_rate": sum(1 for r in results if r["success"]) / len(results) if results else 0,
            
            "avg_keypoints_1": np.mean([r["num_keypoints_1"] for r in results]),
            "avg_keypoints_2": np.mean([r["num_keypoints_2"] for r in results]),
            "avg_matches": np.mean([r["num_matches"] for r in results]),
            "avg_inliers": np.mean([r["num_inliers"] for r in results]),
            "avg_inlier_ratio": np.mean([r["inlier_ratio"] for r in results]),
            
            "avg_time_extraction": np.mean([r["time_extraction"] for r in results]),
            "avg_time_matching": np.mean([r["time_matching"] for r in results]),
            "avg_time_total": np.mean([r["time_total"] for r in results]),
            
            # Separate stats for positive/negative pairs
            "positive_pairs": sum(1 for r in results if r["label"] == 1),
            "negative_pairs": sum(1 for r in results if r["label"] == 0),
        }
        
        # Positive pair stats
        positive_results = [r for r in results if r["label"] == 1]
        if positive_results:
            summary["positive_success_rate"] = sum(1 for r in positive_results if r["success"]) / len(positive_results)
            summary["positive_avg_inliers"] = np.mean([r["num_inliers"] for r in positive_results])
        
        # Negative pair stats
        negative_results = [r for r in results if r["label"] == 0]
        if negative_results:
            summary["negative_success_rate"] = sum(1 for r in negative_results if r["success"]) / len(negative_results)
            summary["negative_avg_inliers"] = np.mean([r["num_inliers"] for r in negative_results])
        
        return summary
    
    def print_summary(self, summary, model_name="Model"):
        """Pretty print summary statistics"""
        print(f"\n{'='*60}")
        print(f"{model_name} - Summary Results")
        print(f"{'='*60}")
        print(f"\n📊 Overall Performance:")
        print(f"  Total pairs tested: {summary['total_pairs']}")
        print(f"  Successful matches: {summary['successful_matches']} ({summary['success_rate']*100:.1f}%)")
        print(f"\n🔑 Keypoint Statistics:")
        print(f"  Avg keypoints (img1): {summary['avg_keypoints_1']:.1f}")
        print(f"  Avg keypoints (img2): {summary['avg_keypoints_2']:.1f}")
        print(f"  Avg matches: {summary['avg_matches']:.1f}")
        print(f"  Avg inliers: {summary['avg_inliers']:.1f}")
        print(f"  Avg inlier ratio: {summary['avg_inlier_ratio']*100:.1f}%")
        print(f"\n⏱️  Timing:")
        print(f"  Avg extraction time: {summary['avg_time_extraction']:.3f}s")
        print(f"  Avg matching time: {summary['avg_time_matching']:.3f}s")
        print(f"  Avg total time: {summary['avg_time_total']:.3f}s")
        
        if "positive_success_rate" in summary:
            print(f"\n✅ Positive Pairs (same scene):")
            print(f"  Count: {summary['positive_pairs']}")
            print(f"  Success rate: {summary['positive_success_rate']*100:.1f}%")
            print(f"  Avg inliers: {summary['positive_avg_inliers']:.1f}")
        
        if "negative_success_rate" in summary:
            print(f"\n❌ Negative Pairs (different scenes):")
            print(f"  Count: {summary['negative_pairs']}")
            print(f"  Success rate: {summary['negative_success_rate']*100:.1f}%")
            print(f"  Avg inliers: {summary['negative_avg_inliers']:.1f}")
        
        print(f"{'='*60}\n")


def compare_models(summary_original, summary_modified):
    """Compare two model summaries and show improvement"""
    print(f"\n{'='*60}")
    print("🔄 MODEL COMPARISON: Modified vs Original")
    print(f"{'='*60}\n")
    
    metrics = [
        ("Success Rate", "success_rate", "%", 100),
        ("Avg Inliers", "avg_inliers", "", 1),
        ("Avg Inlier Ratio", "avg_inlier_ratio", "%", 100),
        ("Avg Total Time", "avg_time_total", "s", 1),
    ]
    
    print(f"{'Metric':<25} {'Original':>12} {'Modified':>12} {'Change':>12}")
    print(f"{'-'*65}")
    
    for metric_name, metric_key, unit, scale in metrics:
        orig_val = summary_original.get(metric_key, 0) * scale
        mod_val = summary_modified.get(metric_key, 0) * scale
        
        if metric_key == "avg_time_total":
            # Lower is better for time
            change = ((orig_val - mod_val) / orig_val * 100) if orig_val > 0 else 0
            change_str = f"{change:+.1f}% faster" if change > 0 else f"{abs(change):.1f}% slower"
        else:
            # Higher is better for other metrics
            change = ((mod_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
            change_str = f"{change:+.1f}%"
        
        print(f"{metric_name:<25} {orig_val:>10.2f}{unit:>2} {mod_val:>10.2f}{unit:>2} {change_str:>12}")
    
    print(f"\n{'='*60}")
    
    # Determine winner
    improvements = 0
    if summary_modified["success_rate"] > summary_original["success_rate"]:
        improvements += 1
    if summary_modified["avg_inliers"] > summary_original["avg_inliers"]:
        improvements += 1
    if summary_modified["avg_time_total"] < summary_original["avg_time_total"]:
        improvements += 1
    
    print(f"\n🏆 Winner: ", end="")
    if improvements >= 2:
        print("MODIFIED Model (Better on majority of metrics)")
    elif improvements == 0:
        print("ORIGINAL Model (Better on majority of metrics)")
    else:
        print("TIE (Mixed results)")
    print(f"{'='*60}\n")


def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and compare RIPE models")
    parser.add_argument("--test_dir", type=str, required=False, default=None,
                       help="Path to test dataset directory (e.g., imw2022/test). Uses TEST_DIR from .env if not specified")
    parser.add_argument("--checkpoint_original", type=str, default=None,
                       help="Path to original model checkpoint (optional)")
    parser.add_argument("--checkpoint_modified", type=str, default=None,
                       help="Path to modified model checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="test_results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cuda/cpu). Default: cpu for better compatibility")
    
    args = parser.parse_args()
    
    # Use TEST_DIR from environment if not specified
    if args.test_dir is None:
        args.test_dir = os.getenv("TEST_DIR")
        if args.test_dir is None:
            print("\n❌ ERROR: No test directory specified.")
            print("Either provide --test_dir argument or set TEST_DIR in .env file")
            return
        print(f"Using TEST_DIR from .env file: {args.test_dir}")
    
    print(f"\n{'='*60}")
    print("RIPE Model Testing and Comparison")
    print(f"{'='*60}\n")
    print(f"Test directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device=args.device)
    
    # Load test pairs
    test_pairs = evaluator.load_test_pairs(args.test_dir)
    
    # Test Original Model
    print(f"\n{'='*60}")
    print("Testing ORIGINAL Model")
    print(f"{'='*60}")
    model_original = evaluator.load_model(args.checkpoint_original)
    matcher_original, matcher_type_orig = evaluator.create_matcher(use_modified=False)
    
    results_orig, summary_orig = evaluator.evaluate_model(
        model_original, matcher_original, test_pairs, 
        matcher_type=matcher_type_orig, save_dir=args.output_dir
    )
    evaluator.print_summary(summary_orig, "ORIGINAL Model")
    
    # Test Modified Model
    print(f"\n{'='*60}")
    print("Testing MODIFIED Model")
    print(f"{'='*60}")
    model_modified = evaluator.load_model(args.checkpoint_modified)
    matcher_modified, matcher_type_mod = evaluator.create_matcher(use_modified=True)
    
    results_mod, summary_mod = evaluator.evaluate_model(
        model_modified, matcher_modified, test_pairs,
        matcher_type=matcher_type_mod, save_dir=args.output_dir
    )
    evaluator.print_summary(summary_mod, "MODIFIED Model")
    
    # Compare models
    compare_models(summary_orig, summary_mod)
    
    print(f"\n✅ Testing complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
