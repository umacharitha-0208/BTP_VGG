"""
Test Modified Model Only - Quick Test Script
"""

# Load environment variables
import pyrootutils
import sys
from pathlib import Path

try:
    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )
except FileNotFoundError:
    # Fallback when repository markers are missing.
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.append(str(root))

    # Keep .env behavior consistent with pyrootutils setup.
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except Exception:
        pass

import os
import glob

# Import from RIPE package
try:
    from RIPE.test_models import ModelEvaluator
except ModuleNotFoundError:
    repo_root_dir = Path(__file__).resolve().parent
    if str(repo_root_dir) not in sys.path:
        sys.path.append(str(repo_root_dir))
    from RIPE.test_models import ModelEvaluator

def find_latest_checkpoint():
    """Find the latest trained PPO model checkpoint"""
    # Search in outputs directory
    patterns = [
        os.path.join("outputs", "**", "model_weights.pt"),
        os.path.join("outputs", "**", "model_step_*_final.pth"),
    ]
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern, recursive=True))
    if not checkpoints:
        return None
    # Return the most recently modified one
    return max(checkpoints, key=os.path.getmtime)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Modified model only")
    parser.add_argument("--test_dir", type=str, required=False, default=None,
                       help="Path to test dataset directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to trained PPO model weights")
    parser.add_argument("--output_dir", type=str, default="test_results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Use TEST_DIR from environment if not specified
    if args.test_dir is None:
        args.test_dir = os.getenv("TEST_DIR")
        if args.test_dir is None:
            print("\n❌ ERROR: No test directory specified.")
            print("Either provide --test_dir argument or set TEST_DIR in .env file")
            return
        print(f"Using TEST_DIR from .env: {args.test_dir}")
    
    # Find checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
    
    print(f"\n{'='*60}")
    print("Testing MODIFIED Model Only (PPO)")
    print(f"{'='*60}\n")
    print(f"Test directory: {args.test_dir}")
    print(f"Checkpoint: {checkpoint or 'None (using pretrained)'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device=args.device)
    
    # Load test pairs
    test_pairs = evaluator.load_test_pairs(args.test_dir)
    
    # Test Modified Model Only
    print(f"\n{'='*60}")
    print("Testing MODIFIED Model")
    print(f"{'='*60}")
    model_modified = evaluator.load_model(checkpoint)
    matcher_modified, matcher_type_mod = evaluator.create_matcher(use_modified=True)
    
    results_mod, summary_mod = evaluator.evaluate_model(
        model_modified, matcher_modified, test_pairs,
        matcher_type=matcher_type_mod, save_dir=args.output_dir
    )
    evaluator.print_summary(summary_mod, "MODIFIED Model (PPO)")
    
    print(f"\n✅ Testing complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
