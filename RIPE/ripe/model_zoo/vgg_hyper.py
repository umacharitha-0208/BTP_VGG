from pathlib import Path
import tempfile

import torch

from ripe.models.backbones.vgg import VGG
from ripe.models.ripe import RIPE
from ripe.models.upsampler.hypercolumn_features import HyperColumnFeatures


def vgg_hyper(model_path: Path = None, desc_shares=None):
    if model_path is None:
        # Use cross-platform temp directory
        temp_dir = Path(tempfile.gettempdir())
        model_path = temp_dir / "ripe_weights.pth"

        if model_path.exists():
            print(f"Using existing weights from {model_path}")
        else:
            print("Weights file not found. Downloading ...")
            torch.hub.download_url_to_file(
                "https://cvg.hhi.fraunhofer.de/RIPE/ripe_weights.pth",
                str(model_path),
            )
    else:
        if not model_path.exists():
            print(f"Error: {model_path} does not exist.")
            raise FileNotFoundError(f"Error: {model_path} does not exist.")

    backbone = VGG(pretrained=False)
    upsampler = HyperColumnFeatures()

    extractor = RIPE(
        net=backbone,
        upsampler=upsampler,
        desc_shares=desc_shares,
    )

    state_dict = torch.load(model_path, map_location="cpu")
    
    # Check if this is a PPO-trained checkpoint (has actor/critic keys)
    has_ppo = any("policy" in k for k in state_dict.keys())
    
    if has_ppo:
        # Full PPO-trained checkpoint - load everything
        extractor.load_state_dict(state_dict, strict=True)
        print("[+] Loaded PPO-trained weights (full model including policy)")
    else:
        # Original pretrained weights - skip missing PPO keys
        extractor.load_state_dict(state_dict, strict=False)
        print("[+] Loaded pretrained weights (PPO policy randomly initialized)")

    return extractor
