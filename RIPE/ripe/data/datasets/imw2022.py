# ripe/data/datasets/imw2022.py
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


def _random_homography(W, H, max_shift=0.15):
    """Return a random projective transform matrix (3×3, float32, pixel coords)."""
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    noise = np.random.uniform(-max_shift, max_shift, (4, 2)).astype(np.float32)
    noise *= np.array([W, H], dtype=np.float32)
    dst = np.clip(src + noise, [0, 0], [W, H]).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _warp_tensor(img_tensor, H_matrix, mean, std):
    """Warp a (3,H,W) normalised tensor with a pixel-space homography."""
    _, h, w = img_tensor.shape
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img_np = ((img_tensor * std_t + mean_t) * 255).byte().permute(1, 2, 0).numpy()
    warped_np = cv2.warpPerspective(img_np, H_matrix, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
    warped_t = torch.from_numpy(warped_np).permute(2, 0, 1).float() / 255.0
    warped_t = (warped_t - mean_t) / std_t
    return warped_t


class IMW2022Dataset(Dataset):
    """
    Samples image pairs:
      - Real positive  (~pos_ratio of real pairs): two different images from the SAME scene
      - Real negative  (~1-pos_ratio of real pairs): images from DIFFERENT scenes
      - Synthetic      (synthetic_ratio of all samples): one image + homography-warped version
                       → label=1, homography matrix provided → no RANSAC needed for reward
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root_dir, phase="train", image_size=512,
                 positive_ratio=0.5, synthetic_ratio=0.5):
        self.root = Path(root_dir) / phase
        self.phase = phase
        self.positive_ratio = positive_ratio
        self.synthetic_ratio = synthetic_ratio
        self.image_size = image_size

        print(f"[IMW2022] Looking for data in: {self.root}")

        if not self.root.exists():
            print(f"[IMW2022 ERROR] Directory does not exist: {self.root}")
            self.root = Path(root_dir)
            print(f"[IMW2022] Trying alternative path: {self.root}")

        self.scene_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        print(f"[IMW2022 {phase.upper()}] Found {len(self.scene_dirs)} scenes")

        self.images_by_scene = {}
        self.all_images = []
        for scene_dir in self.scene_dirs:
            scene_name = scene_dir.name
            # Support both flat (scene/*.jpg) and nested (scene/images/*.jpg) layouts
            img_base = scene_dir / "images" if (scene_dir / "images").is_dir() else scene_dir
            imgs = sorted(
                list(img_base.glob("*.[jJ][pP][gG]"))
                + list(img_base.glob("*.[jJ][pP][eE][gG]"))
                + list(img_base.glob("*.[pP][nN][gG]"))
            )
            if len(imgs) < 2:
                print(f"Skipping scene {scene_name} — only {len(imgs)} images")
                continue
            self.images_by_scene[scene_name] = imgs
            self.all_images.extend(imgs)

        self.scene_names = list(self.images_by_scene.keys())
        if len(self.scene_names) < 2:
            raise ValueError("Need at least 2 scenes with ≥2 images each")

        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self):
        return len(self.scene_names) * 64

    def _load(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    def _synthetic_pair(self):
        """Return a pair created by warping one image with a random homography."""
        path = random.choice(self.all_images)
        img = self._load(path)                       # (3, S, S) normalised
        _, H, W = img.shape
        hmat = _random_homography(W, H, max_shift=0.12)
        warped = _warp_tensor(img, hmat, self.MEAN, self.STD)
        return {
            "image0": img,
            "image1": warped,
            "label": torch.tensor(1.0),
            "scene": path.parent.name,
            "homography": torch.from_numpy(hmat).float(),   # (3,3) pixel-space H
        }

    def __getitem__(self, idx):
        if random.random() < self.synthetic_ratio:
            return self._synthetic_pair()

        # Real pair
        if random.random() < self.positive_ratio:
            scene = random.choice(self.scene_names)
            img_paths = random.sample(self.images_by_scene[scene], 2)
        else:
            scene_a, scene_b = random.sample(self.scene_names, 2)
            img_paths = [
                random.choice(self.images_by_scene[scene_a]),
                random.choice(self.images_by_scene[scene_b]),
            ]

        img0 = self._load(img_paths[0])
        img1 = self._load(img_paths[1])
        label = float(img_paths[0].parent == img_paths[1].parent)
        return {
            "image0": img0,
            "image1": img1,
            "label": torch.tensor(label),
            "scene": img_paths[0].parent.name,
            "homography": None,
        }


def get_imw2022(root_dir: str, phase: str = "train", **kwargs):
    dataset_kwargs = {k: v for k, v in kwargs.items()
                      if k in ["image_size", "positive_ratio", "synthetic_ratio"]}
    return IMW2022Dataset(root_dir, phase=phase, **dataset_kwargs)
