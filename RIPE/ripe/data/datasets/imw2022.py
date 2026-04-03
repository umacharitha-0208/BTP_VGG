# ripe/data/datasets/imw2022.py
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class IMW2022Dataset(Dataset):
    """
    Custom loader for your imw2022 dataset.
    Samples pairs:
      - Positive (~50%): two different images from the SAME scene (train/scenes)
      - Negative (~50%): images from DIFFERENT scenes
    """

    def __init__(self, root_dir, phase="train", image_size=1024, positive_ratio=0.5):
        self.root = Path(root_dir) / phase          # train or test subfolder
        self.phase = phase
        self.positive_ratio = positive_ratio

        print(f"[IMW2022] Looking for data in: {self.root}")
        
        # Check if root directory exists
        if not self.root.exists():
            print(f"[IMW2022 ERROR] Directory does not exist: {self.root}")
            print(f"[IMW2022] Expected structure: {root_dir}/{phase}/<scene_folders>/<images>")
            # Try without phase subfolder
            self.root = Path(root_dir)
            print(f"[IMW2022] Trying alternative path: {self.root}")
        
        # Discover scene folders
        self.scene_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        print(f"[IMW2022 {phase.upper()}] Found {len(self.scene_dirs)} scenes")

        self.images_by_scene = {}
        for scene_dir in self.scene_dirs:
            scene_name = scene_dir.name
            imgs = sorted(list(scene_dir.glob("*.[jJ][pP][gG]")) + list(scene_dir.glob("*.[jJ][pP][eE][gG]")) + list(scene_dir.glob("*.[pP][nN][gG]")))
            if len(imgs) < 2:
                print(f"Skipping scene {scene_name} — only {len(imgs)} images")
                continue
            self.images_by_scene[scene_name] = imgs

        self.scene_names = list(self.images_by_scene.keys())
        if len(self.scene_names) < 2:
            raise ValueError("Need at least 2 scenes with ≥2 images each")

        # Standard transform
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.scene_names) * 64  # arbitrary large number — oversample pairs

    def __getitem__(self, idx):
        if random.random() < self.positive_ratio:
            # Positive pair — same scene
            scene = random.choice(self.scene_names)
            img_paths = random.sample(self.images_by_scene[scene], 2)
        else:
            # Negative pair — different scenes
            scene_a, scene_b = random.sample(self.scene_names, 2)
            img_a = random.choice(self.images_by_scene[scene_a])
            img_b = random.choice(self.images_by_scene[scene_b])
            img_paths = [img_a, img_b]

        img0 = self.transform(Image.open(img_paths[0]).convert("RGB"))
        img1 = self.transform(Image.open(img_paths[1]).convert("RGB"))

        return {
            "image0": img0,
            "image1": img1,
            "label": torch.tensor(1.0 if img_paths[0].parent == img_paths[1].parent else 0.0),
            "scene": img_paths[0].parent.name,
        }


def get_imw2022(root_dir: str, phase: str = "train", **kwargs):
    # Filter out config parameters that are not for the dataset
    dataset_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['image_size', 'positive_ratio']}
    return IMW2022Dataset(root_dir, phase=phase, **dataset_kwargs)
