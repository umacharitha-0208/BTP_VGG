# BTP_VGG

This repository contains the RIPE-based VGG project code, configs, tests, and training outputs tracked in Git.

## Prerequisites

- Python 3.10+ (recommended)
- Git
- Git LFS

## Clone and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/umacharitha-0208/BTP_VGG.git
   cd BTP_VGG
   ```

2. Install Git LFS (one-time on your machine):

   ```bash
   git lfs install
   ```

3. Pull LFS-tracked files (model weights and other large artifacts):

   ```bash
   git lfs pull
   ```

## Git LFS Tracked File Types

This repo uses Git LFS for large model and checkpoint artifacts:

- `*.pt`
- `*.pth`
- `*.ckpt`
- `*.onnx`

If these are not fetched with LFS, you may see small pointer text files instead of actual binaries.

## Notes

- Generated training outputs are ignored via `.gitignore`.
- Existing committed large artifacts are stored and transferred through Git LFS.
