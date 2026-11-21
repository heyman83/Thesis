# PFM-1 Detection and Removal with YOLO11n

A thesis research project investigating visual detection and physical removal of PFM-1 landmine targets using YOLO11n and robotic arm. The repository has tools to train the landmine detector alongside model, program to interface

The intended application is a camera-equipped robot. Current code covers offline training and evaluation, browser-based RGB capture, and a ROS 2 detector with browser preview and timestamped detection messages. Tracking, depth fusion, motion-blur filtering, and automatic robot actions are not yet implemented. This is a research prototype, and dataset benchmarks do not establish operational detection reliability.

## Features

For Raspberry Pi 5 / MentorPi deployment using the current trained model, see the [robot deployment guide](docs/robot-deployment.md). The detector accepts a model path so a later fine-tuned checkpoint can replace the baseline without changing the application code.

- CUDA compatibility checks using real GPU computation and a YOLO11n forward/backward pass.
- Transfer learning from pretrained YOLO11n detection weights.
- Validation at fixed confidence thresholds of 0.10, 0.15, and 0.25.
- Confusion matrices, precision–recall plots, and JSON/Markdown evaluation summaries.
- Pinned Python dependencies for a tested Windows training environment.

## Repository contents

| File | Purpose |
|---|---|
| `train.pyo | Train YOLO11n after verifying GPU support |
| `verify_gpu.py` | Test CUDA, torchvision GPU NMS, and YOLO11n computation |
| `robot_capture.py` | Preview ROS 2 colour images in a browser and save selected frames |
| `evaluate_thresholds.py` | Compare fixed-threshold detection counts for a single-class dataset |
| `data.example.yaml` | Example dataset configuration |
| `requirements-lock.txt` | Exact installed package versions |
| `constraints.txt` | Preserve the tested PyTorch and torchvision builds |
| `THIRD_PARTY.md` | Software licensing and dataset attribution notes |

Datasets, recordings, local settings, and generated runs are excluded from version control. The selected deployment checkpoint is included at `models/camera_finetuned.pt`; other model checkpoints remain excluded.

For another Windows computer, follow the [laptop setup guide](docs/laptop-setup.md). Git is needed to clone/update the repository; Python is optional unless running local Python tools.

## Installation

The environment was tested with Windows, Python 3.12, a GTX 1050 Ti with 4 GB VRAM, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, and Ultralytics 8.4.211. It requires a compatible NVIDIA driver; the PyTorch wheels include the CUDA runtime, so a separate CUDA Toolkit is not required.

Run these commands in PowerShell from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe verify_gpu.py
```

The GPU check writes `gpu-verification.json`. It verifies compatibility, not accuracy or whether every training configuration will fit in memory. Use `-c constraints.txt` when installing additional packages to retain the tested PyTorch versions.

## Dataset preparation

For collecting original camera images on the Raspberry Pi 5 / MentorPi robot, see the [browser capture setup](docs/browser-capture.md). This runs separately inside the robot's ROS environment; it does not use the Windows training environment.

Supply a labelled YOLO dataset with separate training, validation, and test splits:

```text
datasets/pfm1/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

Copy `data.example.yaml` to the dataset directory and set its `path` to the absolute dataset root. Each label file must match its image basename. For bounding-box detection, label rows contain `class_id x_center y_center width height`, with normalized coordinates and class ID `0` for the target. An empty label file represents a genuine background image.

Split camera footage by capture session or location to avoid placing near-identical frames in different splits. Use validation data to choose settings and reserve test data for final evaluation. See `THIRD_PARTY.md` for the original dataset source.

## Training

```powershell
.\.venv\Scripts\python.exe train.py --data '.\datasets\pfm1\data.yaml'
```

| Setting | Default |
|---|---|
| Model | Pretrained `yolo11n.pt` (downloaded on first use) |
| Epochs | 100 |
| Image size | 416 |
| Batch size | 2 |
| Device | GPU 0 |
| Data loader workers | 0 |
| Mixed precision | Disabled |

The launcher repeats the GPU check before training. Images are resized with aspect ratio preserved and padded as needed. Outputs are saved under `runs/yolo11n`, with an incremented name for subsequent runs; the selected checkpoint is in `weights/best.pt`.

Override the supported settings as needed:

```powershell
.\.venv\Scripts\python.exe train.py --data '.\datasets\pfm1\data.yaml' --epochs 100 --imgsz 640 --batch 1
```

If GPU memory runs out, reduce batch size or image size. By default the launcher starts from `yolo11n.pt`. To fine-tune an existing checkpoint, supply `--weights` and a separate run name:

```powershell
.\.venv\Scripts\python.exe train.py --data '.\camera_dataset\data.yaml' --weights '.\runs\yolo11n-2\weights\best.pt' --name camera_finetune --epochs 50 --imgsz 640 --batch 1 --optimizer AdamW --lr0 0.001
```

These are initial fine-tuning settings to validate, not established optimal settings. This starts a new training run from learned weights rather than resuming the previous optimizer/epoch state. A manually specified learning rate requires a fixed optimizer because `auto` can override it. Keep camera sessions separate across training, validation and test sets, and evaluate the starting checkpoint on the camera validation split before fine-tuning for a fair comparison.

## Evaluation

Compare fixed confidence thresholds on the validation set, replacing the checkpoint path with the actual training output:

```powershell
.\.venv\Scripts\python.exe evaluate_thresholds.py --data '.\datasets\pfm1\data.yaml' --weights '.\runs\yolo11n\weights\best.pt'
```

The comparison script supports the single target class used in this project. It evaluates at image size 416 and matching IoU 0.5. Each threshold has a labelled folder under `runs/threshold_comparison`, containing plots and a fixed-threshold JSON summary. Aggregate results are saved as `comparison.json` and `comparison.md`. Rerunning the script overwrites these comparison outputs.

- **Precision:** proportion of reported detections that correctly match labelled targets.
- **Recall:** proportion of labelled targets successfully detected.
- **False positives:** unmatched predicted boxes.
- **False negatives:** missed labelled targets.

The comparison summaries use counts at each fixed confidence threshold. Ultralytics' console precision and recall instead summarize its best-F1 operating point; they should not be substituted for the fixed-threshold counts.

Evaluate a checkpoint on the test split using Ultralytics:

```powershell
.\.venv\Scripts\yolo.exe val 'model=./runs/yolo11n/weights/best.pt' 'data=./datasets/pfm1/data.yaml' split=test imgsz=416 batch=2 device=0 workers=0 plots=True
```

This command uses the default evaluation confidence floor to construct metric curves. To inspect a confusion matrix at a selected deployment threshold, explicitly add a setting such as `conf=0.10`.

## Robot browser dashboard

The [live deployment guide](docs/robot-live-dashboard.md) covers the fine-tuned camera checkpoint on Raspberry Pi 5: 640 input, confidence 0.25, NMS IoU 0.4, with TTA and tiling disabled. It includes a browser preview, per-frame timing/detection logs, session summaries, and SSH transfer/start instructions. The prepared package is under `outputs/robot-live`. Live ROS integration and performance require verification on the Pi.

## Licensing and references

A license for the original project scripts has not yet been selected. Dependency and dataset licenses are documented in [THIRD_PARTY.md](THIRD_PARTY.md).

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics validation](https://docs.ultralytics.com/modes/val/)
- [PyTorch installation versions](https://pytorch.org/get-started/previous-versions/)

Repository publishing instructions are maintained separately in [the GitHub guide](docs/github-publishing.md).
