# YOLO11n training and evaluation

Windows/Python tools for a thesis project detecting PFM-1 targets. Includes CUDA verification, training, and validation confidence-threshold comparisons. Robot video processing and blur gating are planned and are not implemented here. This is a research prototype; benchmark results do not establish suitability for operational mine detection.

The repository contains source code and dependency pins. Obtain your own labelled dataset; local data, weights, recordings, and generated runs are excluded by `.gitignore`. See [THIRD_PARTY.md](THIRD_PARTY.md) for dependency and dataset notices.

Python 3.12 environment: `.venv`. Run all commands from this directory in PowerShell.

Verified on this computer: GTX 1050 Ti (4 GB, compute capability 6.1), NVIDIA driver 581.57, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, and Ultralytics 8.4.211. CUDA kernels, CUDA NMS, and YOLO11n forward/backward all passed. Exact installed packages are recorded in `requirements-lock.txt`.

```powershell
.\.venv\Scripts\python.exe verify_gpu.py
.\.venv\Scripts\python.exe train.py --data 'C:\path\to\data.yaml'
```

The training launcher repeats the GPU check before training and requires a real dataset YAML. Defaults: YOLO11n pretrained weights, GPU 0, 100 epochs, image size 416, batch 2, workers 0, no image cache, and AMP disabled. These conservative defaults suit a 4 GB Pascal GPU; memory needs depend on your dataset. If memory runs out, close GPU-heavy desktop apps and use `--batch 1 --imgsz 320`. No dataset training is started by setup.

`verify_gpu.py` tests CUDA arithmetic/autograd, torchvision CUDA NMS, and a randomly initialized YOLO11n forward/backward pass at 320 pixels. Results are written to `gpu-verification.json`. This is a compatibility smoke test, not an accuracy test or a guarantee that every training configuration fits.

Dataset layout for detection:

```text
datasets/my_dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Each label is a `.txt` file matching its image basename, with rows `class_id x_center y_center width height` using normalized coordinates. Create a YAML with an absolute `path`, `train: images/train`, `val: images/val`, and a `names` mapping for your classes.

Recreate the environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

Keep `constraints.txt` when installing or upgrading packages (`pip install -c constraints.txt ...`) to preserve the tested CUDA wheels. The NVIDIA driver's displayed CUDA version is its maximum supported runtime; the PyTorch wheel bundles CUDA 11.8. A separate CUDA Toolkit is not required.

References: https://pytorch.org/get-started/previous-versions/ and https://docs.ultralytics.com/models/yolo11/

## Evaluate a trained checkpoint

Use validation data to select settings, and reserve the test set for final evaluation. Replace the example paths below with your dataset YAML and checkpoint:

```powershell
.\.venv\Scripts\python.exe evaluate_thresholds.py --data '.\datasets\my_dataset\data.yaml' --weights '.\runs\yolo11n-2\weights\best.pt'
```

Compares confidence 0.10, 0.15, and 0.25 at image size 416 on the validation split. Labelled folders, plots, fixed-threshold JSON summaries, and comparison.md are saved under `runs/threshold_comparison`. Repeating this command replaces files in those comparison folders. Precision and recall in the summaries are calculated from confusion-matrix counts at matching IoU 0.5, rather than the console's best-F1 point.

For a separate test evaluation:

```powershell
.\.venv\Scripts\yolo.exe val 'model=./runs/yolo11n-2/weights/best.pt' 'data=./datasets/my_dataset/data.yaml' split=test imgsz=416 batch=2 device=0 workers=0 plots=True
```

## Publish to GitHub

Create an empty repository at https://github.com/new using your preferred name and visibility. Leave the README, license, and .gitignore initialization options unchecked because these files already exist locally. Choose a project license separately after reviewing THIRD_PARTY.md.

The local repository is initialized on `main`. Run these commands from the project directory, replacing YOUR-USERNAME and YOUR-REPOSITORY:

```powershell
git status --short
git add .gitignore .gitattributes README.md THIRD_PARTY.md data.example.yaml constraints.txt requirements-lock.txt train.py verify_gpu.py evaluate_thresholds.py
git diff --cached --stat
git commit -m "Add YOLO11n training and evaluation tools"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

If Git requests author details, set `git config user.name "Your Name"` and `git config user.email "your GitHub email or GitHub noreply email"`, then repeat the commit. HTTPS authentication normally opens a browser through Git Credential Manager. GitHub account passwords are not accepted for Git pushes; use the offered browser authentication or a personal access token if prompted.

For subsequent changes, review `git status` and `git diff`, stage the intended files, commit, and run `git push`. Inspect new dataset locations before staging: only the named dataset folders in .gitignore are excluded automatically.

GitHub instructions: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
