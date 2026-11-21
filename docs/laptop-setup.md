# Set up another Windows laptop

## Download the project

Install Git in PowerShell:

```powershell
winget install --id Git.Git -e --source winget
```

Alternatively use https://git-scm.com/install/windows. Close and reopen PowerShell after installation. Then:

```powershell
git --version
cd "$HOME\Documents"
git clone https://github.com/heyman83/Thesis.git
cd Thesis
```

Clone is for the first download. Later updates, from this same folder:

```powershell
git pull --ff-only
```

If Git reports local changes or divergent history, stop and inspect `git status`; do not discard work or force reset. A public repository can be cloned without signing in. A private repository requires GitHub access; use the Git Credential Manager browser sign-in when prompted, not your account password as a Git password.

The clone includes scripts, guides, tests and the selected approximately 5.5 MB checkpoint at `models/camera_finetuned.pt`. It does not include datasets, training runs, local virtual environments, session logs or generated deployment ZIPs.

## Connect to the Pi

You do not need Python, ROS or Docker on the laptop just to SSH, transfer files or use the browser dashboard. Connect to the robot hotspot, or use the same home network as the Pi. The Pi's address can differ between connections.

```powershell
ssh YOUR_USERNAME@PI_IP
```

Windows normally provides SSH/SCP through OpenSSH Client. Check `ssh -V`. If unavailable, install OpenSSH Client from Windows Optional Features, then reopen PowerShell.

The existing robot installation stays on the Pi; cloning on the laptop does not reinstall it. Start/stop from the Pi host:

```bash
sudo docker exec -d MentorPi bash /opt/robot-live/start_detector.sh
sudo docker exec MentorPi cat /opt/robot-live/console.log
sudo docker exec MentorPi bash /opt/robot-live/start_detector.sh --stop
```

Do not start a second detector while one is already running. Use the printed Phone browser URL on the laptop, replacing PI_IP with the reachable Pi address, preserving port 8081 and the complete key. The 127.0.0.1 URL is only for a browser on the Pi itself. No Python installation is needed to view it.

## Transfer updated code from the clone

First stop the detector and allow active inference/log finalization to finish. In laptop PowerShell, from the Thesis folder:

```powershell
ssh YOUR_USERNAME@PI_IP "mkdir -p ~/robot-live/models"
scp robot_detect.py robot_capture.py robot_telemetry.py requirements-pi.txt start_detector.sh YOUR_USERNAME@PI_IP:~/robot-live/
scp models/camera_finetuned.pt YOUR_USERNAME@PI_IP:~/robot-live/models/
```

Then in the Pi host SSH terminal:

```bash
sudo docker exec MentorPi mkdir -p /opt/robot-live
sudo docker cp "$HOME/robot-live/." MentorPi:/opt/robot-live/
```

This transfers source files and the selected model, not a Windows venv. Retain the working Pi venv and session logs. If requirements change, install them inside the Pi container's venv before restarting. The selected settings remain image size 640, confidence 0.25, NMS IoU 0.4, TTA/tiling off. No depth or movement features have been implemented.

## Optional Python for local tools

Only needed to run/edit-test Python locally, not to operate the Pi remotely. Install the Python install manager from https://www.python.org/downloads/windows/ and use it to install Python 3.12:

```powershell
py install 3.12
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install ultralytics==8.4.211
.\.venv\Scripts\python.exe -m pip check
```

Use an ordinary 64-bit Windows laptop for this environment. A traditional Python 3.12 installer also works; omit `py install 3.12` if already installed. These are CPU dependencies and can involve large downloads. The original `requirements-lock.txt` includes NVIDIA CUDA builds for the training desktop; do not use it as a generic laptop setup. `requirements-pi.txt` is for the robot, not the laptop. ROS imports and live camera subscriptions run in MentorPi, not native Windows.

Local offline model check with one of your own images:

```powershell
.\.venv\Scripts\python.exe robot_detect.py --model models/camera_finetuned.pt --image "C:\path\to\sample.jpg" --output runs/laptop-smoke
```

Local helper tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

No local GPU setup or training environment is required for SSH and browser use. For training on the NVIDIA desktop, use the main README instructions instead.
