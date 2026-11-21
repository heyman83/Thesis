# Fine-tuned robot detector: browser view and session recording

This package uses `runs/camera_finetune/weights/best.pt`, copied as `models/camera_finetuned.pt`. Selected settings are image size 640, confidence 0.25, NMS IoU 0.4, TTA off and tiling off. These were selected on Version2 validation data, not confirmed on the held-out test set or Pi hardware. No arm or motor commands are sent. Depth fusion and tracking are not implemented.

## 1. Transfer from Windows using SCP

Connect Windows to the Pi/robot network. In Windows PowerShell, replace the login and address:

```powershell
scp -r '.\outputs\robot-live\robot-live' USERNAME@PI_IP:~/
ssh USERNAME@PI_IP
```

Alternatively extract MentorPi_Live_Detector.zip and transfer its robot-live folder on USB. On the Pi host:

```bash
sudo docker exec MentorPi mkdir -p /opt/robot-live
sudo docker cp "$HOME/robot-live/." MentorPi:/opt/robot-live/
sudo docker inspect -f '{{.HostConfig.NetworkMode}}' MentorPi
hostname -I
```

Do not replace the robot container or OS. Keep your host copy as a backup.

## 2. Install once inside the existing container

Internet access is required for package downloads (the robot hotspot alone may not provide it). Do not install Windows CUDA packages on the Pi.

```bash
sudo docker exec -it MentorPi bash
source /opt/ros/humble/setup.bash
cd /opt/robot-live
python3 --version
uname -m
python3 -c "import rclpy, cv2; from cv_bridge import CvBridge"
```

Expect aarch64 and the ROS Humble container's Python 3.10. If venv or imports are missing, install dependencies inside the container:

```bash
apt-get update
apt-get install python3-venv python3-opencv ros-humble-cv-bridge
```

Then:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install -r requirements-pi.txt
python -m pip check
python -c "import rclpy, cv2, torch; from cv_bridge import CvBridge; from ultralytics import YOLO; print('Ready')"
exit
```

The separate venv isolates pip packages while allowing ROS imports. NumPy remains below 2 for the Humble binary bridge. If installation reports incompatible wheels or system-package conflicts, stop and inspect the error instead of upgrading the robot's whole software stack. ARM dependency installation still requires verification on the actual Pi.

## 3. Start from the Pi host, including over SSH

Keep the Aurora plugged in and its existing driver running. Launch detached so closing SSH does not stop the detector:

```bash
sudo docker exec -d MentorPi bash /opt/robot-live/start_detector.sh
sudo docker exec MentorPi cat /opt/robot-live/console.log
```

Wait briefly and repeat the second command if the URL has not appeared. It prints a browser URL including a randomly generated `?key=...`, plus the session log directory. First model startup can be slow. Do not start a second copy while the first is running. This is manual startup; it does not automatically restart after power loss or container restart.

To change a runtime limit, pass an argument after the launcher name, e.g. `--max-fps 2` or `--max-age 5`. Max FPS is an upper limit, not a promised rate. Do not loosen freshness limits simply to hide a slow or disconnected camera.

## 4. Open the dashboard

For Docker `host` network mode, open the printed address in the Pi browser, or replace `127.0.0.1`/`PI_IP` with the Pi's actual network IP in a Windows/phone browser. Keep port 8081 and the complete key. Use HTTP, not HTTPS. The phone/computer must be on the same network. Do not forward this port to the public internet.

For bridge networking, use the container IP in the Pi host's browser, or the explicit port-forwarding steps in browser-capture.md with 8081 at both ends. Do not recreate the robot container merely to publish a port.

The dashboard shows:

- Annotated RGB frames, correlated with the displayed JSON by preview frame ID.
- Model inference milliseconds (Ultralytics forward-pass timing, CPU on the Pi).
- Actual rolling processed FPS, including camera waits, throttling and processing costs.
- Local frame age since the ROS subscriber received the frame (not a measurement of full camera-to-browser latency).
- Boxes in the current frame, session processed-frame count, and CPU temperature when exposed by Linux.
- Download buttons for timing CSV, detection JSONL, and session summary.

Detailed JSON includes source ROS timestamps, NMS/confidence/input settings, model path, per-stage timings, frame processing time, local result age, and counters. Sensor-time synchronization is needed before interpreting camera timestamps as cross-device latency. Camera frames overwritten in the one-slot queue are counted as skipped, not model misses. Browser refresh is independent of inference FPS.

The first inference can include initialization overhead; predict_call_ms records the Python prediction call wall time separately from the model-only inference_ms. Session averages include all recorded frames. A result older than --max-age is marked unavailable. No frame means unknown, not no target. Temperature is informational; no automatic thermal control is implemented.

## 5. Recording files

Every launch creates a unique folder under `/opt/robot-live/session_logs` **inside Docker**:

| File | Contents |
|---|---|
| config.json | Model SHA-256 for .pt files, settings and session start time |
| frames.csv | One row per successful processed frame: stage timings, prediction-call time, frame age, FPS, dimensions, counters, temperature, box count |
| detections.jsonl | One JSON record per successful processed frame, including individual labels, confidence and pixel boxes; valid_at_result flags late results |
| events.jsonl | Changes between available/unavailable states and reasons |
| summary.json | Processed frames, mean/median/p95 inference time, average FPS, box observations, and latest counters |

CSV/JSONL are flushed per frame; summary refreshes every 10 frames and on clean shutdown. Browser summary downloads calculate current values even before the next periodic summary. No video or source images are recorded, keeping storage and CPU costs lower. Downloads are snapshots; download again to include later frames. Use distinct download names/folders for different sessions.

The number of boxes across frames is **not the number of unique targets**. Runtime precision, recall, false positives and misses cannot be calculated without ground-truth annotations. For a scored field trial, record independent ground truth or representative video and annotate it later; this dashboard does not perform that assessment.

## 6. Stop and preserve data

On the Pi host:

```bash
sudo docker exec MentorPi bash /opt/robot-live/start_detector.sh --stop
```

Wait for inference to finish. Confirm the process ended by checking the saved PID or console before copying. Then copy the complete session folders to ordinary Pi storage:

```bash
mkdir -p "$HOME/robot-session-logs"
sudo docker cp MentorPi:/opt/robot-live/session_logs/. "$HOME/robot-session-logs/"
sudo chown -R "$(id -u):$(id -g)" "$HOME/robot-session-logs"
```

On Windows, retrieve them:

```powershell
scp -r USERNAME@PI_IP:~/robot-session-logs '.\robot-session-logs'
```

Do this after each collection session, or download logs directly through the dashboard. Container storage survives an ordinary stop/start but may be lost when containers are removed/recreated. Downloaded/host copies are your durable records. This package does not change Docker mounts automatically. Check free disk space for long experiments.

## 7. Later model replacement or acceleration

Keep the current checkpoint and create a new model filename for rollback. Supply `--model /opt/robot-live/models/new_model.pt` to the launcher after stopping the old process. Re-validate confidence and NMS when model weights change.

For faster inference, export the same fine-tuned checkpoint to NCNN in the activated Pi venv, then pass the full exported directory with --model. Benchmark accuracy and timings before adopting it:

```bash
yolo export model=models/camera_finetuned.pt format=ncnn imgsz=640 device=cpu
```

The export can download additional tools. A 640 export must be evaluated/deployed at its supported input size. Configuration hashing currently fingerprints a .pt file; an exported directory is identified by its model path, not a content hash.

## Troubleshooting

- No page: inspect console.log, Docker networking, IP address and port 8081.
- No RGB: check `ros2 topic hz /ascamera/camera_publisher/rgb0/image` inside MentorPi. Source any additional workspace setup used by the working camera terminal.
- Missing ROS modules: source ROS, use the system-site-packages venv, and verify the container's Python matches ROS.
- Stale frames: inspect inference_ms, frame age, overwritten-frame count and CPU temperature; lower max-fps to reduce load, and investigate Pi cooling/power. Do not interpret skipped frames as successful negative detections.
- Permission or full disk: inspect output path and `df -h` inside MentorPi.

Verified locally: real checkpoint CPU prediction and telemetry/HTTP helper tests. ROS subscriptions, the Pi package environment, browser rendering on the target device and live robot timing still require on-device testing. See https://docs.ultralytics.com/modes/predict/ and https://docs.ultralytics.com/guides/raspberry-pi/ for runtime/export documentation.
