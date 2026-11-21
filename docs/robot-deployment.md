# Deploy the existing detector on Raspberry Pi 5 / MentorPi

This is a detection-only prototype. It subscribes to the existing Aurora RGB topic, performs CPU inference, displays bounding boxes in a browser, and publishes JSON detections as `std_msgs/msg/String` on `/pfm1/detections`. It does not move the robot or arm, estimate depth, track objects, reject blur, or establish that an area is clear. Test using inert targets and retain existing independent robot controls.

The starting model is the completed 100-epoch YOLO11n run trained on the downloaded dataset: `runs/yolo11n-2/weights/best.pt`. It is already trained for the target class, unlike the generic COCO `yolo11n.pt` used before training. The USB package includes this checkpoint as `models/base.pt`. Fine-tuning is not needed to test this deployment pipeline.

## 1. Transfer the package

Extract `MentorPi_Detector.zip` on your Windows computer and copy the `robot-vision` folder onto USB. On the Pi desktop, copy that folder into your home directory. Do not copy the Windows `.venv` or install its CUDA requirements on the Pi.

On the **Debian Pi host**, run:

```bash
sudo docker exec MentorPi mkdir -p /opt/robot-vision
sudo docker cp "$HOME/robot-vision/." MentorPi:/opt/robot-vision/
sudo docker inspect -f '{{.HostConfig.NetworkMode}}' MentorPi
hostname -I
```

Keep the last two outputs for browser access. This copies files into the existing container; it does not replace or restart the robot's camera software. Keep the USB/host copy: files installed only in the container can be lost when it is recreated.

## 2. Create an isolated inference environment inside MentorPi

These commands need internet access to download packages. They can be several hundred MB or more. The robot hotspot alone may not provide internet. Run inside the container, not in the Debian host terminal:

```bash
sudo docker exec -it MentorPi bash
source /opt/ros/humble/setup.bash
cd /opt/robot-vision
python3 --version
uname -m
```

Expect ARM64 (`aarch64`) and Python 3.10 or newer. If imports already work, skip the apt installation; otherwise install the ROS bridge and venv support inside the container:

```bash
python3 -c "import rclpy; from cv_bridge import CvBridge; import cv2"
```

If that reports missing modules or `venv` is unavailable:

```bash
apt-get update
apt-get install python3-venv python3-opencv ros-humble-cv-bridge
```

Then create a venv which can see ROS's system packages while isolating inference dependencies:

```bash
python3 -m venv --system-site-packages /opt/robot-vision/venv
source /opt/robot-vision/venv/bin/activate
python -m pip install -r requirements-pi.txt
python -m pip check
python -c "import rclpy, cv2, numpy, torch; from cv_bridge import CvBridge; from ultralytics import YOLO; print('Inference dependencies available')"
```

NumPy is kept below 2 for compatibility with the ROS Humble cv_bridge binary. These are separate ARM/CPU dependency pins, not the tested Windows/CUDA environment. Installation and live ROS behavior still need verification on the actual Pi. If pip reports no compatible wheel, stop and check the reported Python/architecture rather than upgrading the whole robot system. Do not run a broad system upgrade or change the camera driver.

## 3. Run the existing model

Keep the Aurora connected and its normal camera driver running:

```bash
cd /opt/robot-vision
source /opt/ros/humble/setup.bash
source venv/bin/activate
python robot_detect.py --model models/base.pt --conf 0.10 --imgsz 416
```

If the terminal where camera topics were visible used additional workspace setup files, source those here too. The input defaults to `/ascamera/camera_publisher/rgb0/image`. The detector uses a one-frame pending queue, CPU inference and two Torch threads so it does not build up a backlog of old frames. `--max-fps 3` limits how often inference is attempted; it is not a promise of 3 FPS. Start with the robot stationary and observe actual latency before concurrent driving.

Open the full URL printed by the program, including its access key. With Docker network mode `host`, use `127.0.0.1:8081` in the Pi's browser, or the Pi's network IP and port 8081 from a phone on the same network. The page displays model path, confidence threshold, detection boxes, inference time, and frame age. There is no capture button or motor-control endpoint.

If network mode is not `host`, see the network section of [browser-capture.md](browser-capture.md). For a bridge relay use port **8081** at both ends, rather than the capture page's 8080. Do not recreate the robot container solely to expose the port.

`0.10` is an initial high-recall validation setting, not a confirmed camera operating threshold. A valid result with zero detections differs from `status: unavailable`. Results older than three seconds since callback reception become unavailable; this is a local freshness check, not a full check of camera transport delay. ROS source timestamps are also included. If CPU inference exceeds this limit, investigate speed and choose an appropriate freshness limit with `--max-age`, rather than interpreting stale results as current. The browser requests the overlay by its preview frame ID to match the JSON result; correlate published ROS source timestamps when integrating consumers.

To inspect messages in a second container terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /pfm1/detections
```

JSON includes `status`, `detections`, labels, confidence, pixel `xyxy` boxes in the original camera image, dimensions, camera frame ID, source timestamp, processing timestamp, frame age, and inference time. On camera failure/stale data it reports unavailable with no current boxes; consumers must inspect status before interpreting the list. These are 2D detections only, not target locations in a map. For future depth integration, synchronize/alignment-check depth with the colour source timestamp and use camera calibration before projecting positions.

## 4. Optional faster CPU inference using NCNN

First establish a working `.pt` baseline. Ultralytics recommends NCNN for Raspberry Pi ARM CPU performance; export and benchmark on the Pi. Do not assume published YOLO26 benchmarks apply directly to this YOLO11n checkpoint.

Inside the same activated venv:

```bash
yolo export model=models/base.pt format=ncnn imgsz=416 device=cpu
python robot_detect.py --model models/base_ncnn_model --conf 0.10 --imgsz 416
```

Export can automatically download additional tools/dependencies and may take time. Transfer the entire NCNN output folder when moving exported models. Compare boxes and confidence with the `.pt` baseline on representative saved camera images before switching. NCNN has not been validated on this Pi by the authoring environment.

## 5. Replace with the fine-tuned model later

Stop the detector with Ctrl+C. Copy the new checkpoint from USB into the host `robot-vision/models` directory, keeping `base.pt` for rollback. On the Pi host:

```bash
sudo docker cp "$HOME/robot-vision/models/finetuned.pt" MentorPi:/opt/robot-vision/models/finetuned.pt
```

Inside the activated container environment:

```bash
python robot_detect.py --model models/finetuned.pt --conf 0.10 --imgsz 640
```

Use the image size and confidence you selected on validation data. For NCNN, export the new checkpoint again at the chosen size; an old exported folder still contains the old weights. No application code changes are required.

## Offline check without ROS

The same program can test a saved image on a computer with the training environment:

```powershell
.\.venv\Scripts\python.exe robot_detect.py --model '.\runs\yolo11n-2\weights\best.pt' --image 'C:\path\to\camera.jpg'
```

It writes `runs/robot_smoke/prediction.jpg` and `detections.json`. This verifies model loading and CPU prediction, not ROS subscriptions, browser rendering, Pi latency, or robot behavior. Startup on the Pi is manual in this prototype; add auto-start only after validating operation and storage/network configuration.

## References

- https://docs.ultralytics.com/guides/raspberry-pi/
- https://docs.ultralytics.com/modes/predict/
- https://index.ros.org/p/cv_bridge/

Use the browser on the trusted robot network only. Its per-run key is a basic access check, not internet-facing authentication or HTTPS. The helper does not install a firewall rule, expose motor interfaces, or modify the robot's existing controller.
