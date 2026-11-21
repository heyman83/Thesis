# Browser image capture on MentorPi

This helper previews the existing ROS 2 RGB stream and saves a selected frame as a native-resolution PNG. It subscribes to `/ascamera/camera_publisher/rgb0/image`; it does not open the USB camera or control motors. WonderPi remains available for driving. The preview is JPEG for network efficiency; the saved PNG comes from the corresponding original BGR frame, without preview compression, resizing, or overlays.

## 1. Transfer the script

Copy `robot_capture.py` from this repository onto a USB drive. On the Pi desktop, copy it from the USB drive into your home folder using the file manager.

Open a terminal on the Pi **host** (outside Docker):

```bash
sudo docker cp "$HOME/robot_capture.py" MentorPi:/tmp/robot_capture.py
sudo docker inspect -f '{{.HostConfig.NetworkMode}}' MentorPi
hostname -I
```

Write down the Pi's robot-network IP address. If the network mode is `host`, the page's port 8080 will be available through that IP. For `bridge` or another network mode, see Network troubleshooting below; do not recreate the robot container just to expose a port.

## 2. Start inside the existing ROS container

```bash
sudo docker exec -it MentorPi bash
source /opt/ros/humble/setup.bash
python3 -c "import rclpy, cv2; from cv_bridge import CvBridge; print('Capture dependencies available')"
python3 /tmp/robot_capture.py --output /tmp/camera_dataset
```

Use the container's system `python3`, not a Windows environment or training venv. If dependencies are missing, install only the missing packages inside the Ubuntu/ROS container (requires internet):

```bash
apt-get update
apt-get install python3-opencv ros-humble-cv-bridge
```

The camera and its existing driver must be running. The script prints a local URL, a phone URL template, and the save directory. Each restart creates a new session folder and access key. Leave the terminal running; Ctrl+C stops this capture helper, not the robot's other processes.

## 3. Open and capture

On the Pi browser, open the exact `http://127.0.0.1:8080/?key=...` URL printed by the script if the container uses host networking. On the phone, connect to the same robot network and replace `PI_IP` in the phone URL with the Pi's actual IP. Keep the full `?key=...` part. Do not use HTTPS.

The page shows a live preview and a Capture photo button. Arrange the scene, stop the robot to reduce blur, then tap Capture photo. It saves the frame selected from the preview and displays the filename and session count. On a slow connection, this can be older than the camera's current frame; the helper rejects frames more than five seconds old or no longer in its short buffer. Wait for a fresh preview and retry if requested.

Each PNG has a JSON sidecar recording the image dimensions, source ROS timestamp, topic, frame ID, and UTC save time. Session folder names use the Pi's local clock. Check the Pi's clock before collecting research data. No automatic blur rejection is implemented.

The phone can switch between WonderPi and the browser; phone background-app behavior may pause either app. A second device or the Pi's monitor is useful if you need simultaneous driving controls and capture preview. This page does not hook into WonderPi's photo button.

## 4. Copy images out of the container

The chosen `/tmp/camera_dataset` is **inside Docker**, not yet an ordinary folder in the Pi user's home. After capture, press Ctrl+C and `exit` to return to the Pi host, then run:

```bash
mkdir -p "$HOME/camera_dataset"
sudo docker cp MentorPi:/tmp/camera_dataset/. "$HOME/camera_dataset/"
sudo chown -R "$(id -u):$(id -g)" "$HOME/camera_dataset"
```

Open `camera_dataset` in your home folder and verify several PNGs before copying them to USB for annotation. Copy out after each session. Container `/tmp` is not durable dataset storage: retain the host copy and a backup before container cleanup, replacement, or system maintenance. Future integration can save through an existing host bind mount, once its path is verified.

## Troubleshooting

- **No fresh RGB frames:** inside MentorPi, run `ros2 topic hz /ascamera/camera_publisher/rgb0/image`. Verify the camera is connected. If your working ROS terminal used additional workspace setup files, source those in the capture terminal too. The subscriber uses the sensor-data QoS profile.
- **Connection refused:** check that the script is still running, the IP is correct, and Docker uses host networking or publishes/forwards the capture port. Do not disable firewalls broadly.
- **Port in use:** run with `--port 8081` and use the URL it prints.
- **Forbidden:** use the complete URL from the current run, including its key.
- **Permission/disk error:** choose a writable output folder and check `df -h` inside the container.

### Network troubleshooting for a bridge-mode container

First try the page in the Pi's browser using the container IP instead of 127.0.0.1. Find it on the Pi host:

```bash
sudo docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' MentorPi
```

A Docker bridge IP is usually accessible from the Pi host, but not directly from your phone. If needed, an explicit host relay can forward one port without restarting the robot container. On the **Debian Pi host**, install `socat` if missing:

```bash
sudo apt-get update
sudo apt-get install socat
```

Replace `CONTAINER_IP` with the inspected address, then leave this in a separate host terminal:

```bash
socat TCP-LISTEN:8080,bind=0.0.0.0,reuseaddr,fork TCP:CONTAINER_IP:8080
```

Now use the Pi's network IP on the phone. Stop the relay with Ctrl+C after collection. For other network modes, inspect the configuration before making changes.

## Local verification without a robot

On a computer with Python and OpenCV, `python robot_capture.py --demo --host 127.0.0.1 --output camera_dataset` provides synthetic frames for testing the page. These are test images, not camera captures. ROS dependencies are imported only in real-camera mode.

## References

- ROS cv_bridge: https://index.ros.org/p/cv_bridge/
- ROS sensor-data QoS: https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/qos.py

Use the page on a trusted local robot network. Its URL key prevents casual unauthenticated capture requests; it is not an internet-facing authenticated HTTPS service. Do not forward it to the public internet.
