#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

if [[ "${1:-}" == "--stop" ]]; then
    if [[ ! -f detector.pid ]]; then echo "No detector PID file found."; exit 1; fi
    detector_pid=$(cat detector.pid)
    if [[ ! "$detector_pid" =~ ^[0-9]+$ ]]; then echo "Invalid PID file."; exit 1; fi
    if [[ ! -r "/proc/$detector_pid/cmdline" ]]; then echo "Detector is not running."; exit 0; fi
    if ! tr '\0' ' ' < "/proc/$detector_pid/cmdline" | grep -Fq "$PWD/robot_detect.py"; then
        echo "PID belongs to another process; refusing to stop it."; exit 1
    fi
    kill -INT "$detector_pid"
    echo "Stop requested. Wait for session summary to finish before copying logs."
    exit 0
fi

if [[ -f detector.pid ]]; then
    detector_pid=$(cat detector.pid)
    if [[ "$detector_pid" =~ ^[0-9]+$ ]] && kill -0 "$detector_pid" 2>/dev/null; then
        echo "A process with the saved PID is still running. Check/stop it before restarting."; exit 1
    fi
fi
if [[ ! -x venv/bin/python ]]; then echo "Create the venv first; see README.md."; exit 1; fi
set +u
source /opt/ros/humble/setup.bash
set -u
echo "$$" > detector.pid
exec venv/bin/python -u "$PWD/robot_detect.py" \
    --model "$PWD/models/camera_finetuned.pt" \
    --imgsz 640 --conf 0.25 --iou 0.4 --max-fps 3 \
    --log-dir "$PWD/session_logs" "$@" > console.log 2>&1
