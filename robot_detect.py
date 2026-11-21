"""ROS 2 camera inference and browser preview. No motor or arm commands."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import signal
import threading
import time
from collections import deque
from urllib.parse import parse_qs, urlparse

import cv2
import robot_capture as web
from robot_telemetry import SessionRecorder

DETECTION_PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Robot target detector</title>
<style>body{font:17px system-ui;background:#101b27;color:#edf4fa;max-width:1000px;margin:auto;padding:20px}img{width:100%;height:auto;background:#000;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}p{line-height:1.5}.muted{color:#b7c8d7}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}.card{padding:14px;background:#203347;border-radius:8px}.card b{display:block;font-size:24px;margin-top:6px}button{padding:12px;background:#8ce2b3;border:0;border-radius:6px;cursor:pointer;margin:4px;font:inherit}#status{padding:12px;border-left:4px solid #e4ae58}h1{font-size:26px}</style>
<h1>Robot target detector</h1><p id="status">Waiting for camera and inference…</p>
<p id="session" class="muted"></p><div class="stats">
<div class="card">Model inference<b id="inference">—</b></div><div class="card">Processed FPS<b id="fps">—</b></div>
<div class="card">Frame age<b id="age">—</b></div><div class="card">Boxes this frame<b id="boxes">—</b></div>
<div class="card">Session frames<b id="count">0</b></div><div class="card">CPU temperature<b id="temperature">—</b></div></div>
<img id="preview" alt="No current detection frame">
<p><button onclick="download('frames.csv')">Download timing CSV</button><button onclick="download('detections.jsonl')">Download detections</button><button onclick="download('summary.json')">Download summary</button></p>
<p class="muted">Inference time excludes overlay/transfer. FPS includes throttling and camera waits. Accuracy cannot be measured without ground truth.</p>
<details><summary>Detailed frame results</summary><pre id="details"></pre></details>
<p class="muted">Display and ROS reporting only. No motor or arm control. An unavailable frame is not a negative detection.</p>
<script>
const key=new URLSearchParams(location.search).get('key')||'',headers={'X-Capture-Key':key};let oldUrl=null;
async function poll(){try{
 const response=await fetch('/detections',{headers,cache:'no-store'});if(!response.ok)throw Error('Access denied: use the full URL printed by the server.');
 const d=await response.json();document.querySelector('#status').textContent=d.status==='ok'?`${d.detections.length} target detection(s) | frame age ${d.age_seconds.toFixed(1)}s`:'Vision unavailable: '+d.reason;
 document.querySelector('#inference').textContent=d.status==='ok'?d.inference_ms.toFixed(1)+' ms':'—';
 document.querySelector('#fps').textContent=d.status==='ok'?d.processed_fps.toFixed(2):'—';
 document.querySelector('#age').textContent=d.age_seconds===undefined?'—':d.age_seconds.toFixed(2)+' s';
 document.querySelector('#boxes').textContent=d.status==='ok'?d.detections.length:'—';
 const stats=await (await fetch('/stats',{headers,cache:'no-store'})).json();
 document.querySelector('#count').textContent=stats.processed_frames;
 document.querySelector('#temperature').textContent=stats.latest?.cpu_temperature_c==null?'n/a':stats.latest.cpu_temperature_c.toFixed(1)+' °C';
 document.querySelector('#session').textContent=stats.session+' | 640 input by default | conf '+(d.confidence_threshold??'0.25')+' | NMS '+(d.nms_iou??'0.4')+' | TTA/tiles off';
 document.querySelector('#details').textContent=JSON.stringify(d,null,2);
 if(d.status==='ok'){
  const frame=await fetch('/frame?id='+encodeURIComponent(d.preview_frame_id),{headers,cache:'no-store'});if(frame.ok){const url=URL.createObjectURL(await frame.blob());document.querySelector('#preview').src=url;if(oldUrl)URL.revokeObjectURL(oldUrl);oldUrl=url;}
 }else{document.querySelector('#preview').removeAttribute('src');}
}catch(e){document.querySelector('#status').textContent=e.message;document.querySelector('#preview').removeAttribute('src');}
setTimeout(poll,400);}poll();
async function download(name){const response=await fetch('/download/'+name,{headers});if(!response.ok){alert('Download failed');return;}const url=URL.createObjectURL(await response.blob()),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
</script></html>'''


def predict(model, image, args):
    started = time.monotonic()
    result = model.predict(source=image, imgsz=args.imgsz, conf=args.conf,
                           iou=args.iou, augment=False, tile=False, device='cpu', verbose=False)[0]
    predict_ms = (time.monotonic()-started)*1000
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls.item())
            boxes.append(dict(class_id=cls, label=result.names[cls],
                              confidence=float(box.conf.item()),
                              xyxy=[float(v) for v in box.xyxy[0].tolist()]))
    timings = {f'{key}_ms': round(float(result.speed.get(key, 0)), 3)
               for key in ['preprocess', 'inference', 'postprocess']}
    timings['predict_call_ms'] = round(predict_ms, 3)
    return result.plot(), boxes, timings


class State:
    def __init__(self, max_age):
        self.lock = threading.Lock()
        self.result = None
        self.error = 'Waiting for the first camera frame and inference result'
        self.max_age = max_age

    def snapshot(self):
        with self.lock:
            if self.error:
                return dict(status='unavailable', reason=self.error, detections=[])
            result = dict(self.result)
        age = time.monotonic() - result.pop('_received')
        result['age_seconds'] = round(age, 3)
        if age > self.max_age:
            result.update(status='unavailable', reason='Camera/inference result is stale', detections=[])
        return result


def handler_for(frames, state, key, recorder=None):
    parent = web.handler_for(frames, key)

    class Handler(parent):
        def do_GET(self):
            if not self.authorized():
                return self.reply(403, {'error': 'Use the full URL printed in the terminal'})
            route = urlparse(self.path).path
            if recorder is not None and route == '/stats':
                return self.reply(200, recorder.snapshot())
            if recorder is not None and route.startswith('/download/'):
                name = route.rsplit('/', 1)[-1]
                if name == 'summary.json':
                    return self.reply(200, recorder.snapshot())
                if name not in {'frames.csv', 'detections.jsonl', 'events.jsonl', 'config.json'}:
                    return self.reply(404, {'error': 'Not found'})
                with recorder.lock:
                    body = (recorder.folder/name).read_bytes()
                return self.reply(200, body, 'text/csv' if name.endswith('.csv') else 'application/octet-stream')
            if route == '/detections':
                return self.reply(200, state.snapshot())
            if route == '/frame' and state.snapshot()['status'] != 'ok':
                return self.reply(503, 'No current detection result', 'text/plain')
            if route == '/frame':
                frame_id = parse_qs(urlparse(self.path).query).get('id', [''])[0]
                with frames.lock:
                    frame = frames.frames.get(frame_id)
                if frame is None or time.monotonic() - frame[2] > state.max_age:
                    return self.reply(409, 'Preview frame expired', 'text/plain')
                return self.reply(200, frame[1], 'image/jpeg')
            super().do_GET()

        def do_POST(self):
            self.reply(405, {'error': 'This detector has no capture or robot-control endpoint'})
    return Handler


def run_ros(args, model):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge

    rclpy.init()
    node = Node('pfm1_detector')
    bridge = CvBridge()
    frames = web.Frames(Path('live_preview'), args.topic)  # In memory; no capture files.
    state = State(args.max_age)
    stop = threading.Event()
    pending_lock = threading.Lock()
    pending = [None]
    counters = dict(received_frames=0, overwritten_frames=0, expired_frames=0)
    completed = deque(maxlen=30)
    recorder = SessionRecorder(args.log_dir, dict(model=str(args.model.resolve()),
                               topic=args.topic, conf=args.conf, nms_iou=args.iou,
                               imgsz=args.imgsz, tta=False, tile=False, device='cpu',
                               max_fps=args.max_fps, threads=args.threads, max_age=args.max_age))
    publisher = node.create_publisher(String, args.result_topic, 10)

    def receive(message):
        with pending_lock:
            counters['received_frames'] += 1
            if pending[0] is not None:
                counters['overwritten_frames'] += 1
            pending[0] = (message, time.monotonic())  # One slot; replace old queued frames.

    node.create_subscription(Image, args.topic, receive, qos_profile_sensor_data)

    previous_status = [None]
    def report():
        message = String()
        current = state.snapshot()
        message.data = json.dumps(current)
        publisher.publish(message)
        signature = (current['status'], current.get('reason', ''))
        if signature != previous_status[0]:
            recorder.event(*signature)
            previous_status[0] = signature

    node.create_timer(1.0, report)  # Also reports unavailable/stale camera state.

    def infer():
        while not stop.is_set():
            with pending_lock:
                item, pending[0] = pending[0], None
            if item is None:
                stop.wait(.03)
                continue
            message, received = item
            if time.monotonic() - received > args.max_age:
                with pending_lock:
                    counters['expired_frames'] += 1
                continue
            try:
                loop_started = time.monotonic()
                image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
                overlay, detections, timings = predict(model, image, args)
                frames.add(overlay)
                preview_id = frames.latest()[0]
                now = time.monotonic()
                completed.append(now)
                fps = (len(completed)-1)/(completed[-1]-completed[0]) if len(completed)>1 else 0
                with pending_lock:
                    current_counters = dict(counters)
                with state.lock:
                    state.result = dict(status='ok', model=str(args.model), confidence_threshold=args.conf,
                                        imgsz=args.imgsz, width=image.shape[1], height=image.shape[0],
                                        nms_iou=args.iou, tta=False, tile=False,
                                        ros_stamp=dict(sec=message.header.stamp.sec, nanosec=message.header.stamp.nanosec),
                                        camera_frame_id=message.header.frame_id,
                                        processed_utc=datetime.now(timezone.utc).isoformat(),
                                        **timings, detections=detections, _received=received,
                                        frame_sequence=len(recorder.times)+1,
                                        frame_pipeline_ms=round((now-loop_started)*1000,3),
                                        frame_age_at_result_ms=round((now-received)*1000,3),
                                        processed_fps=round(fps,3), **current_counters,
                                        preview_frame_id=preview_id)
                    state.error = None
                    record = {k:v for k,v in state.result.items() if not k.startswith('_')}
                record['valid_at_result'] = now-received <= args.max_age
                recorder.record(record, current_counters)
                report()
            except Exception as error:
                with state.lock:
                    state.error = str(error)
                node.get_logger().error(f'Inference failed: {error}')
                stop.wait(1)
            stop.wait(1 / args.max_fps)

    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    worker = threading.Thread(target=infer, daemon=True)
    spin.start()
    worker.start()
    web.PAGE = DETECTION_PAGE
    key = secrets.token_urlsafe(24)
    server = web.ThreadingHTTPServer((args.host, args.port), handler_for(frames, state, key, recorder))
    server.daemon_threads = True
    print(f'Pi browser: http://127.0.0.1:{args.port}/?key={key}', flush=True)
    print(f'Phone browser: http://PI_IP:{args.port}/?key={key}', flush=True)
    print(f'Input: {args.topic}\nResults: {args.result_topic}\nNo motors or arm commands are sent.', flush=True)
    print(f'Session logs: {recorder.folder}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        worker.join()  # Let active inference finish before destroying the publisher.
        rclpy.shutdown()
        spin.join(timeout=2)
        node.destroy_node()
        recorder.close()


def main():
    # Docker detached launches must still support graceful stop and log finalization.
    def stop_signal(_number, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Local .pt file or exported NCNN model folder')
    parser.add_argument('--topic', default='/ascamera/camera_publisher/rgb0/image')
    parser.add_argument('--result-topic', default='/pfm1/detections')
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--iou', type=float, default=.4, help='NMS overlap threshold')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--log-dir', type=Path, default=Path('session_logs'))
    parser.add_argument('--max-fps', type=float, default=3)
    parser.add_argument('--max-age', type=float, default=3, help='Maximum seconds since frame reception for valid results')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--image', type=Path, help='Offline smoke test; no ROS needed')
    parser.add_argument('--output', type=Path, default=Path('runs/robot_smoke'))
    args = parser.parse_args()
    if not args.model.exists():
        parser.error('Model file/folder not found. Copy your trained model first.')
    if not 0 < args.conf < 1 or not 0 < args.iou <= 1 or min(args.max_fps, args.max_age, args.threads, args.imgsz) <= 0:
        parser.error('Use confidence between 0 and 1, and positive sizes/rates/thread count.')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(Path.home() / '.config/pfm1_vision'))
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    model = YOLO(str(args.model.resolve()), task='detect')
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            parser.error('Cannot read --image')
        overlay, detections, timings = predict(model, image, args)
        args.output.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output / 'prediction.jpg'), overlay):
            raise OSError('Could not save prediction image')
        data = dict(model=str(args.model), confidence=args.conf, imgsz=args.imgsz,
                    nms_iou=args.iou, tta=False, tile=False, **timings, detections=detections)
        (args.output / 'detections.json').write_text(json.dumps(data, indent=2))
        print(json.dumps(data, indent=2))
        return
    run_ros(args, model)


if __name__ == '__main__':
    main()
