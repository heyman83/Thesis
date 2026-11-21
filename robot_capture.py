"""Browser capture for a ROS 2 RGB topic. Run inside the robot's ROS container."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import parse_qs, urlparse

import cv2

PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot camera capture</title><style>
body{font:17px system-ui;background:#101b27;color:#edf4fa;margin:0;padding:20px;max-width:900px;margin:auto}
h1{font-size:25px}img{width:100%;height:auto;background:#000;display:block;border-radius:10px}
button{font:inherit;background:#8ce2b3;color:#092117;border:0;border-radius:8px;padding:16px 28px;margin:16px 0;cursor:pointer}
button:disabled{opacity:.4}p{line-height:1.5}#status{min-height:3em}.muted{color:#b7c8d7}label{display:block;margin:15px 0}input{font:inherit;width:180px}
</style><h1>Robot camera capture</h1>
<p class="muted">Live RGB preview. Capture saves the displayed frame as an original-resolution PNG.</p>
<img id="preview" alt="Waiting for camera frames">
<button id="capture" disabled>Capture photo</button>
<p id="status" role="status">Waiting for camera…</p><p id="info" class="muted"></p>
<p class="muted">Keep WonderPi open separately for driving. Stop the robot before capturing sharp images.</p>
<script>
const key=new URLSearchParams(location.search).get('key')||'';
const headers={'X-Capture-Key':key};
let current=null,busy=false,paused=false,lastUrl=null;
const image=document.querySelector('#preview'),button=document.querySelector('#capture'),status=document.querySelector('#status');
async function poll(){
 if(!paused){try{
  const response=await fetch('/frame',{headers,cache:'no-store'});
  if(!response.ok)throw new Error(await response.text());
  const id=response.headers.get('X-Frame-ID'), url=URL.createObjectURL(await response.blob());
  const decoded=new Image();decoded.src=url;await decoded.decode();
  if(!paused){image.src=url;current=id;button.disabled=busy;
   if(lastUrl)URL.revokeObjectURL(lastUrl);lastUrl=url;
  }else URL.revokeObjectURL(url);
 }catch(e){current=null;button.disabled=true;if(!busy)status.textContent='Camera unavailable: '+e.message;}}
 setTimeout(poll,200);
}
button.onclick=async()=>{
 if(!current||busy)return;
 busy=true;paused=true;button.disabled=true;const id=current;
 try{
  const response=await fetch('/capture',{method:'POST',headers:{...headers,'Content-Type':'application/json'},body:JSON.stringify({frame_id:id})});
  const data=await response.json();if(!response.ok)throw new Error(data.error);
  status.textContent='Saved '+data.filename+' — '+data.count+' photos in this session.';
 }catch(e){status.textContent='Not saved: '+e.message;}
 finally{busy=false;paused=false;}
};
fetch('/info',{headers}).then(r=>r.json()).then(d=>{document.querySelector('#info').textContent='Session: '+d.session+' | Topic: '+d.topic;status.textContent='Ready. Preview updates about 5 times per second.';});
poll();
</script></html>'''


class Frames:
    def __init__(self, folder, topic):
        self.folder = folder
        self.topic = topic
        self.lock = threading.Lock()
        self.frames = OrderedDict()
        self.sequence = 0
        self.count = len(list(folder.glob('*.png')))

    def add(self, image, stamp=None):
        ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        with self.lock:
            self.sequence += 1
            self.frames[str(self.sequence)] = (image.copy(), jpeg.tobytes(), time.monotonic(), stamp)
            while len(self.frames) > 30:
                self.frames.popitem(last=False)

    def latest(self):
        with self.lock:
            if not self.frames:
                return None
            key = next(reversed(self.frames))
            frame = self.frames[key]
            return (key, frame[1]) if time.monotonic() - frame[2] < 3 else None

    def save(self, frame_id):
        with self.lock:
            frame = self.frames.get(str(frame_id))
            if frame is None or time.monotonic() - frame[2] > 5:
                raise ValueError('Preview frame expired. Wait for a fresh camera image and try again.')
            image, _, _, stamp = frame
            name = datetime.now(timezone.utc).strftime('rgb_%Y%m%dT%H%M%S_%fZ_') + secrets.token_hex(3) + '.png'
            path = self.folder / name
            ok, encoded = cv2.imencode('.png', image)
            if not ok:
                raise OSError('PNG encoding failed')
            # Exclusive creation prevents accidental replacement of an existing capture.
            with path.open('xb') as stream:
                stream.write(encoded.tobytes())
            metadata = dict(filename=name, frame_id=str(frame_id), ros_stamp=stamp,
                            width=image.shape[1], height=image.shape[0], topic=self.topic,
                            saved_utc=datetime.now(timezone.utc).isoformat())
            path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
            self.count += 1
            return dict(filename=name, count=self.count)


def handler_for(frames, key):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log the private URL key.

        def reply(self, code, body, content_type='application/json', extra=None):
            if isinstance(body, dict):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            supplied = self.headers.get('X-Capture-Key', '')
            if urlparse(self.path).path == '/':
                supplied = parse_qs(urlparse(self.path).query).get('key', [''])[0]
            return secrets.compare_digest(supplied, key)

        def do_GET(self):
            if not self.authorized():
                return self.reply(403, {'error': 'Open the full URL printed in the Pi terminal, including ?key=…'})
            route = urlparse(self.path).path
            if route == '/':
                return self.reply(200, PAGE, 'text/html; charset=utf-8')
            if route == '/info':
                return self.reply(200, dict(session=frames.folder.name, topic=frames.topic))
            if route == '/frame':
                latest = frames.latest()
                if latest is None:
                    return self.reply(503, 'No fresh RGB frames. Check camera and ROS topic.', 'text/plain')
                return self.reply(200, latest[1], 'image/jpeg', {'X-Frame-ID': latest[0]})
            return self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.authorized():
                return self.reply(403, {'error': 'Invalid capture key'})
            if urlparse(self.path).path != '/capture':
                return self.reply(404, {'error': 'Not found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1024:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                saved = frames.save(body.get('frame_id'))
            except (ValueError, TypeError, AttributeError) as error:
                return self.reply(409, {'error': str(error)})
            except OSError:
                return self.reply(500, {'error': 'Cannot save image. Check disk space and output-folder permissions.'})
            self.reply(200, saved)
    return Handler


def start_ros(frames, topic):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    rclpy.init()
    node = Node('robot_browser_capture')
    bridge = CvBridge()
    previous = [0.0]

    def receive(message):
        now = time.monotonic()
        if now - previous[0] < 0.2:
            return
        previous[0] = now
        try:
            image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            frames.add(image, dict(sec=message.header.stamp.sec, nanosec=message.header.stamp.nanosec))
        except Exception as error:
            node.get_logger().error(f'RGB frame conversion failed: {error}')

    node.create_subscription(Image, topic, receive, qos_profile_sensor_data)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    return rclpy, node, thread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/ascamera/camera_publisher/rgb0/image')
    parser.add_argument('--output', type=Path, default=Path.home() / 'camera_dataset')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--demo', action='store_true', help='Synthetic frames for testing; does not use ROS or the camera')
    args = parser.parse_args()
    session = args.output.resolve() / datetime.now().strftime('session_%Y%m%d_%H%M%S_%f')
    session.mkdir(parents=True, exist_ok=False)
    frames = Frames(session, 'DEMO — synthetic frames' if args.demo else args.topic)
    stop = threading.Event()
    ros = None
    if args.demo:
        import numpy as np
        def demo():
            while not stop.is_set():
                image = np.zeros((400, 640, 3), dtype=np.uint8)
                cv2.putText(image, 'DEMO CAMERA', (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 240, 180), 2)
                cv2.putText(image, datetime.now().isoformat(timespec='milliseconds'), (30, 240), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                frames.add(image)
                stop.wait(.2)
        threading.Thread(target=demo, daemon=True).start()
    else:
        ros = start_ros(frames, args.topic)
    key = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(frames, key))
    server.daemon_threads = True
    print(f'On the Pi: http://127.0.0.1:{args.port}/?key={key}', flush=True)
    print(f'On your phone: http://PI_IP:{args.port}/?key={key}', flush=True)
    print(f'Saving in: {session}\nPress Ctrl+C to stop. Use only on your trusted robot network.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        stop.set()
        if ros:
            rclpy, node, thread = ros
            rclpy.shutdown()
            thread.join(timeout=2)
            node.destroy_node()


if __name__ == '__main__':
    main()
