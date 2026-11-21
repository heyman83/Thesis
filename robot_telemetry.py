"""Session recording for the robot detector; CSV timings and JSONL detections."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time


class SessionRecorder:
    fields = ['processed_utc', 'frame_sequence', 'ros_sec', 'ros_nanosec',
              'width', 'height', 'detections', 'inference_ms', 'preprocess_ms',
              'postprocess_ms', 'predict_call_ms', 'frame_pipeline_ms',
              'frame_age_at_result_ms', 'processed_fps', 'received_frames',
              'overwritten_frames', 'expired_frames', 'cpu_temperature_c']

    def __init__(self, root, config):
        self.folder = Path(root).resolve() / datetime.now().strftime('session_%Y%m%d_%H%M%S_%f')
        self.folder.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.config = dict(config)
        model = Path(config['model'])
        if model.is_file():
            self.config['model_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
        self.config['started_utc'] = self.started_utc
        (self.folder/'config.json').write_text(json.dumps(self.config, indent=2))
        self.csv = (self.folder/'frames.csv').open('w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.csv, fieldnames=self.fields)
        self.writer.writeheader()
        self.csv.flush()
        self.jsonl = (self.folder/'detections.jsonl').open('w', encoding='utf-8')
        self.events = (self.folder/'events.jsonl').open('w', encoding='utf-8')
        self.times = []
        self.detection_counts = []
        self.latest = None

    def record(self, result, counters):
        row = dict(processed_utc=result['processed_utc'], frame_sequence=result['frame_sequence'],
                   ros_sec=result['ros_stamp']['sec'], ros_nanosec=result['ros_stamp']['nanosec'],
                   width=result['width'], height=result['height'], detections=len(result['detections']))
        for key in self.fields:
            if key in result and key != 'detections':
                row[key] = result[key]
        row.update(counters)
        try:
            row['cpu_temperature_c'] = float(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000
        except (OSError, ValueError):
            row['cpu_temperature_c'] = None
        with self.lock:
            self.writer.writerow({key: row.get(key) for key in self.fields})
            self.csv.flush()
            self.jsonl.write(json.dumps(result)+'\n')
            self.jsonl.flush()
            self.times.append(result['inference_ms'])
            self.detection_counts.append(len(result['detections']))
            self.latest = row
            if len(self.times) % 10 == 0:
                self._save_summary()

    def event(self, status, reason):
        with self.lock:
            self.events.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                             status=status, reason=reason))+'\n')
            self.events.flush()

    def _summary(self):
        elapsed = max(time.monotonic()-self.started, 1e-9)
        ordered = sorted(self.times)
        p95 = ordered[max(0, int(.95*len(ordered)+.999999)-1)] if ordered else None
        return dict(session=self.folder.name, started_utc=self.started_utc,
                    duration_seconds=round(elapsed, 2), processed_frames=len(ordered),
                    mean_inference_ms=round(statistics.mean(ordered), 2) if ordered else None,
                    median_inference_ms=round(statistics.median(ordered), 2) if ordered else None,
                    p95_inference_ms=p95, average_processed_fps=round(len(ordered)/elapsed, 3),
                    frames_with_detections=sum(n>0 for n in self.detection_counts),
                    total_box_observations=sum(self.detection_counts), latest=self.latest,
                    note='Boxes across frames are observations, not unique targets. Runtime accuracy requires ground truth.')

    def snapshot(self):
        with self.lock:
            return self._summary()

    def _save_summary(self):
        path = self.folder/'summary.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self._summary(), indent=2))
        temporary.replace(path)

    def close(self):
        with self.lock:
            self._save_summary()
            for stream in [self.csv, self.jsonl, self.events]:
                stream.close()
