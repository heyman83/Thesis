import csv
import json
from pathlib import Path
import tempfile
import unittest
from robot_telemetry import SessionRecorder


class TelemetryTests(unittest.TestCase):
    def test_session_records_timings_boxes_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)/'weights.pt'
            model.write_bytes(b'test weights')
            recorder = SessionRecorder(Path(temp)/'logs', dict(model=str(model),conf=.25,nms_iou=.4))
            counters = dict(received_frames=10,overwritten_frames=7,expired_frames=0)
            for sequence, inference in enumerate([10.,20.,30.], 1):
                result = dict(processed_utc='2026-11-18T00:00:00+00:00',frame_sequence=sequence,
                              ros_stamp=dict(sec=1,nanosec=2),width=640,height=400,
                              detections=[dict(label='target',confidence=.8,xyxy=[1,2,3,4])],
                              inference_ms=inference,preprocess_ms=1.,postprocess_ms=2.,
                              predict_call_ms=inference+3,frame_pipeline_ms=inference+5,
                              frame_age_at_result_ms=inference+6,processed_fps=2.)
                recorder.record(result,counters)
            recorder.event('unavailable','stale')
            self.assertEqual(recorder.snapshot()['mean_inference_ms'],20.)
            recorder.close()
            summary=json.loads((recorder.folder/'summary.json').read_text())
            self.assertEqual(summary['processed_frames'],3)
            self.assertEqual(summary['p95_inference_ms'],30.)
            self.assertEqual(summary['total_box_observations'],3)
            with (recorder.folder/'frames.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[-1]['overwritten_frames'],'7')
            self.assertEqual(rows[-1]['detections'],'1')
            detections=[json.loads(line) for line in (recorder.folder/'detections.jsonl').read_text().splitlines()]
            self.assertEqual(detections[-1]['detections'][0]['xyxy'],[1,2,3,4])
            self.assertEqual(len(json.loads((recorder.folder/'config.json').read_text())['model_sha256']),64)


if __name__=='__main__':
    unittest.main()
