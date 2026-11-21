import time
import unittest
import json
import tempfile
import threading
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from robot_detect import State, handler_for
from robot_capture import Frames, ThreadingHTTPServer
from robot_telemetry import SessionRecorder


class DetectionStateTests(unittest.TestCase):
    def test_live_stats_download_and_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)/'model.pt'
            model.write_bytes(b'model')
            recorder = SessionRecorder(Path(temp)/'logs', {'model':str(model)})
            server = ThreadingHTTPServer(('127.0.0.1',0), handler_for(Frames(Path(temp), 'camera'), State(3), 'secret', recorder))
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            base = 'http://127.0.0.1:'+str(server.server_address[1])
            try:
                with self.assertRaises(HTTPError) as denied:
                    urlopen(base+'/stats')
                self.assertEqual(denied.exception.code,403)
                for route in ['/stats','/download/summary.json']:
                    with urlopen(Request(base+route,headers={'X-Capture-Key':'secret'})) as response:
                        self.assertEqual(json.load(response)['processed_frames'],0)
                with urlopen(Request(base+'/download/frames.csv',headers={'X-Capture-Key':'secret'})) as response:
                    self.assertIn('inference_ms',response.read().decode())
                with self.assertRaises(HTTPError) as denied:
                    urlopen(Request(base+'/download/robot_detect.py',headers={'X-Capture-Key':'secret'}))
                self.assertEqual(denied.exception.code,404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                recorder.close()

    def test_unavailable_is_distinct_from_valid_empty_detection(self):
        state = State(3)
        self.assertEqual(state.snapshot()['status'], 'unavailable')
        state.error = None
        state.result = dict(status='ok', detections=[], _received=time.monotonic())
        self.assertEqual(state.snapshot()['status'], 'ok')
        self.assertEqual(state.snapshot()['detections'], [])

    def test_stale_results_do_not_retain_current_detections(self):
        state = State(3)
        state.error = None
        state.result = dict(status='ok', detections=[{'label': 'target'}], _received=time.monotonic() - 10)
        result = state.snapshot()
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['detections'], [])
        self.assertGreater(result['age_seconds'], 3)


if __name__ == '__main__':
    unittest.main()
