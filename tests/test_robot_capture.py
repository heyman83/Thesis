import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from robot_capture import Frames, ThreadingHTTPServer, handler_for


class CaptureTests(unittest.TestCase):
    def test_saves_requested_frame_not_latest(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Frames(Path(temp), '/test/rgb')
            first = np.full((40, 64, 3), (10, 20, 30), dtype=np.uint8)
            store.add(first)
            frame_id = store.latest()[0]
            store.add(np.full_like(first, 200))
            saved = store.save(frame_id)
            restored = cv2.imread(str(Path(temp) / saved['filename']))
            np.testing.assert_array_equal(restored, first)
            metadata = json.loads((Path(temp) / saved['filename']).with_suffix('.json').read_text())
            self.assertEqual((metadata['width'], metadata['height']), (64, 40))
            self.assertNotEqual(saved['filename'], store.save(frame_id)['filename'])

    def test_missing_and_stale_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Frames(Path(temp), '/test')
            self.assertIsNone(store.latest())
            with self.assertRaises(ValueError):
                store.save('missing')
            store.add(np.zeros((10, 10, 3), dtype=np.uint8))
            image, jpeg, _, stamp = store.frames['1']
            store.frames['1'] = image, jpeg, time.monotonic() - 10, stamp
            self.assertIsNone(store.latest())
            with self.assertRaises(ValueError):
                store.save('1')

    def test_http_preview_capture_and_auth(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Frames(Path(temp), '/test')
            store.add(np.zeros((40, 64, 3), dtype=np.uint8))
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(store, 'test-key'))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with self.assertRaises(HTTPError) as denied:
                    urlopen(base + '/frame')
                self.assertEqual(denied.exception.code, 403)
                with urlopen(base + '/?key=test-key') as page:
                    self.assertIn(b'Capture photo', page.read())
                with urlopen(Request(base + '/frame', headers={'X-Capture-Key': 'test-key'})) as response:
                    frame_id = response.headers['X-Frame-ID']
                    self.assertEqual(response.headers['Content-Type'], 'image/jpeg')
                request = Request(base + '/capture', data=json.dumps({'frame_id': frame_id}).encode(),
                                  headers={'X-Capture-Key': 'test-key'}, method='POST')
                with urlopen(request) as response:
                    result = json.load(response)
                self.assertEqual(result['count'], 1)
                self.assertTrue((Path(temp) / result['filename']).is_file())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
