from pathlib import Path
import tempfile
import unittest

from extract_aurora_rgb import extract


class ExtractionTests(unittest.TestCase):
    def test_filters_flattens_and_can_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'usb'
            output = Path(folder) / 'flat'
            for scene in ['Aurora930_Scene One', 'Aurora930_Scene Two']:
                for kind in ['RGB', 'Depth', 'IR']:
                    target = source / 'nested' / scene / kind / 'image.JPG'
                    target.parent.mkdir(parents=True)
                    target.write_bytes((scene + kind).encode())
            self.assertEqual(extract(source, output), (2, 0))
            self.assertEqual(len(list(output.iterdir())), 2)
            self.assertEqual({p.read_bytes() for p in output.iterdir()},
                             {b'Aurora930_Scene OneRGB', b'Aurora930_Scene TwoRGB'})
            self.assertEqual(extract(source, output), (0, 2))
            self.assertEqual(len(list(source.rglob('*.JPG'))), 6)

    def test_single_capture_root_dry_run_and_nested_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'Aurora930_Single'
            image = source / 'RGB' / 'frame.jpeg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'RGB')
            output = source / 'extracted'
            self.assertEqual(extract(source, output, dry_run=True), (1, 0))
            self.assertFalse(output.exists())
            self.assertEqual(extract(source, output), (1, 0))
            self.assertEqual(extract(source, output), (0, 1))


if __name__ == '__main__':
    unittest.main()
