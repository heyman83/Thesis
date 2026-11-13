"""Train YOLO11n on a supplied YOLO detection dataset after a GPU check."""
import argparse
from pathlib import Path
from verify_gpu import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to your dataset YAML')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--batch', type=int, default=2)
    args = parser.parse_args()
    data = Path(args.data).resolve()
    if not data.is_file():
        parser.error(f'Dataset YAML does not exist: {data}')
    verify()
    from ultralytics import YOLO
    YOLO(str(Path(__file__).with_name('yolo11n.pt'))).train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, workers=0, amp=False, cache=False,
        project=str(Path(__file__).with_name('runs')), name='yolo11n')


if __name__ == '__main__':
    main()
