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
    parser.add_argument('--weights', type=Path, help='Existing local checkpoint to fine-tune; defaults to yolo11n.pt')
    parser.add_argument('--name', default='yolo11n', help='Output run name under runs/')
    parser.add_argument('--optimizer', default='auto', help='Ultralytics optimizer, e.g. auto or AdamW')
    parser.add_argument('--lr0', type=float, help='Initial learning rate; select a fixed optimizer to use this value')
    args = parser.parse_args()
    data = Path(args.data).resolve()
    if not data.is_file():
        parser.error(f'Dataset YAML does not exist: {data}')
    weights = args.weights.resolve() if args.weights else Path(__file__).with_name('yolo11n.pt')
    if args.weights and not weights.is_file():
        parser.error(f'Checkpoint does not exist: {weights}')
    if args.lr0 is not None and (args.lr0 <= 0 or args.optimizer.lower() == 'auto'):
        parser.error('--lr0 must be positive and requires a fixed --optimizer such as AdamW')
    verify()
    from ultralytics import YOLO
    options = {} if args.lr0 is None else {'lr0': args.lr0}
    YOLO(str(weights)).train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, workers=0, amp=False, cache=False,
        project=str(Path(__file__).with_name('runs')), name=args.name,
        optimizer=args.optimizer, **options)


if __name__ == '__main__':
    main()
