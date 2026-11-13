"""Compare fixed confidence thresholds on the validation split."""
import json
import argparse
from pathlib import Path
import verify_gpu  # Keep Ultralytics configuration inside this workspace.
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    args = parser.parse_args()
    data = args.data.resolve()
    weights = args.weights.resolve()
    if not data.is_file() or not weights.is_file():
        parser.error('Both --data and --weights must point to existing files.')
    output = root / 'runs/threshold_comparison'
    summaries = []
    for confidence in (0.10, 0.15, 0.25):
        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data), split='val', imgsz=416, batch=2, device=0,
            workers=0, conf=confidence, plots=True,
            project=str(output), name=f'validation_conf_{confidence:.2f}', exist_ok=True,
        )
        matrix = metrics.confusion_matrix.matrix
        tp, fp, fn = int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0])
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        summary = dict(
            split='val', confidence=confidence, matching_iou=0.5,
            weights=str(weights), data=str(data), imgsz=416,
            true_positives=tp, false_positives=fp, false_negatives=fn,
            labelled_targets=tp + fn, precision=precision, recall=recall,
            f1=2*precision*recall/(precision+recall) if precision+recall else 0,
            note='Fixed-threshold counts from the confusion matrix; not the headline best-F1 metrics.',
        )
        folder = Path(metrics.save_dir)
        summary['results_folder'] = str(folder)
        (folder / 'fixed_threshold_summary.json').write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(summaries, indent=2))
    lines = ['# Validation confidence threshold comparison', '',
             'YOLO11n best.pt from yolo11n-2; imgsz=416; matching IoU=0.5.',
             'Counts and percentages are measured at each fixed confidence threshold.', '',
             '| Confidence | Correct detections | Misses | False alarms | Precision | Recall | F1 |',
             '|---|---|---|---|---|---|---|']
    for s in summaries:
        lines.append(f"| {s['confidence']:.2f} | {s['true_positives']} | {s['false_negatives']} | {s['false_positives']} | {s['precision']:.1%} | {s['recall']:.1%} | {s['f1']:.1%} |")
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
