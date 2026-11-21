"""Inspect camera validation false positives using the installed Ultralytics matcher."""
import csv
import json
from pathlib import Path
import shutil
import verify_gpu
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runs/camera_finetune_review'


class RecordingMatrix(ConfusionMatrix):
    def __init__(self, names, folder):
        super().__init__(names=names, save_matches=True)
        self.folder = folder
        self.records = []

    def plot_matches(self, img, *args, **kwargs):
        # Tiled validation can merge GPU predictions with CPU original targets.
        # Normalize only the plotting copies, leaving scoring tensors untouched.
        if self.matches is not None:
            for group in self.matches.values():
                for key, values in group.items():
                    group[key] = [value.cpu() for value in values]
        return super().plot_matches(img.cpu(), *args, **kwargs)

    def process_batch(self, detections, batch, conf=.25, iou_thres=.5):
        before = self.matrix.copy()
        super().process_batch(detections, batch, conf, iou_thres)
        difference = self.matrix - before
        source = Path(batch['im_file'])
        tp, fp, fn = int(difference[0, 0]), int(difference[0, 1]), int(difference[1, 0])
        scores = [float(v.item()) for v in self.matches['FP'].get('conf', [])]
        self.records.append(dict(image=source.name, correct=tp, false_positives=fp, missed=fn,
                                 highest_false_positive_confidence=max(scores, default=0)))
        for label, count in [('false_positive_originals', fp), ('missed_originals', fn)]:
            if count:
                destination = self.folder / label
                destination.mkdir(exist_ok=True)
                shutil.copy2(source, destination / source.name)


class ReviewValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.confusion_matrix = RecordingMatrix(model.names, self.save_dir)
        self.confusion_matrix.source_tile_counts = list(self.dataloader.dataset.tile_counts) if self.args.tile else []


def run(weights, confidence, name, review=True, nms_iou=.7, output=OUTPUT,
        data=ROOT / 'camera_dataset/data.yaml', augment=False, imgsz=640, tile=False):
    model = YOLO(str(ROOT / weights))
    metrics = model.val(data=str(data), split='val',
                        imgsz=imgsz, batch=1, workers=0, device=0, conf=confidence,
                        iou=nms_iou, augment=augment, tile=tile, plots=True, visualize=review, project=str(output), name=name,
                        validator=ReviewValidator if review else DetectionValidator)
    folder = Path(metrics.save_dir)
    matrix = metrics.confusion_matrix
    tp, fp, fn = int(matrix.matrix[0, 0]), int(matrix.matrix[0, 1]), int(matrix.matrix[1, 0])
    summary = dict(run=name, split='val', model=weights, data=str(data), imgsz=imgsz, confidence=confidence,
                   matching_iou=.5, nms_iou=nms_iou, augment=augment, tile=tile, correct=tp, missed=fn, false_positives=fp,
                   precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                   best_f1_precision=float(metrics.box.mp), best_f1_recall=float(metrics.box.mr),
                   map50=float(metrics.box.map50), map50_95=float(metrics.box.map),
                   speed_ms_per_image=dict(metrics.speed), folder=str(folder))
    if review:
        if tile:
            summary['source_tile_counts'] = matrix.source_tile_counts
            summary['total_tiles'] = sum(matrix.source_tile_counts)
        records = sorted(matrix.records, key=lambda x:x['false_positives'], reverse=True)
        summary['images_with_false_positives'] = sum(r['false_positives'] > 0 for r in records)
        summary['images'] = len(records)
        summary['false_positive_confidence_bands'] = 'See per-image highest confidence; low-threshold counts include very weak boxes.'
        with (folder/'per_image_counts.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        assert sum(r['false_positives'] for r in records) == fp
        assert sum(r['correct'] for r in records) == tp
        assert sum(r['missed'] for r in records) == fn
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    summaries = []
    for confidence in [.001,.10,.15,.25]:
        summaries.append(run('runs/camera_finetune/weights/best.pt',confidence,f'validation_conf_{confidence:g}'))
    summaries.append(run('runs/yolo11n-2/weights/best.pt',.001,'baseline_camera_validation',False))
    (OUTPUT/'comparison.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
