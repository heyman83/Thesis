"""Evaluate NMS thresholds at fixed confidence on the camera validation split."""
import hashlib
import json
from pathlib import Path
from review_finetune import ROOT, run


def dataset_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT/'camera_dataset/valid/labels').glob('*.txt')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update((ROOT/'camera_dataset/data.yaml').read_bytes())
    return digest.hexdigest()


def main():
    output = ROOT/'runs/nms_comparison'
    fingerprint = dataset_fingerprint()
    summaries = []
    for nms in [.7, .5, .4]:
        summary = run('runs/camera_finetune/weights/best.pt', .25,
                      f'validation_conf_0.25_nms_{nms:.2f}',
                      nms_iou=nms, output=output)
        summary['validation_annotation_fingerprint'] = fingerprint
        summaries.append(summary)
    if dataset_fingerprint() != fingerprint:
        raise RuntimeError('Validation annotations changed during comparison; rerun on one fixed version.')
    (output/'comparison.json').write_text(json.dumps(summaries,indent=2))
    lines = ['# NMS comparison on camera validation data', '',
             'Checkpoint: runs/camera_finetune/weights/best.pt. Confidence=0.25; imgsz=640; matching IoU=0.5.',
             'Counts are fixed-threshold confusion-matrix counts, not console best-F1 precision/recall.',
             'Uses current local annotations; missing target labels still produce false positives.', '',
             '| NMS IoU | Images | Targets | Correct | Missed | False alarms | Precision | Recall |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['nms_iou']:.2f} | {s['images']} | {s['correct']+s['missed']} | {s['correct']} | {s['missed']} | {s['false_positives']} | {s['precision']:.1%} | {s['recall']:.1%} |")
    lines += ['', 'Each labelled folder includes visualizations, confusion matrices, original images with false positives/misses, per_image_counts.csv, and summary.json.', '',
              'All three runs used validation annotation fingerprint: '+fingerprint]
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
