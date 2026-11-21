"""Evaluate Version2 validation at NMS 0.4 and confidence 0.10/0.15/0.25."""
import json
from review_finetune import ROOT, run


def main():
    output = ROOT/'runs/version2_nms040'
    summaries = [run('runs/camera_finetune/weights/best.pt', confidence,
                     f'validation_conf_{confidence:.2f}_nms_0.40', nms_iou=.4,
                     output=output, data=ROOT/'Version2/data.yaml')
                 for confidence in [.10, .15, .25]]
    (output/'comparison.json').write_text(json.dumps(summaries,indent=2))
    lines = ['# Version2 validation comparison', '',
             'Existing camera_finetune best.pt; Version2/data.yaml; imgsz=640; NMS IoU=0.4; matching IoU=0.5.',
             'Fixed-threshold confusion-matrix counts. No training or test-split evaluation performed.', '',
             '| Confidence | Images | Targets | Correct | Missed | False alarms | Precision | Recall |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['confidence']:.2f} | {s['images']} | {s['correct']+s['missed']} | {s['correct']} | {s['missed']} | {s['false_positives']} | {s['precision']:.1%} | {s['recall']:.1%} |")
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
