"""Compare TTA on/off on Version2 validation with fixed confidence/NMS."""
import json
from review_finetune import ROOT, run


def main():
    output = ROOT/'runs/tta_v2'
    summaries = []
    for augment in [False, True]:
        summary = run('runs/camera_finetune/weights/best.pt', .25,
                      f'tta_{"on" if augment else "off"}',
                      nms_iou=.4, output=output, data=ROOT/'Version2/data.yaml', augment=augment)
        summaries.append(summary)
    (output/'comparison.json').write_text(json.dumps(summaries,indent=2))
    lines = ['# Version2 test-time augmentation comparison', '',
             'Same fine-tuned best.pt, Version2 validation, confidence=0.25, NMS IoU=0.4, imgsz=640, batch=1, GPU 0.',
             'Fixed-threshold confusion-matrix counts at matching IoU=0.5. Both runs evaluated fresh; no training or test-set evaluation.', '',
             '| TTA | Images | Targets | Correct | Missed | False alarms | Precision | Recall | Inference ms/image |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {'On' if s['augment'] else 'Off'} | {s['images']} | {s['correct']+s['missed']} | {s['correct']} | {s['missed']} | {s['false_positives']} | {s['precision']:.1%} | {s['recall']:.1%} | {s['speed_ms_per_image']['inference']:.2f} |")
    lines += ['', 'Timing is indicative on this GTX 1050 Ti, not a Raspberry Pi benchmark or a controlled repeated speed test.',
              'Each run folder includes confusion matrices, four-panel visualizations, originals associated with false positives/misses, per-image counts, and summary.json.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
