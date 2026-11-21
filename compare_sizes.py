"""Compare 640/832/960 validation inference at conf .25, NMS .4, TTA off."""
import json
from review_finetune import ROOT, run


def main():
    output = ROOT/'runs/size_v2'
    summaries = [run('runs/camera_finetune/weights/best.pt',.25,f'size_{size}',
                     nms_iou=.4, output=output, data=ROOT/'Version2/data.yaml',
                     augment=False, imgsz=size) for size in [640,832,960]]
    baseline = summaries[0]['speed_ms_per_image']['inference']
    for s in summaries:
        s['inference_time_vs_640'] = s['speed_ms_per_image']['inference']/baseline
        s['inference_only_fps'] = 1000/s['speed_ms_per_image']['inference']
        s['pipeline_ms_per_image'] = sum(s['speed_ms_per_image'].values())
        (ROOT/s['folder']/'summary.json').write_text(json.dumps(s,indent=2))
    (output/'comparison.json').write_text(json.dumps(summaries,indent=2))
    lines=['# Version2 inference-size comparison','',
           'Same camera_finetune best.pt, validation split, confidence=0.25, NMS IoU=0.4, TTA off, batch=1, GPU 0.',
           'Fixed-threshold confusion-matrix counts; matching IoU=0.5. Fresh runs at all three sizes.', '',
           '| Image size | Correct | Missed | False alarms | Precision | Recall | Inference ms/image | Relative time | Preprocess ms | Postprocess ms | Pipeline ms/image |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        t=s['speed_ms_per_image']
        lines.append(f"| {s['imgsz']} | {s['correct']} | {s['missed']} | {s['false_positives']} | {s['precision']:.1%} | {s['recall']:.1%} | {t['inference']:.2f} | {s['inference_time_vs_640']:.2f}x | {t['preprocess']:.2f} | {t['postprocess']:.2f} | {s['pipeline_ms_per_image']:.2f} |")
    lines += ['', 'Timing: Ultralytics mean stage timings on this GTX 1050 Ti. Pipeline is preprocess + inference + postprocess (loss is zero in standalone validation).',
              'These are indicative single-run measurements, not controlled repeated benchmarks or Raspberry Pi FPS. Disk I/O, plotting, browser transfer and robot processing are not included.',
              'Each size folder includes annotated visualizations, confusion matrices, per-image counts, summary.json, and copies of originals with false positives or misses. No retraining or test-set evaluation.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
