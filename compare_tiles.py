"""Compare 320-pixel tiles against whole-image inference on Version2 validation."""
import json
from review_finetune import ROOT, run


def main():
    output = ROOT/'runs/tile_v2'
    baseline_path = output/'whole_640/summary.json'
    baseline_summary = json.loads(baseline_path.read_text()) if baseline_path.exists() else run(
        'runs/camera_finetune/weights/best.pt',.25,'whole_640',nms_iou=.4,
        output=output,data=ROOT/'Version2/data.yaml',augment=False,imgsz=640)
    summaries = [baseline_summary, run('runs/camera_finetune/weights/best.pt',.25,'tiles_320',
                     nms_iou=.4, output=output, data=ROOT/'Version2/data.yaml',
                     augment=False, imgsz=640, tile=320)]
    baseline=summaries[0]['speed_ms_per_image']['inference']
    for s in summaries:
        s['inference_time_vs_whole'] = s['speed_ms_per_image']['inference']/baseline
        s['pipeline_ms_per_source_image'] = sum(s['speed_ms_per_image'].values())
        (ROOT/s['folder']/'summary.json').write_text(json.dumps(s,indent=2))
    (output/'comparison.json').write_text(json.dumps(summaries,indent=2))
    lines=['# Version2 tiled inference comparison','',
           'Same camera_finetune best.pt, validation split, confidence=0.25, NMS IoU=0.4, TTA off, model input imgsz=640, batch=1, GTX 1050 Ti.',
           'Tiled configuration: tile=320 (320 x 320 source windows), built-in overlap, each window resized to model input. Predictions merged before scoring.',
           'Fixed-threshold counts at matching IoU=0.5. Both runs evaluated fresh. No retraining or test-set evaluation.', '',
           '| Mode | Images | Targets | Correct | Missed | False alarms | Precision | Recall | Inference ms/source image | Relative inference time | Postprocess/merge ms | Pipeline ms/source image |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        t=s['speed_ms_per_image']
        lines.append(f"| {'320 tiles' if s['tile'] else 'Whole image 640'} | {s['images']} | {s['correct']+s['missed']} | {s['correct']} | {s['missed']} | {s['false_positives']} | {s['precision']:.1%} | {s['recall']:.1%} | {t['inference']:.2f} | {s['inference_time_vs_whole']:.2f}x | {t['postprocess']:.2f} | {s['pipeline_ms_per_source_image']:.2f} |")
    lines += ['',f"Tiled run used {summaries[1]['total_tiles']} total windows across {summaries[1]['images']} original images.",
              'Ultralytics timings are normalized per original source image, not per tile; postprocess includes tile merging.',
              'Pipeline includes preprocess, inference and postprocess; excludes I/O, plots, browser transmission and robot control. Single-run indicative timings, not Pi benchmarks.',
              'Each labelled folder contains standard plots, 53 four-panel visualizations, per-image counts, summary.json, and copies of originals with false positives/misses.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
