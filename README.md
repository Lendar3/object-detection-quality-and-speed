# Object detection quality and speed

A Python experiment comparing YOLO11 detectors on BDD100K road scenes. Eight configurations were evaluated on the same 10,000 images (seed 12) across four classes: car, truck, bus and traffic light.

| Experiment | Fixed settings | Observed result |
| --- | --- | --- |
| Input resolution | YOLO11n, confidence 0.25 | Increasing 320 to 640 px raised micro F1 from 0.278 to 0.441. |
| Confidence threshold | YOLO11s, 480 px | Raising confidence from 0.15 to 0.50 increased precision from 59.0% to 90.2% and reduced recall from 38.5% to 22.6%. |
| Model size | 640 px, confidence 0.15 | YOLO11s reached F1 0.518 versus 0.454 for YOLO11n. |

The highest-F1 configuration ran at 63.3 FPS with a p95 prediction latency of 19.92 ms on the measured Apple GPU. Detection matching requires the same class and IoU of at least 0.50. Timing covers prediction after warm-up.

[Explore the saved metrics, error examples and plots](results/runs/).

## Run locally

Use Python 3.13 and install the dependencies in a virtual environment.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place BDD100K road images in `data/bdd100k/images/<id>.jpg` and their matching 2018 per-image annotations in `data/bdd100k/labels/<id>.json`. The loader expects objects under `frames[0].objects` with a category and pixel-coordinate box2d. Place the pretrained Ultralytics weights `yolo11n.pt` and `yolo11s.pt` in `data/weights/`.

```sh
python main.py
# Reproduce the confidence experiment
python main.py experiment --images 10000 --seed 12 --mode confidence --model yolo11s --size 480 --confidences 0.15 0.25 0.50
```

The menu lets you choose the sample size, seed and settings. Each run writes per-frame and aggregate metrics and figures to its own directory under `results/runs/`. Dataset loading, detection, evaluation, experiment execution and plotting are implemented in separate modules under [src](src/).
