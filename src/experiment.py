import csv
import json
import platform
from collections import Counter
from dataclasses import asdict, replace
from tempfile import mkdtemp
from pathlib import Path

import numpy as np

from src.bdd100k import CLASSES, ROOT, load_frame, select_frames
from src.detector import CPU_THREADS, MODELS, NMS_IOU, Detector, Setting
from src.evaluation import FP_REASONS, MATCH_IOU, detection_confusions, false_positive_reason, match_boxes, metrics

SIZES = (320, 480, 640)
CONFIDENCES = (0.15, 0.25, 0.50)
VALUES = {"resolution": SIZES, "confidence": CONFIDENCES, "model": MODELS}
FIELDS = {"resolution": "size", "confidence": "confidence", "model": "model"}
SEED = 12
DEFAULT_IMAGES = 400
FRAME_BUDGET_MS = 1000 / 30
SUMMARY = ROOT / "results" / "summary"
RUNS = ROOT / "results" / "runs"
WARMUP = 5
FRAME_COLUMNS = ("split", "model", "size", "confidence", "image_id", "tp", "fp", "fn",
                 *FP_REASONS, "missed_classes", "latency_ms")


def experiment_settings(mode, base, values=None):
    if mode not in FIELDS:
        raise ValueError("Choose resolution, confidence or model.")
    values = list(dict.fromkeys(VALUES[mode] if values is None else values))
    if len(values) < 2:
        raise ValueError("Enter at least two different values to compare.")
    return [replace(base, **{FIELDS[mode]: value}) for value in values]


def evaluate(detector, split, image_ids, setting, frame_writer):
    parameters = asdict(setting)
    for image_id in image_ids[:WARMUP]:
        image, _ = load_frame(image_id)
        detector.predict(image, setting.size, setting.confidence)

    class_counts = {name: Counter(tp=0, fp=0, fn=0) for name in CLASSES}
    timings = []
    iou_sum = 0.0
    examples = {}
    reason_totals = Counter()
    confusions = Counter()
    for number, image_id in enumerate(image_ids, start=1):
        image, truth = load_frame(image_id)
        predictions, milliseconds = detector.predict(image, setting.size, setting.confidence)
        timings.append(milliseconds)
        hits, missed = match_boxes(truth, predictions)
        iou_sum += sum(hits.values())
        for index, (name, _, _) in enumerate(predictions):
            class_counts[name]["tp" if index in hits else "fp"] += 1
        for index in missed:
            class_counts[truth[index][0]]["fn"] += 1

        reasons = {index: false_positive_reason(prediction, truth)
                   for index, prediction in enumerate(predictions) if index not in hits}
        frame_reasons = Counter(reasons.values())
        reason_totals.update(frame_reasons)
        confusions.update(detection_confusions(truth, predictions, hits, missed))
        missed_classes = Counter(truth[i][0] for i in missed)
        frame_writer.writerow({
            "split": split, **parameters, "image_id": image_id,
            "tp": len(hits), "fp": len(reasons), "fn": len(missed),
            **{name: frame_reasons[name] for name in FP_REASONS},
            "missed_classes": ", ".join(f"{name}: {count}" for name, count in missed_classes.items()),
            "latency_ms": round(milliseconds, 4),
        })
        box_count = len(truth) + len(predictions)
        for error, count in (("fp", len(reasons)), ("fn", len(missed))):
            if count:
                # Favor frames with a few visible errors and room for labels.
                rank = (box_count <= 18, min(count, 4), -box_count)
                if error not in examples or rank > examples[error][0]:
                    examples[error] = (rank, {
                        "split": split, **parameters, "image_id": image_id,
                        "truth": truth, "predictions": predictions, "hits": hits, "missed": missed,
                        "fp_reasons": reasons,
                    })
        if number % 500 == 0:
            print(f"{split} {setting.label}: {number}/{len(image_ids)} frames", flush=True)

    mean_ms = float(np.mean(timings))
    totals = {key: sum(counts[key] for counts in class_counts.values()) for key in ("tp", "fp", "fn")}
    overall = {
        "split": split, **parameters, "images": len(image_ids), **metrics(**totals),
        "mean_matched_iou": iou_sum / totals["tp"] if totals["tp"] else None,
        "mean_ms": mean_ms, "p95_ms": float(np.percentile(timings, 95)),
        "fps": 1000 / mean_ms, **{name: reason_totals[name] for name in FP_REASONS},
    }
    per_class = [
        {"split": split, **parameters, "class": name, **metrics(**counts)}
        for name, counts in class_counts.items()
    ]
    print(f"{split} {setting.label}: F1={overall['f1']:.4f}, {overall['fps']:.1f} FPS, "
          f"p95={overall['p95_ms']:.1f} ms", flush=True)
    confusion_rows = [
        {"split": split, **parameters, "actual": actual, "predicted": predicted, "count": count}
        for (actual, predicted), count in sorted(confusions.items())
    ]
    return overall, per_class, confusion_rows, [dict(sample, error=error) for error, (_, sample) in examples.items()]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(count=DEFAULT_IMAGES, output=None, *, mode="resolution", base=None, values=None, seed=SEED):
    """Compare all configurations on the same sample of labeled images."""
    settings = experiment_settings(mode, base or Setting(), values)
    comparison_ids = select_frames(count, seed)
    models = dict.fromkeys(setting.model for setting in settings)
    detectors = {model: Detector(model) for model in models}

    if output is None:
        RUNS.mkdir(parents=True, exist_ok=True)
        prefix = f"{mode}_{len(comparison_ids)}_"
        output = Path(mkdtemp(prefix=prefix, dir=RUNS))
    else:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {output}", flush=True)

    rows, class_rows, confusion_rows, examples = [], [], [], []
    with (output / "frame_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FRAME_COLUMNS)
        writer.writeheader()

        for setting in settings:
            result, classes, pairs, samples = evaluate(
                detectors[setting.model], "all", comparison_ids, setting, writer
            )
            rows.append(result)
            class_rows.extend(classes)
            confusion_rows.extend(pairs)
            examples.extend(samples)

    best = min(range(len(rows)), key=lambda i: (-rows[i]["f1"], rows[i]["p95_ms"]))
    fast = min(range(len(rows)), key=lambda i: rows[i]["p95_ms"])
    within_budget = rows[fast]["p95_ms"] <= FRAME_BUDGET_MS
    print(f"Best sample F1: {settings[best].label}", flush=True)
    print(f"Lowest sample p95: {settings[fast].label}; within 30 FPS budget: {within_budget}", flush=True)

    metadata = {
        "dataset": "BDD100K 2018 annotations", "mode": mode, "comparison_split": "all",
        "classes": CLASSES, "device": detectors[settings[0].model].device, "platform": platform.platform(),
        "cpu_threads": CPU_THREADS, "seed": seed,
        "comparison_settings": [asdict(setting) for setting in settings],
        "highest_quality": asdict(settings[best]), "fastest": asdict(settings[fast]),
        "fastest_meets_budget": within_budget,
        "nms_iou": NMS_IOU, "matching_iou": MATCH_IOU, "warmup_frames": WARMUP,
        "frame_budget_ms": FRAME_BUDGET_MS, "comparison_images": len(comparison_ids),
    }
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "class_metrics.csv", class_rows)
    if confusion_rows:
        write_csv(output / "confusion.csv", confusion_rows)
    (output / "examples.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output
