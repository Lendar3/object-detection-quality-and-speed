import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from PIL import Image, ImageDraw, ImageFont

from src.bdd100k import CLASSES, ROOT, load_frame
from src.detector import Setting
from src.evaluation import FP_REASONS
from src.experiment import FIELDS, SUMMARY

AXIS_LABELS = {"resolution": "Input size", "confidence": "Confidence threshold", "model": "Model"}
FIGURES = ROOT / "results" / "figures"


def draw_boxes(image, boxes, font):
    draw = ImageDraw.Draw(image)
    labels = []
    for box, text, color in boxes:
        x1, y1, x2, y2 = map(round, box)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        labels.append((x1, y1, text, color))

    occupied = []
    for x1, y1, text, color in sorted(labels, key=lambda label: label[1]):
        width = round(font.getlength(text)) + 4
        x = max(0, min(x1 + 2, image.width - width))
        y = max(0, min(y1 - 19, image.height - 19))
        for left, top, right in occupied:
            if x < right and x + width > left and abs(y - top) < 19:
                y = top + 19
        if y + 19 <= image.height:
            draw.text((x, y), text, fill=color, font=font,
                      stroke_width=2, stroke_fill="#151921")
            occupied.append((x, y, x + width))


def save_comparison(image, truth, predictions, hits, missed, setting, image_id, output):
    font = ImageFont.load_default(size=16)
    truth_panel, prediction_panel = image.copy(), image.copy()
    missed = set(missed)
    truth_boxes = [
        (box, f"FN {name}", "#6cd9ff") if index in missed else (box, name, "#73efa2")
        for index, (name, box) in enumerate(truth)
    ]
    prediction_boxes = []
    for index, (name, box, score) in enumerate(predictions):
        label, color = ("TP", "#73efa2") if index in hits else ("FP", "#ff9d76")
        prediction_boxes.append((box, f"{label} {name} {score:.2f}", color))
    draw_boxes(truth_panel, truth_boxes, font)
    draw_boxes(prediction_panel, prediction_boxes, font)

    header, gap = 36, 8
    comparison = Image.new("RGB", (2 * image.width + gap, image.height + header), "#151921")
    comparison.paste(truth_panel, (0, header))
    comparison.paste(prediction_panel, (image.width + gap, header))
    titles = ImageDraw.Draw(comparison)
    titles.text((12, 9), f"BDD100K annotations | {image_id} | {len(missed)} FN", fill="white", font=font)
    titles.text((image.width + gap + 12, 9),
                f"{setting.label} | {len(hits)} TP, {len(predictions) - len(hits)} FP",
                fill="white", font=font)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output, quality=93)


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def configuration_label(row, mode):
    if mode == "resolution":
        return f"{row['size']} px"
    return row[FIELDS[mode]]


def plot_quality_and_speed(rows, frame_budget_ms, mode, comparison_split, output):
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for split, title, marker, color in (
        (comparison_split, "Validation" if comparison_split == "val" else f"Exploratory ({comparison_split})",
         "o", "#226eb2"),
        ("test", "Held-out test", "D", "#b24e21"),
    ):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            continue
        x = [float(row["p95_ms"]) for row in selected]
        y = [float(row["f1"]) for row in selected]
        ax.plot(x, y, marker=marker, linestyle="-" if split != "test" else "none",
                color=color, label=title)
        for row, latency, f1 in zip(selected, x, y):
            ax.annotate(configuration_label(row, mode), (latency, f1), xytext=(6, -13 if split != "test" else 8),
                        textcoords="offset points", color=color, fontsize=9)
    ax.axvline(frame_budget_ms, ls="--", color="#6f747a", label="30 FPS frame budget")
    ax.set(xlabel="p95 prediction latency (ms)", ylabel="Micro F1 at IoU ≥ 0.50",
           title="Detection quality and latency on BDD100K", ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    save_figure(fig, output)


def plot_class_recall(rows, mode, comparison_split, output):
    validation = [row for row in rows if row["split"] == comparison_split]
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for name in CLASSES:
        class_rows = [row for row in validation if row["class"] == name]
        labels = [configuration_label(row, mode) for row in class_rows]
        recall = [float(row["recall"]) for row in class_rows]
        ax.plot(labels, recall, marker="o", label=name)
    ax.set(xlabel=AXIS_LABELS[mode], ylabel="Recall at IoU ≥ 0.50",
           title=f"Recall by class ({comparison_split})", ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2, loc="upper left")
    save_figure(fig, output)


def plot_confidence_tradeoff(rows, output):
    rows = sorted(rows, key=lambda row: float(row["confidence"]))
    thresholds = [float(row["confidence"]) for row in rows]
    precision = [float(row["precision"]) for row in rows]
    recall = [float(row["recall"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    ax.plot(thresholds, precision, marker="o", color="#226eb2", label="Precision")
    ax.plot(thresholds, recall, marker="s", color="#b24e21", label="Recall")
    ax.set(xlabel="Confidence threshold", ylabel="Score at IoU ≥ 0.50",
           title="Confidence changes the precision–recall trade-off", ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output / "confidence_tradeoff.png")

    fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
    ax.plot(recall, precision, marker="o", color="#226eb2")
    for x, y, value in zip(recall, precision, thresholds):
        ax.annotate(f"{value:g}", (x, y), xytext=(6, 6), textcoords="offset points")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall operating points",
           xlim=(0, 1), ylim=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.25)
    save_figure(fig, output / "precision_recall.png")


def plot_error_breakdown(rows, mode, output):
    fig, (fp_ax, fn_ax) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    labels = [configuration_label(row, mode) for row in rows]
    bottom = np.zeros(len(rows))
    for reason in FP_REASONS:
        counts = np.array([int(row[reason]) for row in rows])
        fp_ax.bar(labels, counts, bottom=bottom, label=reason.replace("_", " "))
        bottom += counts
    xlabel = AXIS_LABELS[mode]
    fp_ax.set(title="False positives by type", ylabel="Box count", xlabel=xlabel)
    fp_ax.legend(fontsize=8)
    fn_ax.bar(labels, [int(row["fn"]) for row in rows], color="#226eb2")
    fn_ax.set(title="False negatives", ylabel="Missed boxes", xlabel=xlabel)
    save_figure(fig, output)


def plot_detection_confusion(rows, run, output):
    selected = run["highest_quality"]
    matching = [row for row in rows if row["model"] == selected["model"]
                and int(row["size"]) == selected["size"]
                and float(row["confidence"]) == selected["confidence"]]
    test_rows = [row for row in matching if row["split"] == "test"]
    matching = test_rows or matching
    if not matching:
        return
    names = (*CLASSES, "background")
    counts = np.zeros((len(names), len(names)), dtype=int)
    for row in matching:
        counts[names.index(row["actual"]), names.index(row["predicted"])] = int(row["count"])
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = counts / np.maximum(1, row_totals)
    fig, ax = plt.subplots(figsize=(7.4, 6), layout="constrained")
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for y, x in np.ndindex(counts.shape):
        ax.text(x, y, f"{counts[y, x]:,}", ha="center", va="center",
                color="white" if normalized[y, x] > 0.55 else "#172338")
    label = Setting(**selected).label
    ax.set(title=f"Detection confusion matrix ({matching[0]['split']})\n{label}",
           xlabel="Predicted class", ylabel="Annotated class",
           xticks=range(len(names)), xticklabels=names, yticks=range(len(names)), yticklabels=names)
    ax.tick_params(axis="x", rotation=26)
    save_figure(fig, output)


def plot_saved_results(summary, output=None):
    output = output or (FIGURES if summary == SUMMARY else summary / "figures")
    run = json.loads((summary / "run.json").read_text(encoding="utf-8"))
    mode = run.get("mode", "resolution")
    comparison_split = run.get("comparison_split", "val")
    rows = read_rows(summary / "metrics.csv")
    comparison_rows = [row for row in rows if row["split"] == comparison_split]
    plot_quality_and_speed(rows, run["frame_budget_ms"], mode, comparison_split,
                           output / "quality_vs_latency.png")
    plot_class_recall(read_rows(summary / "class_metrics.csv"), mode, comparison_split,
                      output / "class_recall.png")
    if mode == "confidence":
        plot_confidence_tradeoff(comparison_rows, output)
    if comparison_rows and FP_REASONS[0] in comparison_rows[0]:
        plot_error_breakdown(comparison_rows, mode, output / "error_breakdown.png")
    confusion_path = summary / "confusion.csv"
    if confusion_path.is_file():
        plot_detection_confusion(read_rows(confusion_path), run, output / "confusion_matrix.png")
    examples = json.loads((summary / "examples.json").read_text(encoding="utf-8"))
    for sample in examples:
        image_id, split = sample["image_id"], sample["split"]
        setting = Setting(sample.get("model", "yolo11n"), sample["size"], sample.get("confidence", 0.25))
        image, _ = load_frame(Path(image_id).name)  # Older runs stored split/ID.
        hits = {int(index): iou for index, iou in sample["hits"].items()}
        if "model" in sample:
            prefix = f"{setting.model}_{setting.size}_{setting.confidence:g}"
        else:
            prefix = str(setting.size)
        display_id = image_id.replace("/", "_")
        path = output / "errors" / f"{split}_{prefix}_{sample['error']}_{display_id}.jpg"
        save_comparison(image, sample["truth"], sample["predictions"], hits, sample["missed"],
                        setting, image_id, path)
    print(f"Saved charts and example frames to {output}")
