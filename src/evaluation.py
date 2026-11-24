from collections import Counter

MATCH_IOU = 0.50


def box_iou(first, second):
    """Intersection divided by union for two xyxy boxes."""
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    width = max(0, min(x2, a2) - max(x1, a1))
    height = max(0, min(y2, b2) - max(y1, b1))
    intersection = width * height
    first_area = max(0, x2 - x1) * max(0, y2 - y1)
    second_area = max(0, a2 - a1) * max(0, b2 - b1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(annotations, predictions):
    """Match predictions in descending confidence, once per same-class target."""
    targets_by_class = {}
    for index, (name, box) in enumerate(annotations):
        targets_by_class.setdefault(name, []).append((index, box))

    hits = {}
    used = set()
    order = sorted(range(len(predictions)), key=lambda index: -predictions[index][2])
    for index in order:
        name, box, _ = predictions[index]
        best_target = None
        best_overlap = 0.0
        for target, truth_box in targets_by_class.get(name, ()):
            if target not in used:
                overlap = box_iou(box, truth_box)
                if overlap > best_overlap:
                    best_target, best_overlap = target, overlap
        if best_overlap >= MATCH_IOU:
            used.add(best_target)
            hits[index] = best_overlap
    return hits, [index for index in range(len(annotations)) if index not in used]


def metrics(tp, fp, fn):
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
    }


FP_REASONS = ("duplicate", "class_mismatch", "localization", "no_close_annotation")


def false_positive_reason(prediction, annotations):
    """Describe an unmatched prediction by overlap with evaluated annotations."""
    name, box, _ = prediction
    same_class, other_class = 0.0, 0.0
    for label, target in annotations:
        overlap = box_iou(box, target)
        if name == label:
            same_class = max(same_class, overlap)
        else:
            other_class = max(other_class, overlap)
    if same_class >= MATCH_IOU:
        return "duplicate"
    if other_class >= MATCH_IOU:
        return "class_mismatch"
    if same_class >= 0.10:
        return "localization"
    return "no_close_annotation"


def detection_confusions(annotations, predictions, hits, missed):
    """Pair mutually unmatched boxes for class errors; count leftovers as background."""
    available = set(missed)
    counts = Counter()
    for index in sorted(range(len(predictions)), key=lambda i: -predictions[i][2]):
        name, box, _ = predictions[index]
        actual = name if index in hits else "background"
        if index not in hits:
            best_overlap, best_target = 0.0, None
            for candidate in sorted(available):
                label, target = annotations[candidate]
                overlap = box_iou(box, target)
                if label != name and overlap > best_overlap:
                    best_overlap, best_target = overlap, candidate
            if best_overlap >= MATCH_IOU:
                actual = annotations[best_target][0]
                available.remove(best_target)
        counts[actual, name] += 1
    for index in available:
        counts[annotations[index][0], "background"] += 1
    return counts
