import json
import random
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bdd100k"
CLASSES = ("car", "truck", "bus", "traffic light")


def load_annotations(path: Path):
    """Read the selected classes as (class name, pixel coordinates) pairs."""
    sample = json.loads(path.read_text(encoding="utf-8"))
    annotations = []
    for obj in sample["frames"][0].get("objects", []):
        name = obj["category"]
        if name in CLASSES and "box2d" in obj:
            box = obj["box2d"]
            annotations.append((name, (box["x1"], box["y1"], box["x2"], box["y2"])))
    return annotations


def load_frame(image_id):
    image_path = DATA / "images" / f"{image_id}.jpg"
    annotation_path = DATA / "labels" / f"{image_id}.json"
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    return image, load_annotations(annotation_path)


def select_frames(limit, seed):
    """Take a seeded sample without replacement. Zero selects every image."""
    image_ids = sorted(path.stem for path in (DATA / "images").glob("*.jpg"))
    if not image_ids:
        raise FileNotFoundError(f"No images in {DATA / 'images'}")
    if not 0 <= limit <= len(image_ids):
        raise ValueError(f"Choose 0 to {len(image_ids)} images.")
    random.Random(seed).shuffle(image_ids)
    return image_ids[:limit] if limit else image_ids
