from dataclasses import dataclass
from time import perf_counter

import cv2
import torch
from ultralytics import YOLO

from src.bdd100k import CLASSES, ROOT

WEIGHTS_DIR = ROOT / "data" / "weights"
MODELS = ("yolo11n", "yolo11s")
CONFIDENCE = 0.25
NMS_IOU = 0.70
CPU_THREADS = 4


def inference_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class Setting:
    model: str = MODELS[0]
    size: int = 640
    confidence: float = CONFIDENCE

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"Choose a model from {', '.join(MODELS)}.")
        if self.size < 32 or self.size % 32:
            raise ValueError("Input size must be a positive multiple of 32.")
        if not 0 < self.confidence <= 1:
            raise ValueError("Confidence must be greater than 0 and at most 1.")

    @property
    def label(self):
        return f"{self.model} / {self.size} px / confidence {self.confidence:g}"


class Detector:
    def __init__(self, model=MODELS[0]):
        weights = WEIGHTS_DIR / f"{model}.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Place the {model} weights at {weights}")
        torch.set_num_threads(CPU_THREADS)
        cv2.setNumThreads(1)
        self.device = inference_device()
        print(f"Inference device: {self.device}")
        self.model = YOLO(str(weights))
        self.class_ids = [index for index, name in self.model.names.items() if name in CLASSES]

    def predict(self, image, size, confidence=CONFIDENCE):
        """Return (class, original-image xyxy box, confidence) and wall-clock ms."""
        start = perf_counter()
        result = self.model.predict(
            image, imgsz=size, conf=confidence, iou=NMS_IOU, classes=self.class_ids,
            rect=True, batch=1, device=self.device, verbose=False, save=False,
            project=str(ROOT), name="results", exist_ok=True,
        )[0]
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed_ms = (perf_counter() - start) * 1000
        boxes = result.boxes
        predictions = [
            (self.model.names[int(class_id)], box, score)
            for class_id, box, score in zip(
                boxes.cls.tolist(), boxes.xyxy.tolist(), boxes.conf.tolist()
            )
        ]
        return predictions, elapsed_ms
