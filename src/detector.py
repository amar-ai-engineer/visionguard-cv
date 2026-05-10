import numpy as np
from PIL import Image

# Classes that represent PPE violations
VIOLATION_CLASSES = {"no-helmet", "no-vest"}
COMPLIANCE_CLASSES = {"helmet", "vest"}


class SafetyDetector:
    def __init__(self, model_path="ppe_model.pt"):
        self._model_path = model_path
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)

    def detect(self, frame: np.ndarray, confidence: float = 0.25):
        """
        Run PPE detection on a numpy frame.
        Returns (annotated_frame, stats_dict).
        stats_dict keys: persons, helmets, no_helmets, vests, no_vests, violations
        """
        self._load()
        results = self._model.predict(frame, conf=confidence, verbose=False)
        annotated = results[0].plot()

        stats = {"persons": 0, "helmets": 0, "no_helmets": 0, "vests": 0, "no_vests": 0}
        for box in results[0].boxes:
            cls_name = self._model.names[int(box.cls[0])]
            if cls_name == "person":
                stats["persons"] += 1
            elif cls_name == "helmet":
                stats["helmets"] += 1
            elif cls_name == "no-helmet":
                stats["no_helmets"] += 1
            elif cls_name == "vest":
                stats["vests"] += 1
            elif cls_name == "no-vest":
                stats["no_vests"] += 1

        stats["violations"] = stats["no_helmets"] + stats["no_vests"]
        return annotated, stats

    def detect_pil(self, image: Image.Image, confidence: float = 0.25):
        return self.detect(np.array(image), confidence)
