# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator
from .compress import DetectionCompressor, DetectionFinetune

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "DetectionCompressor, DetectionFinetune"
