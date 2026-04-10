"""YOLO-based troop/spell/structure detection.

Used for the *custom replay* path (raw video → detections). The HF TV-replay
dataset already ships structured card placements and bypasses this stage.
"""

from crpod.detection.yolo import Detection, YoloDetector

__all__ = ["Detection", "YoloDetector"]
