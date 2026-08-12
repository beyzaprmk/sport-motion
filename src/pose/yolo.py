import numpy as np
from ultralytics import YOLO

from src.pose.base import PoseEstimator, PoseFrame


class YOLOPoseEstimator(PoseEstimator):

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def estimate(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> PoseFrame | None:

        results = self.model(
            frame,
            verbose=False,
            conf=self.confidence,
        )

        result = results[0]

        if result.keypoints is None:
            return None

        if len(result.keypoints.data) == 0:
            return None

        if result.boxes is not None:
            person_index = int(
                result.boxes.conf.argmax()
            )
            person_confidence = float(
                result.boxes.conf[person_index].item()
            )
        else:
            person_index = 0
            person_confidence = 0.0

        keypoints = (
            result.keypoints.data[person_index]
            .detach()
            .cpu()
            .numpy()
        )

        return PoseFrame(
            frame_index=frame_index,
            timestamp=timestamp,
            keypoints=keypoints,
            person_confidence=person_confidence,
        )