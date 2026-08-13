import numpy as np

from src.pose.base import PoseFrame


class PoseProcessor:

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        smoothing_alpha: float = 0.4,
    ):
        self.confidence_threshold = confidence_threshold
        self.smoothing_alpha = smoothing_alpha

        self.previous_keypoints: np.ndarray | None = None

    def process(self, pose_frame: PoseFrame) -> PoseFrame:

        keypoints = pose_frame.keypoints.copy()

        confidence = keypoints[:, 2]

        if self.previous_keypoints is not None:

            low_confidence = (
                confidence < self.confidence_threshold
            )

            keypoints[low_confidence] = (
                self.previous_keypoints[low_confidence]
            )

            alpha = self.smoothing_alpha

            keypoints[:, :2] = (
                alpha * keypoints[:, :2]
                + (1 - alpha)
                * self.previous_keypoints[:, :2]
            )

        self.previous_keypoints = keypoints.copy()

        return PoseFrame(
            frame_index=pose_frame.frame_index,
            timestamp=pose_frame.timestamp,
            keypoints=keypoints,
            person_confidence=pose_frame.person_confidence,
        )