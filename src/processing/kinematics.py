from dataclasses import dataclass
from math import degrees

import numpy as np

from src.pose.base import Keypoint, PoseFrame

@dataclass
class KinematicFrame:
    left_knee_angle: float | None = None
    right_knee_angle: float | None = None

    left_hip_angle: float | None = None
    right_hip_angle: float | None = None

    torso_angle: float | None = None


def calculate_angle(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator == 0:
        return float("nan")

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return degrees(
        np.arccos(cosine)
    )


def calculate_knee_angle(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
) -> float:

    return calculate_angle(
        hip,
        knee,
        ankle,
    )


def calculate_hip_angle(
    shoulder: np.ndarray,
    hip: np.ndarray,
    knee: np.ndarray,
) -> float:
   
    return calculate_angle(
        shoulder,
        hip,
        knee,
    )


class KinematicCalculator:

    def calculate(
        self,
        pose_frame: PoseFrame,
    ) -> KinematicFrame:

        keypoints = pose_frame.keypoints

        left_hip = keypoints[Keypoint.LEFT_HIP][:2]
        left_knee = keypoints[Keypoint.LEFT_KNEE][:2]
        left_ankle = keypoints[Keypoint.LEFT_ANKLE][:2]

        right_hip = keypoints[Keypoint.RIGHT_HIP][:2]
        right_knee = keypoints[Keypoint.RIGHT_KNEE][:2]
        right_ankle = keypoints[Keypoint.RIGHT_ANKLE][:2]

        left_shoulder = keypoints[
            Keypoint.LEFT_SHOULDER
        ][:2]

        right_shoulder = keypoints[
            Keypoint.RIGHT_SHOULDER
        ][:2]

        left_knee_angle = calculate_knee_angle(
            left_hip,
            left_knee,
            left_ankle,
        )

        right_knee_angle = calculate_knee_angle(
            right_hip,
            right_knee,
            right_ankle,
        )

        left_hip_angle = calculate_hip_angle(
            left_shoulder,
            left_hip,
            left_knee,
        )

        right_hip_angle = calculate_hip_angle(
            right_shoulder,
            right_hip,
            right_knee,
        )

        return KinematicFrame(
            left_knee_angle=left_knee_angle,
            right_knee_angle=right_knee_angle,
            left_hip_angle=left_hip_angle,
            right_hip_angle=right_hip_angle,
        )