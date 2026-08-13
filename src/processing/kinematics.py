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

    left_elbow_angle: float | None = None
    right_elbow_angle: float | None = None

    torso_angle: float | None = None

    body_alignment_angle: float | None = None

    body_center: np.ndarray | None = None
    body_perpendicular_position: float | None = None

    left_hand_body_axis_distance: float | None = None
    right_hand_body_axis_distance: float | None = None


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


def calculate_elbow_angle(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
) -> float:

    return calculate_angle(
        shoulder,
        elbow,
        wrist,
    )


def calculate_torso_angle(
    shoulder: np.ndarray,
    hip: np.ndarray,
) -> float:

    torso_vector = shoulder - hip

    vertical_vector = np.array(
        [0.0, -1.0]
    )

    denominator = (
        np.linalg.norm(torso_vector)
        * np.linalg.norm(vertical_vector)
    )

    if denominator == 0:
        return float("nan")

    cosine = (
        np.dot(
            torso_vector,
            vertical_vector,
        )
        / denominator
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return degrees(
        np.arccos(cosine)
    )


def calculate_body_alignment_angle(
    shoulder_center: np.ndarray,
    hip_center: np.ndarray,
    ankle_center: np.ndarray,
) -> float:

    return calculate_angle(
        shoulder_center,
        hip_center,
        ankle_center,
    )


def calculate_body_center(
    shoulder_center: np.ndarray,
    hip_center: np.ndarray,
    ankle_center: np.ndarray,
) -> np.ndarray:

    return (
        shoulder_center
        + hip_center
        + ankle_center
    ) / 3.0


def calculate_body_perpendicular_position(
    body_center: np.ndarray,
    shoulder_center: np.ndarray,
    ankle_center: np.ndarray,
) -> float:

    body_axis = ankle_center - shoulder_center

    axis_length = np.linalg.norm(body_axis)

    if axis_length == 0:
        return float("nan")

    axis = body_axis / axis_length

    perpendicular = np.array(
        [-axis[1], axis[0]]
    )

    return float(
        np.dot(
            body_center,
            perpendicular,
        )
    )


def calculate_point_to_body_axis_distance(
    point: np.ndarray,
    shoulder_center: np.ndarray,
    ankle_center: np.ndarray,
) -> float:

    body_axis = ankle_center - shoulder_center

    axis_length = np.linalg.norm(body_axis)

    if axis_length == 0:
        return float("nan")

    relative_point = point - shoulder_center

    cross_product = (
        body_axis[0] * relative_point[1]
        - body_axis[1] * relative_point[0]
    )

    return abs(
        cross_product
    ) / axis_length


class KinematicCalculator:

    def calculate(
        self,
        pose_frame: PoseFrame,
    ) -> KinematicFrame:

        keypoints = pose_frame.keypoints

        # Lower body

        left_hip = keypoints[
            Keypoint.LEFT_HIP
        ][:2]

        left_knee = keypoints[
            Keypoint.LEFT_KNEE
        ][:2]

        left_ankle = keypoints[
            Keypoint.LEFT_ANKLE
        ][:2]

        right_hip = keypoints[
            Keypoint.RIGHT_HIP
        ][:2]

        right_knee = keypoints[
            Keypoint.RIGHT_KNEE
        ][:2]

        right_ankle = keypoints[
            Keypoint.RIGHT_ANKLE
        ][:2]

        # Upper body

        left_shoulder = keypoints[
            Keypoint.LEFT_SHOULDER
        ][:2]

        right_shoulder = keypoints[
            Keypoint.RIGHT_SHOULDER
        ][:2]

        left_elbow = keypoints[
            Keypoint.LEFT_ELBOW
        ][:2]

        right_elbow = keypoints[
            Keypoint.RIGHT_ELBOW
        ][:2]

        left_wrist = keypoints[
            Keypoint.LEFT_WRIST
        ][:2]

        right_wrist = keypoints[
            Keypoint.RIGHT_WRIST
        ][:2]

        # Centers

        shoulder_center = (
            left_shoulder
            + right_shoulder
        ) / 2.0

        hip_center = (
            left_hip
            + right_hip
        ) / 2.0

        ankle_center = (
            left_ankle
            + right_ankle
        ) / 2.0

        # Knee angles

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

        # Hip angles

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

        # Elbow angles
        

        left_elbow_angle = calculate_elbow_angle(
            left_shoulder,
            left_elbow,
            left_wrist,
        )

        right_elbow_angle = calculate_elbow_angle(
            right_shoulder,
            right_elbow,
            right_wrist,
        )

        # Torso angle

        torso_angle = calculate_torso_angle(
            shoulder_center,
            hip_center,
        )

        # Body alignment

        body_alignment_angle = (
            calculate_body_alignment_angle(
                shoulder_center,
                hip_center,
                ankle_center,
            )
        )

        # Body center

        body_center = calculate_body_center(
            shoulder_center,
            hip_center,
            ankle_center,
        )

        # Body movement axis

        body_perpendicular_position = (
            calculate_body_perpendicular_position(
                body_center,
                shoulder_center,
                ankle_center,
            )
        )

        # Hand position

        left_hand_body_axis_distance = (
            calculate_point_to_body_axis_distance(
                left_wrist,
                shoulder_center,
                ankle_center,
            )
        )

        right_hand_body_axis_distance = (
            calculate_point_to_body_axis_distance(
                right_wrist,
                shoulder_center,
                ankle_center,
            )
        )

        return KinematicFrame(
            left_knee_angle=left_knee_angle,
            right_knee_angle=right_knee_angle,

            left_hip_angle=left_hip_angle,
            right_hip_angle=right_hip_angle,

            left_elbow_angle=left_elbow_angle,
            right_elbow_angle=right_elbow_angle,

            torso_angle=torso_angle,

            body_alignment_angle=(
                body_alignment_angle
            ),

            body_center=body_center,

            body_perpendicular_position=(
                body_perpendicular_position
            ),

            left_hand_body_axis_distance=(
                left_hand_body_axis_distance
            ),

            right_hand_body_axis_distance=(
                right_hand_body_axis_distance
            ),
        )