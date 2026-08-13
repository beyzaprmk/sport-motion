import cv2


# COCO 17 keypoint bağlantıları
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),

    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),
]


class PoseRenderer:

    def __init__(
        self,
        confidence_threshold: float = 0.5,
    ):
        self.confidence_threshold = (
            confidence_threshold
        )

    def draw_pose(
        self,
        frame,
        keypoints,
    ):
        """
        Draw pose keypoints and skeleton.
        """

        # -------------------------
        # Keypoints
        # -------------------------

        for x, y, confidence in keypoints:

            if confidence < self.confidence_threshold:
                continue

            cv2.circle(
                frame,
                (int(x), int(y)),
                5,
                (0, 255, 0),
                -1,
            )

        # -------------------------
        # Skeleton
        # -------------------------

        for start_idx, end_idx in SKELETON:

            start = keypoints[start_idx]
            end = keypoints[end_idx]

            if (
                start[2] < self.confidence_threshold
                or end[2] < self.confidence_threshold
            ):
                continue

            start_point = (
                int(start[0]),
                int(start[1]),
            )

            end_point = (
                int(end[0]),
                int(end[1]),
            )

            cv2.line(
                frame,
                start_point,
                end_point,
                (255, 255, 255),
                2,
            )

        return frame


class AnalysisRenderer:

    def draw_information(
        self,
        frame,
        kinematic_frame,
        exercise_manager,
    ):
        """
        Draw kinematic values and exercise analysis.
        """

        left_knee = (
            kinematic_frame.left_knee_angle
        )

        right_knee = (
            kinematic_frame.right_knee_angle
        )

        cv2.putText(
            frame,
            f"Left Knee: {left_knee:.1f} deg",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Right Knee: {right_knee:.1f} deg",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Repetitions: "
            f"{exercise_manager.repetition_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return frame