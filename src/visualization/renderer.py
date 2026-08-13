import cv2


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
        for x, y, confidence in keypoints:

            if confidence < (
                self.confidence_threshold
            ):
                continue

            cv2.circle(
                frame,
                (int(x), int(y)),
                5,
                (0, 255, 0),
                -1,
            )

        for start_idx, end_idx in SKELETON:

            start = keypoints[start_idx]
            end = keypoints[end_idx]

            if (
                start[2]
                < self.confidence_threshold
                or end[2]
                < self.confidence_threshold
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

        # Common kinematics
       

        if (
            kinematic_frame.left_knee_angle
            is not None
        ):
            cv2.putText(
                frame,
                (
                    "Left Knee: "
                    f"{kinematic_frame.left_knee_angle:.1f}"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        if (
            kinematic_frame.right_knee_angle
            is not None
        ):
            cv2.putText(
                frame,
                (
                    "Right Knee: "
                    f"{kinematic_frame.right_knee_angle:.1f}"
                ),
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        if (
            kinematic_frame.left_hip_angle
            is not None
        ):
            cv2.putText(
                frame,
                (
                    "Left Hip: "
                    f"{kinematic_frame.left_hip_angle:.1f}"
                ),
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        if (
            kinematic_frame.right_hip_angle
            is not None
        ):
            cv2.putText(
                frame,
                (
                    "Right Hip: "
                    f"{kinematic_frame.right_hip_angle:.1f}"
                ),
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        if (
            kinematic_frame.torso_angle
            is not None
        ):
            cv2.putText(
                frame,
                (
                    "Torso: "
                    f"{kinematic_frame.torso_angle:.1f}"
                ),
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )


        # Push-up kinematics
      

        if (
            kinematic_frame.left_elbow_angle
            is not None
        ):
            cv2.putText(
                frame,
                (
                    "Left Elbow: "
                    f"{kinematic_frame.left_elbow_angle:.1f}"
                ),
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        if (
            kinematic_frame.right_elbow_angle
            is not None
        ):
            cv2.putText(
                frame,
                (
                    "Right Elbow: "
                    f"{kinematic_frame.right_elbow_angle:.1f}"
                ),
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        if (
            kinematic_frame.body_alignment_angle
            is not None
        ):
            cv2.putText(
                frame,
                (
                    "Body Alignment: "
                    f"{kinematic_frame.body_alignment_angle:.1f}"
                ),
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        if (
            kinematic_frame.body_reference_point
            is not None
        ):
            body_y = (
                kinematic_frame
                .body_reference_point[1]
            )

            cv2.putText(
                frame,
                (
                    "Body Y: "
                    f"{body_y:.1f}"
                ),
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Exercise state
        
        cv2.putText(
            frame,
            (
                "State: "
                f"{exercise_manager.state}"
            ),
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            (
                "Repetitions: "
                f"{exercise_manager.repetition_count}"
            ),
            (20, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return frame