from enum import Enum

import numpy as np

from src.exercises.base import (
    AnalysisResult,
    ExerciseAnalyzer,
)
from src.processing.kinematics import KinematicFrame


class PushUpState(Enum):
    TOP = "top"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


class PushUpAnalyzer(ExerciseAnalyzer):

    def __init__(
        self,
        top_elbow_threshold: float = 160.0,
        bottom_elbow_threshold: float = 100.0,
        body_movement_threshold: float = 0.01,
        elbow_movement_threshold: float = 1.0,
    ):
        self.top_elbow_threshold = (
            top_elbow_threshold
        )

        self.bottom_elbow_threshold = (
            bottom_elbow_threshold
        )

        self.body_movement_threshold = (
            body_movement_threshold
        )

        self.elbow_movement_threshold = (
            elbow_movement_threshold
        )

        self.state = PushUpState.TOP

        self.repetitions = 0

        self.previous_body_y: float | None = None
        self.previous_elbow_angle: float | None = None

        self.elbow_angles: list[float] = []
        self.body_alignment_angles: list[float] = []
        self.body_y_positions: list[float] = []

        self.rep_min_elbow_angle: float | None = None

    @property
    def repetition_count(self) -> int:
        return self.repetitions

    def _average_elbow_angle(
        self,
        frame: KinematicFrame,
    ) -> float | None:

        angles = [
            frame.left_elbow_angle,
            frame.right_elbow_angle,
        ]

        valid_angles = [
            angle
            for angle in angles
            if angle is not None
            and np.isfinite(angle)
        ]

        if not valid_angles:
            return None

        return (
            sum(valid_angles)
            / len(valid_angles)
        )

    def update(
        self,
        kinematic_frame: KinematicFrame,
    ) -> None:

        body_reference_point = (
            kinematic_frame.body_reference_point
        )

        elbow_angle = (
            self._average_elbow_angle(
                kinematic_frame
            )
        )

        # We need both signals for
        # push-up movement detection.
        if body_reference_point is None:
            return

        if elbow_angle is None:
            return

        body_y = float(
            body_reference_point[1]
        )

        if not np.isfinite(body_y):
            return

        # -------------------------
        # Store measurements
        # -------------------------

        self.elbow_angles.append(
            elbow_angle
        )

        if (
            kinematic_frame.body_alignment_angle
            is not None
            and np.isfinite(
                kinematic_frame.body_alignment_angle
            )
        ):
            self.body_alignment_angles.append(
                kinematic_frame.body_alignment_angle
            )

        self.body_y_positions.append(
            body_y
        )

        # -------------------------
        # First frame
        # -------------------------

        if (
            self.previous_body_y is None
            or self.previous_elbow_angle is None
        ):
            self.previous_body_y = body_y
            self.previous_elbow_angle = (
                elbow_angle
            )
            return

        # -------------------------
        # Frame-to-frame movement
        # -------------------------

        body_y_delta = (
            body_y
            - self.previous_body_y
        )

        elbow_delta = (
            elbow_angle
            - self.previous_elbow_angle
        )

        body_movement = (
            body_y_delta
        )

        # -------------------------
        # Movement direction
        #
        # OpenCV:
        # positive Y = downward
        # negative Y = upward
        # -------------------------

        body_moving_down = (
            body_movement
            > self.body_movement_threshold
        )

        body_moving_up = (
            body_movement
            < -self.body_movement_threshold
        )

        elbow_bending = (
            elbow_delta
            < -self.elbow_movement_threshold
        )

        elbow_extending = (
            elbow_delta
            > self.elbow_movement_threshold
        )

        # -------------------------
        # TOP
        # -------------------------

        if self.state == PushUpState.TOP:

            if (
                body_moving_down
                and elbow_bending
            ):
                self.state = (
                    PushUpState.DESCENDING
                )

                self.rep_min_elbow_angle = (
                    elbow_angle
                )

        # -------------------------
        # DESCENDING
        # -------------------------

        elif self.state == PushUpState.DESCENDING:

            if (
                self.rep_min_elbow_angle
                is None
            ):
                self.rep_min_elbow_angle = (
                    elbow_angle
                )
            else:
                self.rep_min_elbow_angle = min(
                    self.rep_min_elbow_angle,
                    elbow_angle,
                )

            if (
                elbow_angle
                <= self.bottom_elbow_threshold
            ):
                self.state = (
                    PushUpState.BOTTOM
                )

        # -------------------------
        # BOTTOM
        # -------------------------

        elif self.state == PushUpState.BOTTOM:

            if (
                body_moving_up
                and elbow_extending
            ):
                self.state = (
                    PushUpState.ASCENDING
                )

        # -------------------------
        # ASCENDING
        # -------------------------

        elif self.state == PushUpState.ASCENDING:

            if (
                elbow_angle
                >= self.top_elbow_threshold
            ):
                self.repetitions += 1

                self.state = (
                    PushUpState.TOP
                )

                self.rep_min_elbow_angle = None

        # -------------------------
        # Previous values
        # -------------------------

        self.previous_body_y = body_y
        self.previous_elbow_angle = (
            elbow_angle
        )

    def finalize(self) -> AnalysisResult:

        average_elbow_angle = None
        minimum_elbow_angle = None
        average_body_alignment = None

        if self.elbow_angles:

            average_elbow_angle = (
                sum(self.elbow_angles)
                / len(self.elbow_angles)
            )

            minimum_elbow_angle = min(
                self.elbow_angles
            )

        if self.body_alignment_angles:

            average_body_alignment = (
                sum(self.body_alignment_angles)
                / len(
                    self.body_alignment_angles
                )
            )

        return AnalysisResult(
            exercise="push_up",
            repetitions=self.repetitions,
            metrics={
                "average_elbow_angle": (
                    average_elbow_angle
                ),
                "minimum_elbow_angle": (
                    minimum_elbow_angle
                ),
                "average_body_alignment_angle": (
                    average_body_alignment
                ),
            },
        )