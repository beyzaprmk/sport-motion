from enum import Enum

from src.exercises.base import AnalysisResult, ExerciseAnalyzer
from src.processing.kinematics import KinematicFrame


class SquatState(Enum):
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


class SquatAnalyzer(ExerciseAnalyzer):

    def __init__(
        self,
        standing_threshold: float = 160.0,
        bottom_threshold: float = 100.0,
    ):
        self.standing_threshold = standing_threshold
        self.bottom_threshold = bottom_threshold

        self.state = SquatState.STANDING

        self.repetitions = 0

        self.knee_angles: list[float] = []
        self.rep_min_angles: list[float] = []

        self.current_rep_min_angle: float | None = None

    def update(
        self,
        kinematic_frame: KinematicFrame,
    ) -> None:

        knee_angles = [
            kinematic_frame.left_knee_angle,
            kinematic_frame.right_knee_angle,
        ]

        valid_angles = [
            angle
            for angle in knee_angles
            if angle is not None
            and angle == angle
        ]

        if not valid_angles:
            return

        knee_angle = sum(valid_angles) / len(valid_angles)

        self.knee_angles.append(knee_angle)

        if self.state == SquatState.STANDING:

            if knee_angle < self.standing_threshold:
                self.state = SquatState.DESCENDING

                self.current_rep_min_angle = knee_angle

        elif self.state == SquatState.DESCENDING:

            self.current_rep_min_angle = min(
                self.current_rep_min_angle,
                knee_angle,
            )

            if knee_angle <= self.bottom_threshold:
                self.state = SquatState.BOTTOM

        elif self.state == SquatState.BOTTOM:

            self.current_rep_min_angle = min(
                self.current_rep_min_angle,
                knee_angle,
            )

            if knee_angle > self.bottom_threshold:
                self.state = SquatState.ASCENDING

        elif self.state == SquatState.ASCENDING:

            self.current_rep_min_angle = min(
                self.current_rep_min_angle,
                knee_angle,
            )

            if knee_angle >= self.standing_threshold:

                self.repetitions += 1

                self.rep_min_angles.append(
                    self.current_rep_min_angle
                )

                self.current_rep_min_angle = None

                self.state = SquatState.STANDING

    def finalize(self) -> AnalysisResult:

        average_knee_angle = None
        minimum_knee_angle = None

        if self.knee_angles:
            average_knee_angle = (
                sum(self.knee_angles)
                / len(self.knee_angles)
            )

            minimum_knee_angle = min(
                self.knee_angles
            )

        return AnalysisResult(
            exercise="squat",
            repetitions=self.repetitions,
            metrics={
                "average_knee_angle": average_knee_angle,
                "minimum_knee_angle": minimum_knee_angle,
                "rep_min_angles": self.rep_min_angles,
            },
        )