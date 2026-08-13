from enum import Enum

from src.exercises.base import (
    AnalysisResult,
    ExerciseAnalyzer,
)
from src.processing.kinematics import KinematicFrame


class SitUpState(Enum):
    LYING = "lying"
    ASCENDING = "ascending"
    TOP = "top"
    DESCENDING = "descending"


class SitUpAnalyzer(ExerciseAnalyzer):

    def __init__(
        self,
        bent_knee_threshold: float = 140.0,
        lying_torso_threshold: float = 65.0,
        top_torso_threshold: float = 45.0,
        top_hip_threshold: float = 70.0,
    ):
        self.bent_knee_threshold = bent_knee_threshold
        self.lying_torso_threshold = lying_torso_threshold
        self.top_torso_threshold = top_torso_threshold
        self.top_hip_threshold = top_hip_threshold

        self.state = SitUpState.LYING

        self.repetitions = 0

        self.torso_angles: list[float] = []
        self.hip_angles: list[float] = []
        self.knee_angles: list[float] = []

        self.rep_min_torso_angle: float | None = None
        self.rep_min_hip_angle: float | None = None

    @property
    def repetition_count(self) -> int:
        return self.repetitions

    def update(
        self,
        kinematic_frame: KinematicFrame,
    ) -> None:

        knee_angles = [
            kinematic_frame.left_knee_angle,
            kinematic_frame.right_knee_angle,
        ]

        hip_angles = [
            kinematic_frame.left_hip_angle,
            kinematic_frame.right_hip_angle,
        ]

        valid_knee_angles = [
            angle
            for angle in knee_angles
            if angle is not None
            and angle == angle
        ]

        valid_hip_angles = [
            angle
            for angle in hip_angles
            if angle is not None
            and angle == angle
        ]

        torso_angle = kinematic_frame.torso_angle

        if not valid_knee_angles:
            return

        if not valid_hip_angles:
            return

        if torso_angle is None or torso_angle != torso_angle:
            return

        knee_angle = sum(valid_knee_angles) / len(
            valid_knee_angles
        )

        hip_angle = sum(valid_hip_angles) / len(
            valid_hip_angles
        )

        self.knee_angles.append(knee_angle)
        self.hip_angles.append(hip_angle)
        self.torso_angles.append(torso_angle)

        knees_bent = (
            knee_angle <= self.bent_knee_threshold
        )

        if not knees_bent:
            return

        # LYING
      

        if self.state == SitUpState.LYING:

            if torso_angle < self.lying_torso_threshold:

                self.state = SitUpState.ASCENDING

                self.rep_min_torso_angle = torso_angle
                self.rep_min_hip_angle = hip_angle

        
        # ASCENDING
       

        elif self.state == SitUpState.ASCENDING:

            self.rep_min_torso_angle = min(
                self.rep_min_torso_angle,
                torso_angle,
            )

            self.rep_min_hip_angle = min(
                self.rep_min_hip_angle,
                hip_angle,
            )

            if (
                torso_angle <= self.top_torso_threshold
                and hip_angle <= self.top_hip_threshold
            ):
                self.state = SitUpState.TOP

      
        elif self.state == SitUpState.TOP:

            self.rep_min_torso_angle = min(
                self.rep_min_torso_angle,
                torso_angle,
            )

            self.rep_min_hip_angle = min(
                self.rep_min_hip_angle,
                hip_angle,
            )

            if torso_angle > self.top_torso_threshold:
                self.state = SitUpState.DESCENDING

        # -------------------------
        # DESCENDING
        # -------------------------

        elif self.state == SitUpState.DESCENDING:

            if torso_angle >= self.lying_torso_threshold:

                self.repetitions += 1

                self.state = SitUpState.LYING

                self.rep_min_torso_angle = None
                self.rep_min_hip_angle = None

    def finalize(self) -> AnalysisResult:

        average_knee_angle = None
        average_hip_angle = None
        average_torso_angle = None

        if self.knee_angles:
            average_knee_angle = (
                sum(self.knee_angles)
                / len(self.knee_angles)
            )

        if self.hip_angles:
            average_hip_angle = (
                sum(self.hip_angles)
                / len(self.hip_angles)
            )

        if self.torso_angles:
            average_torso_angle = (
                sum(self.torso_angles)
                / len(self.torso_angles)
            )

        return AnalysisResult(
            exercise="sit_up",
            repetitions=self.repetitions,
            metrics={
                "average_knee_angle": average_knee_angle,
                "average_hip_angle": average_hip_angle,
                "average_torso_angle": average_torso_angle,
            },
        )