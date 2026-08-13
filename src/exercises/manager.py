from src.exercises.base import (
    AnalysisResult,
    ExerciseAnalyzer,
)
from src.exercises.factory import create_exercise_analyzer
from src.processing.kinematics import KinematicFrame


class ExerciseManager:

    def __init__(self, exercise_name: str):
        self.exercise_name = exercise_name

        self.analyzer = create_exercise_analyzer(
            exercise_name
        )

    def update(
        self,
        kinematic_frame: KinematicFrame,
    ) -> None:
        self.analyzer.update(
            kinematic_frame
        )

    def finalize(self) -> AnalysisResult:
        return self.analyzer.finalize()

    @property
    def repetition_count(self) -> int:
        return self.analyzer.repetition_count