from src.exercises.base import ExerciseAnalyzer
from src.exercises.sit_up import SitUpAnalyzer
from src.exercises.squat import SquatAnalyzer


def create_exercise_analyzer(
    exercise_name: str,
) -> ExerciseAnalyzer:

    if exercise_name == "squat":
        return SquatAnalyzer()

    if exercise_name == "sit_up":
        return SitUpAnalyzer()

    raise ValueError(
        f"Unsupported exercise: {exercise_name}"
    )