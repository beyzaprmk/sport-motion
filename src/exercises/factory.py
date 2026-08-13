from src.exercises.base import ExerciseAnalyzer
from src.exercises.push_up import PushUpAnalyzer
from src.exercises.sit_up import SitUpAnalyzer
from src.exercises.squat import SquatAnalyzer


def create_exercise_analyzer(
    exercise_name: str,
) -> ExerciseAnalyzer:

    if exercise_name == "squat":
        return SquatAnalyzer()

    if exercise_name == "sit_up":
        return SitUpAnalyzer()

    if exercise_name == "push_up":
        return PushUpAnalyzer()

    raise ValueError(
        f"Unsupported exercise: {exercise_name}"
    )