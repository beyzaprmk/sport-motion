from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.processing.kinematics import KinematicFrame


@dataclass
class AnalysisResult:
    exercise: str
    repetitions: int
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class ExerciseAnalyzer(ABC):

    @abstractmethod
    def update(
        self,
        kinematic_frame: KinematicFrame,
    ) -> None:
        pass

    @abstractmethod
    def finalize(self) -> AnalysisResult:
        pass