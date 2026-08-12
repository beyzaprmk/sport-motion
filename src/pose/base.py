from abc import ABC, abstractmethod

import numpy as np

from dataclasses import dataclass


@dataclass
class PoseFrame:
    frame_index: int
    timestamp: float
    keypoints: np.ndarray
    person_confidence: float 


class PoseEstimator(ABC):

    @abstractmethod
    def estimate(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> PoseFrame | None:
        pass