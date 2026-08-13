import numpy as np


def exponential_smoothing(
    current: np.ndarray,
    previous: np.ndarray,
    alpha: float,
) -> np.ndarray:

    if not 0 < alpha <= 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    return (
        alpha * current
        + (1.0 - alpha) * previous
    )