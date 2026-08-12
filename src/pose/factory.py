from src.pose.base import PoseEstimator
from src.pose.yolo import YOLOPoseEstimator


def create_pose_estimator(config: dict) -> PoseEstimator:
    model = config["model"]
    confidence = config.get("confidence", 0.5)

    if model.endswith(".pt"):
        return YOLOPoseEstimator(
            model_path=model,
            confidence=confidence,
        )

    raise ValueError(
        f"Unsupported pose model: {model}"
    )