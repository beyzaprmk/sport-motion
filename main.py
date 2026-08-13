import sys
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config.loader import load_config
from src.pose.factory import create_pose_estimator
from src.video.reader import VideoReader
from src.processing.processor import PoseProcessor
from src.processing.kinematics import KinematicCalculator
from src.session.manager import SessionManager
from src.visualization.renderer import (
    PoseRenderer,
    AnalysisRenderer,
)


CONFIG_PATH = "configs/config.yaml"


def main():

    config = load_config(
        CONFIG_PATH
    )

    # -------------------------
    # Session / Exercise
    # -------------------------

    session_manager = (
        SessionManager()
    )

    exercise_manager = (
        session_manager
        .create_exercise_manager()
    )

    # -------------------------
    # Video
    # -------------------------

    video_path = (
        config["video"]["path"]
    )

    reader = VideoReader(
        video_path
    )

    # -------------------------
    # Pose Estimation
    # -------------------------

    pose_estimator = (
        create_pose_estimator(
            config["pose"]
        )
    )

    # -------------------------
    # Pose Processing
    # -------------------------

    processing_config = (
        config["processing"]
    )

    pose_processor = PoseProcessor(
        confidence_threshold=(
            processing_config[
                "confidence_threshold"
            ]
        ),
        smoothing_alpha=(
            processing_config[
                "smoothing_alpha"
            ]
        ),
    )

    # -------------------------
    # Kinematics
    # -------------------------

    kinematic_calculator = (
        KinematicCalculator()
    )

    # -------------------------
    # Visualization
    # -------------------------

    pose_renderer = PoseRenderer(
        confidence_threshold=(
            processing_config[
                "confidence_threshold"
            ]
        ),
    )

    analysis_renderer = (
        AnalysisRenderer()
    )

    # -------------------------
    # Frame Processing
    # -------------------------

    for (
        frame_index,
        timestamp,
        frame,
    ) in reader:

        pose_frame = (
            pose_estimator.estimate(
                frame=frame,
                frame_index=frame_index,
                timestamp=timestamp,
            )
        )

        if pose_frame is None:
            continue

        processed_pose = (
            pose_processor.process(
                pose_frame
            )
        )

        kinematic_frame = (
            kinematic_calculator
            .calculate(
                processed_pose
            )
        )

        exercise_manager.update(
            kinematic_frame
        )

        # -------------------------
        # Visualization
        # -------------------------

        frame = (
            pose_renderer.draw_pose(
                frame,
                processed_pose.keypoints,
            )
        )

        frame = (
            analysis_renderer
            .draw_information(
                frame,
                kinematic_frame,
                exercise_manager,
            )
        )

        cv2.imshow(
            "SportMotion",
            frame,
        )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):
            break

    # -------------------------
    # Cleanup
    # -------------------------

    reader.release()

    cv2.destroyAllWindows()

    # -------------------------
    # Final Result
    # -------------------------

    result = (
        exercise_manager.finalize()
    )

    print(
        "\n=== SportMotion ==="
    )

    print(
        f"Exercise: "
        f"{result.exercise}"
    )

    print(
        f"Repetitions: "
        f"{result.repetitions}"
    )

    print(
        f"Metrics: "
        f"{result.metrics}"
    )


if __name__ == "__main__":
    main()