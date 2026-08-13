import sys
from pathlib import Path

# Projenin kök dizinini (src'nin bir üst klasörü) sys.path'e ekler
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config.loader import load_config
from src.pose.factory import create_pose_estimator
from src.video.reader import VideoReader
from src.processing.processor import PoseProcessor
from src.processing.kinematics import KinematicCalculator


CONFIG_PATH = "configs/config.yaml"


def main():
    config = load_config(CONFIG_PATH)

    # -------------------------
    # Video
    # -------------------------

    video_path = config["video"]["path"]

    reader = VideoReader(video_path)

    # -------------------------
    # Pose Estimation
    # -------------------------

    pose_estimator = create_pose_estimator(
        config["pose"]
    )

    # -------------------------
    # Pose Processing
    # -------------------------

    processing_config = config["processing"]

    pose_processor = PoseProcessor(
        confidence_threshold=processing_config[
            "confidence_threshold"
        ],
        smoothing_alpha=processing_config[
            "smoothing_alpha"
        ],
    )

    # -------------------------
    # Kinematics
    # -------------------------

    kinematic_calculator = KinematicCalculator()

    # -------------------------
    # Pipeline
    # -------------------------

    for frame_index, timestamp, frame in reader:

        # 1. Pose estimation
        pose_frame = pose_estimator.estimate(
            frame=frame,
            frame_index=frame_index,
            timestamp=timestamp,
        )

        if pose_frame is None:
            continue

        # 2. Pose processing / filtering
        processed_pose = pose_processor.process(
            pose_frame
        )

        # 3. Kinematic calculations
        kinematic_frame = kinematic_calculator.calculate(
            processed_pose
        )

        print(
            f"Frame: {processed_pose.frame_index} | "
            f"Time: {processed_pose.timestamp:.2f}s | "
            f"Left Knee: "
            f"{kinematic_frame.left_knee_angle:.2f}° | "
            f"Right Knee: "
            f"{kinematic_frame.right_knee_angle:.2f}°"
        )

    reader.release()


if __name__ == "__main__":
    main()