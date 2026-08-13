import sys
from pathlib import Path

# Projenin kök dizinini (src'nin bir üst klasörü) sys.path'e ekler
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from src.config.loader import load_config
from src.pose.factory import create_pose_estimator
from src.video.reader import VideoReader


CONFIG_PATH = "configs/config.yaml"


def main():
    config = load_config(CONFIG_PATH)

    video_path = config["video"]["path"]

    reader = VideoReader(video_path)

    pose_estimator = create_pose_estimator(
        config["pose"]
    )

    for frame_index, timestamp, frame in reader:
        pose_frame = pose_estimator.estimate(
            frame=frame,
            frame_index=frame_index,
            timestamp=timestamp,
        )

        if pose_frame is None:
            continue

        print(
            f"Frame: {pose_frame.frame_index} | "
            f"Time: {pose_frame.timestamp:.2f}s | "
            f"Keypoints: {pose_frame.keypoints.shape} | "
            f"Confidence: {pose_frame.person_confidence:.3f}"
        )

    reader.release()


if __name__ == "__main__":
    main()