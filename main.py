import sys
from pathlib import Path

import cv2

# Projenin kök dizinini (src'nin bir üst klasörü) sys.path'e ekler
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.config.loader import load_config
from src.pose.factory import create_pose_estimator
from src.video.reader import VideoReader
from src.processing.processor import PoseProcessor
from src.processing.kinematics import KinematicCalculator
from src.exercises.squat import SquatAnalyzer


CONFIG_PATH = "configs/config.yaml"


# COCO 17 keypoint bağlantıları
SKELETON = [
    (0, 1),   # nose - left eye
    (0, 2),   # nose - right eye
    (1, 3),   # left eye - left ear
    (2, 4),   # right eye - right ear

    (5, 6),   # shoulders

    (5, 7),   # left shoulder - left elbow
    (7, 9),   # left elbow - left wrist

    (6, 8),   # right shoulder - right elbow
    (8, 10),  # right elbow - right wrist

    (5, 11),  # left shoulder - left hip
    (6, 12),  # right shoulder - right hip

    (11, 12), # hips

    (11, 13), # left hip - left knee
    (13, 15), # left knee - left ankle

    (12, 14), # right hip - right knee
    (14, 16), # right knee - right ankle
]


def draw_pose(
    frame,
    keypoints,
    confidence_threshold=0.5,
):
    

    for x, y, confidence in keypoints:

        if confidence < confidence_threshold:
            continue

        x = int(x)
        y = int(y)

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1,
        )

    # Skeleton
  

    for start_idx, end_idx in SKELETON:

        start = keypoints[start_idx]
        end = keypoints[end_idx]

        if (
            start[2] < confidence_threshold
            or end[2] < confidence_threshold
        ):
            continue

        start_point = (
            int(start[0]),
            int(start[1]),
        )

        end_point = (
            int(end[0]),
            int(end[1]),
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            (255, 255, 255),
            2,
        )

    return frame


def draw_information(
    frame,
    kinematic_frame,
    squat_analyzer,
):
    

    left_knee = kinematic_frame.left_knee_angle
    right_knee = kinematic_frame.right_knee_angle

    cv2.putText(
        frame,
        f"Left Knee: {left_knee:.1f} deg",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Right Knee: {right_knee:.1f} deg",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"State: {squat_analyzer.state.value}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Repetitions: {squat_analyzer.repetitions}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return frame


def main():
    config = load_config(CONFIG_PATH)

    video_path = config["video"]["path"]

    reader = VideoReader(video_path)

    # -------------------------
    # Pose Estimation
    # -------------------------

    pose_estimator = create_pose_estimator(
        config["pose"]
    )

    # Pose Processing
  
    processing_config = config["processing"]

    pose_processor = PoseProcessor(
        confidence_threshold=processing_config[
            "confidence_threshold"
        ],
        smoothing_alpha=processing_config[
            "smoothing_alpha"
        ],
    )

    # Kinematics
  

    kinematic_calculator = KinematicCalculator()

   
    # Squat Analyzer

    squat_analyzer = SquatAnalyzer()


    # Frame processing

    for frame_index, timestamp, frame in reader:

        # 1. Pose estimation
        pose_frame = pose_estimator.estimate(
            frame=frame,
            frame_index=frame_index,
            timestamp=timestamp,
        )

        if pose_frame is None:
            continue

        # 2. Pose processing
        processed_pose = pose_processor.process(
            pose_frame
        )

        # 3. Kinematics
        kinematic_frame = kinematic_calculator.calculate(
            processed_pose
        )

        # 4. Squat analysis
        squat_analyzer.update(
            kinematic_frame
        )

        # -------------------------
        # Visualization
        # -------------------------

        frame = draw_pose(
            frame,
            processed_pose.keypoints,
            confidence_threshold=processing_config[
                "confidence_threshold"
            ],
        )

        frame = draw_information(
            frame,
            kinematic_frame,
            squat_analyzer,
        )

        cv2.imshow(
            "SportMotion",
            frame,
        )

        # q → quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # -------------------------
    # Cleanup
    # -------------------------

    reader.release()
    cv2.destroyAllWindows()

    # -------------------------
    # Final result
    # -------------------------

    result = squat_analyzer.finalize()

    print("\n=== SportMotion ===")
    print(f"Exercise: {result.exercise}")
    print(f"Repetitions: {result.repetitions}")
    print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()