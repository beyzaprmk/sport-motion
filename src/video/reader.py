from pathlib import Path
import cv2

class VideoReader:
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        self.capture = cv2.VideoCapture(str(self.video_path))

        if not self.capture.isOpened():
            raise ValueError(
                f"Could not open video: {self.video_path}"
            )

        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(
            self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.width = int(
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        self.height = int(
            self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    def __iter__(self):
        frame_index = 0

        while True:
            success, frame = self.capture.read()

            if not success:
                break

            timestamp = (
                frame_index / self.fps
                if self.fps > 0
                else 0.0
            )

            yield frame_index, timestamp, frame

            frame_index += 1

    def release(self):
        self.capture.release()