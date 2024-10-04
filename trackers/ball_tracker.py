import pickle
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO  # type: ignore

sys.path.append("../")
import constants


class BallTracker:
    """Responsible for tracking the tennis ball in the video feed."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def get_ball_shot_frames(self, ball_positions: list) -> list:
        """Get the frames where the ball was shot."""
        ball_positions = [x.get(1, []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        df["ball_hit"] = 0

        df["mid_y"] = (df["y1"] + df["y2"]) / 2
        df["mid_y_rolling_mean"] = (
            df["mid_y"].rolling(window=5, min_periods=1, center=False).mean()
        )
        df["delta_y"] = df["mid_y_rolling_mean"].diff()

        minimum_change_frames_for_hit = 25
        for i in range(1, len(df) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = (
                df["delta_y"].iloc[i] > 0 and df["delta_y"].iloc[i + 1] < 0
            )
            positive_position_change = (
                df["delta_y"].iloc[i] < 0 and df["delta_y"].iloc[i + 1] > 0
            )

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(
                    i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1
                ):
                    negative_position_change_following_frame = (
                        df["delta_y"].iloc[i] > 0 and df["delta_y"].iloc[change_frame] < 0
                    )
                    positive_position_change_following_frame = (
                        df["delta_y"].iloc[i] < 0 and df["delta_y"].iloc[change_frame] > 0
                    )

                    if (
                        negative_position_change
                        and negative_position_change_following_frame
                    ):
                        change_count += 1
                    elif (
                        positive_position_change
                        and positive_position_change_following_frame
                    ):
                        change_count += 1

                    if change_count >= minimum_change_frames_for_hit:
                        df["ball_hit"].iloc[i] = 1
                        break

        ball_shot_frames = df[df["ball_hit"] == 1].index.tolist()
        return ball_shot_frames

    def interpolate_ball_detections(self, ball_detections: list) -> list:
        """Interpolate the ball detections."""
        ball_positions = [x.get(1, []) for x in ball_detections]
        df = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate()
        df = df.bfill()
        interpolated_ball_positions = [
            {1: x} for x in df.to_numpy().tolist()
        ]  # 1=track_id, x=bbox
        return interpolated_ball_positions

    def detect_frames(
        self,
        frames: list,
        read_from_stub: bool,
        stub_path: Union[str, None] = None,
    ) -> list:
        """Detect the tennis ball in the frames."""
        ball_detections = []

        if not stub_path:
            read_from_stub = False

        if read_from_stub and stub_path:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path:
            if not stub_path.endswith(".pkl"):
                raise ValueError("Stub path should end with .pkl")

            Path("tracker_stubs").mkdir(exist_ok=True)

            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame: np.ndarray) -> dict:
        """Detect the tennis ball in a frame."""
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_bboxes(self, frames: list, ball_detections: list) -> list:
        """Draw bounding boxes around the tennis ball."""
        output_video_frames = []
        for frame, ball_dict in zip(frames, ball_detections):
            for _, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox

                # Draw the tennis ball label
                cv2.putText(
                    frame,
                    f"tennis ball",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    constants.YELLOW,
                    2,
                )

                # Draw bounding box around the ball
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), constants.YELLOW, 2
                )
            output_video_frames.append(frame)
        return output_video_frames
