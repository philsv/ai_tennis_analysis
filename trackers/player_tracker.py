import pickle
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore

sys.path.append("../")
import constants
from utils import get_center_of_bbox, measure_distance


class PlayerTracker:
    """Responsible for tracking players in the video feed."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def choose_and_filter_players(
        self, court_keypoints: dict, player_detections: list
    ) -> list:
        """Choose and filter players based on court keypoints."""
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(
            court_keypoints, player_detections_first_frame
        )
        filtered_player_detections = []
        for player_dict in player_detections:

            # Filter the players based on the chosen players from the first frame
            filtered_player_dict = {
                track_id: bbox
                for track_id, bbox in player_dict.items()
                if track_id in chosen_players
            }
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints: dict, player_dict: dict) -> list:
        """Choose players based on court keypoints."""
        distances = []
        for track_id, bbox in player_dict.items():

            # Get the center of the player bounding box
            player_center = get_center_of_bbox(bbox)
            min_distance = float("inf")

            for i in range(0, len(court_keypoints), 2):
                # Get the court point
                court_point = (court_keypoints[i], court_keypoints[i + 1])

                # Measure the distance between the player and the court point
                distance = measure_distance(player_center, court_point)

                # Update the minimum distance
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Sort the distances between the players and the court points
        distances.sort(key=lambda x: x[1])

        # Choose the first two players with the minimum distance
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(
        self,
        frames: list,
        read_from_stub: bool,
        stub_path: Union[str, None] = None,
    ) -> list:
        """Detect players in the frames."""
        player_detections = []

        if not stub_path:
            read_from_stub = False

        if read_from_stub and stub_path:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path:
            if not stub_path.endswith(".pkl"):
                raise ValueError("Stub path should end with .pkl")

            Path("tracker_stubs").mkdir(exist_ok=True)

            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame: np.ndarray) -> dict:
        """Detect players in a frame."""
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            # Only consider person class
            if object_cls_name == "person":
                player_dict[track_id] = result
        return player_dict

    def draw_bboxes(self, frames: list, player_detections: list) -> list:
        """Draw bounding boxes around players."""
        output_video_frames = []
        for frame, player_dict in zip(frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                player_name = constants.PLAYER_NAMES[track_id]
                # Draw player ID
                cv2.putText(
                    frame,
                    f"Player {track_id}: {player_name}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    constants.RED,
                    2,
                )

                # Draw bounding box around the players
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), constants.RED, 2
                )
            output_video_frames.append(frame)
        return output_video_frames
