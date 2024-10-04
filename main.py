import warnings
from copy import deepcopy
from pathlib import Path

import cv2
import pandas as pd
from pandas.errors import SettingWithCopyWarning

import constants
from court_lines import CourtLineDetector
from mini_court import MiniCourt
from trackers import BallTracker, PlayerTracker
from utils import (convert_pixel_distance_to_meters, draw_player_stats,
                   measure_distance, read_video, save_video, save_video_as_gif)

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)
warnings.simplefilter("ignore", FutureWarning)


def main():
    """Run the player, ball tracking and court line detection."""
    video_frames = read_video("inputs/video.mp4")

    # Initialize the trackers.
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")

    # Detect the court lines.
    court_line_detector = CourtLineDetector(
        model_path="models/tennis_court_keypoints_last.pth"
    )
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Detect the players, ball, and court lines.
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/player_detections.pkl",
    )
    # Filter the players based on the court lines.
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )

    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/ball_detections.pkl",
    )
    ball_detections = ball_tracker.interpolate_ball_detections(ball_detections)

    # Initialize the mini court.
    mini_court = MiniCourt(video_frames[0])

    # Detect the tennis ball hits.
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court coordinates.
    player_mini_court_detections, ball_mini_court_detections = (
        mini_court.convert_bboxes_to_mini_court_coordinates(
            player_detections, ball_detections, court_keypoints
        )
    )
    # Get the distance covered by the ball in meters.

    player_stats = [
        {
            "frame_number": 0,
            "player_1_number_of_shots": 0,
            "player_1_total_shot_speed": 0,
            "player_1_last_shot_speed": 0,
            "player_1_total_player_speed": 0,
            "player_1_last_player_speed": 0,
            "player_2_number_of_shots": 0,
            "player_2_total_shot_speed": 0,
            "player_2_last_shot_speed": 0,
            "player_2_total_player_speed": 0,
            "player_2_last_player_speed": 0,
        }
    ]

    for ball_shot_index in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_index]
        end_frame = ball_shot_frames[ball_shot_index + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24FPS

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1],
        )
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )

        # Speed of the ball shot in kilometers per hour.
        speed_of_ball_shot = (
            distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
        )

        # Player who hit the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id], ball_mini_court_detections[start_frame][1]
            ),
        )

        # Speed of opponent player
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_player_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id],
        )
        distance_covered_by_opponent_player_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_player_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )

        # Speed of the opponent player in kilometers per hour.
        speed_of_opponent_player = (
            distance_covered_by_opponent_player_meters / ball_shot_time_in_seconds * 3.6
        )

        # Update player stats.
        current_player_stats = deepcopy(player_stats[-1])
        current_player_stats["frame_number"] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1

        current_player_stats[
            f"player_{player_shot_ball}_total_shot_speed"
        ] += speed_of_ball_shot
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = (
            speed_of_ball_shot
        )

        current_player_stats[
            f"player_{opponent_player_id}_total_player_speed"
        ] += speed_of_opponent_player
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = (
            speed_of_opponent_player
        )

        player_stats.append(current_player_stats)

    player_stats_df = pd.DataFrame(player_stats)
    frames_df = pd.DataFrame({"frame_number": list(range(len(video_frames)))})
    player_stats_df = pd.merge(
        frames_df, player_stats_df, on="frame_number", how="left"
    )
    player_stats_df = player_stats_df.ffill()

    player_stats_df["player_1_average_shot_speed"] = (
        player_stats_df["player_1_total_shot_speed"]
        / player_stats_df["player_1_number_of_shots"]
    )
    player_stats_df["player_2_average_shot_speed"] = (
        player_stats_df["player_2_total_shot_speed"]
        / player_stats_df["player_2_number_of_shots"]
    )
    player_stats_df["player_1_average_player_speed"] = (
        player_stats_df["player_1_total_player_speed"]
        / player_stats_df["player_2_number_of_shots"]
    )
    player_stats_df["player_2_average_player_speed"] = (
        player_stats_df["player_2_total_player_speed"]
        / player_stats_df["player_1_number_of_shots"]
    )

    # Draw the bounding boxes and keypoints.
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(
        output_video_frames, court_keypoints
    )

    # Draw the mini court.
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames,
        player_mini_court_detections,
    )
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames,
        ball_mini_court_detections,
        color=constants.YELLOW,
    )

    # Draw player stats.
    output_video_frames = draw_player_stats(output_video_frames, player_stats_df)

    # Draw frame number on top left corner.
    for i, frame in enumerate(output_video_frames):
        cv2.putText(
            frame,
            f"Frame: {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            constants.GREEN,
            2,
        )

    # Save the output video.
    Path("outputs").mkdir(exist_ok=True)
    save_video(output_video_frames, "outputs/output_video.avi")


if __name__ == "__main__":
    main()
