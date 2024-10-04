"""
Utility functions for reading and saving videos.
"""

import cv2


def read_video(video_path: str) -> list:
    """Read the video and return the frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames: list, output_video_path: str) -> None:
    """Save the video."""
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    print(f"Video saved at {output_video_path}")
