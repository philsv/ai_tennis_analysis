"""
Utility functions for reading and saving videos.
"""

import cv2
from PIL import Image


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


def save_video_as_gif(
    output_video_frames: list,
    output_video_path: str,
    fps: int = 24,
    duration: int = 10,
) -> None:
    """Save the video frames as a GIF."""
    from tqdm import tqdm
    
    max_frames = fps * duration
    limited_frames = output_video_frames[:max_frames]

    images = []
    for frame in tqdm(limited_frames, desc="Saving GIF"):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        images.append(image)

    images[0].save(
        output_video_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 / fps,
        loop=0,
    )
    print(f"GIF saved at {output_video_path}")
