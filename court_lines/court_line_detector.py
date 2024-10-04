import sys

import cv2
import numpy as np
import torch
import torchvision.models as models  # type: ignore
import torchvision.transforms as transforms  # type: ignore

sys.path.append("../")
import constants


class CourtLineDetector:
    """Responsible for detecting the court lines in the video feed."""

    def __init__(self, model_path: str):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Predict the court line keypoints in a frame."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        height, width = img_rgb.shape[:2]

        keypoints[::2] *= width / 224.0
        keypoints[1::2] *= height / 224.0
        return keypoints

    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
    ) -> np.ndarray:
        """Draw the court line keypoints on the frame."""
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(
                frame,
                str(i // 2),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                constants.BLUE,
                2,
            )
            cv2.circle(frame, (x, y), 5, constants.BLUE, cv2.FILLED)
        return frame

    def draw_keypoints_on_video(
        self,
        video_frames: list[np.ndarray],
        keypoints: np.ndarray,
    ) -> list[np.ndarray]:
        """Draw the court line keypoints on the video frames."""
        output_video_frames = []
        for frame in video_frames:

            if frame is None or not isinstance(frame, np.ndarray):
                print("Warning: Skipping an invalid frame.")
                continue

            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
