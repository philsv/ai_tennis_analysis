"""
Train a YOLOv5 model on the tennis ball dataset.
"""

import os

from dotenv import load_dotenv
from roboflow import Roboflow  # type: ignore
from ultralytics import YOLO  # type: ignore

load_dotenv()

model = YOLO("yolov5l6u.pt")


def download_dataset() -> None:
    """Download the tennis ball dataset from Roboflow."""
    roboflow_api_key = str(os.getenv("ROBOFLOW_API_KEY"))

    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace("viren-dhanwani").project("tennis-ball-detection")
    dataset = project.version(6)
    dataset.download("yolov5")


def update_yml_file() -> None:
    """Update the yml file text to change the train and valid paths."""
    yml_path = "tennis-ball-detection-6/data.yaml"
    
    if not os.path.exists(yml_path):
        raise Exception("data.yaml not found in path.")

    with open(yml_path, "r") as f:
        data = f.read()
        
    data = data.replace(
        "train: tennis-ball-detection-6/train/images", "train: ../train/images"
    )
    data = data.replace(
        "val: tennis-ball-detection-6/valid/images", "val: ../valid/images"
    )
    
    with open(yml_path, "w") as f:
        f.write(data)
        
    print("Updated yml file.")


def train_model(data: str) -> None:
    """Train the model."""
    model.train(data=data, epochs=100, imgsz=640)


if __name__ == "__main__":
    # Download tennis ball dataset to train the model on.
    download_dataset()

    # Train the model.
    data_yml_path = str(os.getenv("DATA_YML_PATH"))
    train_model(data=data_yml_path)
