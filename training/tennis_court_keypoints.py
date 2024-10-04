"""
Train a YOLOv5 model on the tennis court keypoints dataset.
"""

import json
import os
import warnings
import zipfile

import cv2
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms  # type: ignore
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore

warnings.simplefilter("ignore", UserWarning)

load_dotenv()

# Load the YOLOv5 model.
model = YOLO("yolov5l6u.pt")

# Set the device to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_dataset() -> str:
    """Download the tennis court keypoints dataset and unzip it."""
    file_name = "tennis_court_det_dataset.zip"

    if os.path.exists("data/data_train.json") and os.path.exists("data/data_val.json"):
        return file_name

    url = "https://drive.usercontent.google.com/download?id=1lhAaeQCmk2y440PmagA0KmIVBIysVMwu&export=download&authuser=0&confirm=t&uuid=3077628e-fc9b-4ef2-8cde-b291040afb30&at=APZUnTU9lSikCSe3NqbxV5MVad5T%3A1708243355040"

    headers = {
        "Host": "drive.usercontent.google.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        "Cookie": "HSID=Ag2OIHvsd2Wub4C7z; SSID=AWnBcQKwDHiTrZAU1; APISID=pltrFZgE9lJ0o1gq/AN9feEHYvs8oHd519; SAPISID=zgF45F21ZPWzYWZw/AgUMJ8b7QQXuWGn19; __Secure-1PAPISID=zgF45F21ZPWzYWZw/AgUMJ8b7QQXuWGn19; __Secure-3PAPISID=zgF45F21ZPWzYWZw/AgUMJ8b7QQXuWGn19; SID=g.a000fwgYx1PcnW-rFyFhg3x6mQHzCrwXz-KFhoOLogUl7YTWI-uttBbVDRolhF-hY16nwHXw0gACgYKAWISAQASFQHGX2MivNTw_E_toJuIRy6LMpKNOBoVAUF8yKpFSmvq7AMjvEWeNc50Zff40076; __Secure-1PSID=g.a000fwgYx1PcnW-rFyFhg3x6mQHzCrwXz-KFhoOLogUl7YTWI-utbSY2jBY1VXuw8gYl5hIO2QACgYKAXsSAQASFQHGX2MihVCJ1PwLozGqZgdSatM9QhoVAUF8yKpgrsTvI8i_UE-YHpoN7Gx-0076; __Secure-3PSID=g.a000fwgYx1PcnW-rFyFhg3x6mQHzCrwXz-KFhoOLogUl7YTWI-utwVfPl2imdPimZJ9tdDZGQAACgYKAUESAQASFQHGX2MiEJ49mV4jME2kttDAV5hwWBoVAUF8yKp80mIgju1lu-q4nI7VsFDM0076; NID=511=efI9IZpxtyJ7Dw1MAUXU8FlzS5jXGewY4Er8HliWc3A0RSWdgvNDyKY66ETjgRyTGWPbWODSmiSeYSBab5SPHVwqbJxd6ZeGW2f6BkHi61UKksXPH0CVJRM1hKpMjHPU5qw7tboM2Mi87NrosV8COB-GCLulLLbjOoSAEQewTe8NVZ5Owq8IkwvxFGfJkmUKEMkFWrw9yb5nTDl3wbZEsGFI92iEdNTSxSRovNCIPN2US-SCFdQ0m2BtvwdiWZbgnn7dSQ8yPA145Kk2BA-ATpJNJ6SJHEHLQY-9CPail9D5qgJgxR925EUg5RGCpEu9wS5xbA62KTa19wAvbAq7Dk3TWc-iX4p1s7ESFyDC7yMpFxiFPJjqkWwFi_ZfiK2TW2t0TQ60DFBxqOytQaLyHrkEvD-CQPVj6OCOP22cZY0Cu61HaAQgFO9pXH-kJUlywzVdbirJumN5gswyaQ49b3KdLcG0Jb7brOMTM24T2nGtQ10hJzsnTwX7dBk3ujqQrI_DGuURvPassPUrIZ0; AEC=Ae3NU9MOEGeKAZjP6INpOYbyMraWAWztmx5pJB_1ILu1furiTy1K37k15u0; __Secure-1PSIDTS=sidts-CjEBYfD7Z9twEKTWJ9gU7KG-rLbxJGNRQIoG3wH6JVu6yiCC2fsRrm7tN8L6d5WlILrnEAA; __Secure-3PSIDTS=sidts-CjEBYfD7Z9twEKTWJ9gU7KG-rLbxJGNRQIoG3wH6JVu6yiCC2fsRrm7tN8L6d5WlILrnEAA; 1P_JAR=2024-02-18-08; SIDCC=ABTWhQExCxkfmwCkG1RaEgz8U1ZkPeh3HmLMUdMt8S5cNSsLY5U5rAL6wlvq7dtjRw7zrtAbqsFI; __Secure-1PSIDCC=ABTWhQH0jLeRIS6Tu3LS8DXB5Q3gGDq9LTmlk60FKu795Bf0UbzsOcYWVAE96clq5aAL8i724Q0; __Secure-3PSIDCC=ABTWhQHIFcyv3nZYwp78WXEQal71jCE_ZsGT5lXs8VLr7XDIfFqHcLTIPz4HxzJb9ZnYQ5l2s9eU",
        "Connection": "keep-alive",
    }

    response = requests.get(url, headers=headers, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(file_name, "wb") as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong during the download")
    return file_name


def unzip_dataset(file_name: str) -> None:
    """Unzip the dataset."""
    if os.path.exists("data"):
        return

    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall()
    os.remove(file_name)


class KeypointsDataset(Dataset):
    """Tennis court keypoints dataset."""

    def __init__(self, img_dir, data_file):
        self.img_dir = img_dir
        with open(data_file, "r") as f:
            self.data = json.load(f)

        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.imread(f"{self.img_dir}/{item['id']}.png")
        h, w = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        kps = np.array(item["kps"]).flatten()
        kps = kps.astype(np.float32)

        kps[::2] *= 224.0 / w  # Adjust x coordinates
        kps[1::2] *= 224.0 / h  # Adjust y coordinates

        return img, kps


def save_checkpoint(epoch, model, optimizer, best_loss, filename="models/checkpoint.pth.tar"):
    """Save the checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch+1}")


def load_checkpoint(model, optimizer, filename="models/checkpoint.pth.tar"):
    """Load the checkpoint to resume training."""
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        print(
            f"Checkpoint loaded. Resuming from epoch {epoch+1} with best loss: {best_loss}"
        )
        return epoch, best_loss
    else:
        print("No checkpoint found. Starting training from scratch.")
        return -1, float("inf")  # If no checkpoint exists, start from scratch


def train_model() -> None:
    """Train the mode and save it."""
    train_dataset = KeypointsDataset("data/images", "data/data_train.json")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Load the ResNet50 model.
    model = models.resnet50(pretrained=True)

    # Replace the last layer with a new layer.
    # 14 Keypoints * 2 (x, y coordinates) = 28 total outputs.
    model.fc = torch.nn.Linear(model.fc.in_features, 14 * 2)

    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 1e-4 = 0.0001

    num_epochs = 20
    
    # Load checkpoint if it exists
    start_epoch, best_loss = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch + 1, num_epochs):
        running_loss = 0.0

        model.train()  # Ensure the model is in training mode
        for i, (imgs, kps) in enumerate(train_loader):
            imgs = imgs.to(device)
            kps = kps.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, kps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log the loss every 10 iterations.
            if i % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Iteration {i}, Loss: {loss.item()}"
                )

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")
        
        # Save the checkpoint every epoch
        save_checkpoint(epoch, model, optimizer, best_loss)

        # Save the model if the average loss is the best we've seen so far.
        if avg_loss < best_loss:
            torch.save(model.state_dict(), "models/tennis_court_keypoints_best.pth")
            print(f"Best model saved with loss: {best_loss}")

    # Save the last model (regardless of whether it's the best).
    torch.save(model.state_dict(), "models/tennis_court_keypoints_last.pth")
    print("Last model saved.")


if __name__ == "__main__":
    # Download tennis court keypoints dataset.
    file_name = download_dataset()

    # Unzip the dataset.
    unzip_dataset(file_name)

    # Train the model.
    train_model()
