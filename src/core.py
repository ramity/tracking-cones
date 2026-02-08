import os
import glob
import pickle
from typing import Tuple, List
import numpy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def sin_cost_to_6D(cx, sx, cy, sy, cz, sz):

    cx = torch.tensor(cx)
    sx = torch.tensor(sx)
    cy = torch.tensor(cy)
    sy = torch.tensor(sy)
    cz = torch.tensor(cz)
    sz = torch.tensor(sz)

    Rx = torch.stack([
        torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx),
        torch.zeros_like(cx), cx, -sx,
        torch.zeros_like(cx), sx, cx
    ], dim=0).reshape(-1, 3, 3)

    Ry = torch.stack([
        cy, torch.zeros_like(cy), sy,
        torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy),
        -sy, torch.zeros_like(cy), cy
    ], dim=0).reshape(-1, 3, 3)

    Rz = torch.stack([
        cz, -sz, torch.zeros_like(cz),
        sz, cz, torch.zeros_like(cz),
        torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)
    ], dim=0).reshape(-1, 3, 3)

    return Rz @ Ry @ Rx

# -----------------------------
# Dataset
# -----------------------------
def parse_pose_from_filename(path: str) -> torch.Tensor:
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    pose = list(map(float, parts[-10:]))

    distance, x, y, z, sa, ca, sb, cb, sc, cc = pose
    final = [x, y, z, sa, ca, sb, cb, sc, cc]

    return torch.tensor(final, dtype=torch.float32)

class PoseDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.paths = sorted(
            glob.glob(os.path.join(image_dir, "*.png"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        pose = parse_pose_from_filename(path)

        # debug
        # image.save(f"/data/renders/debug_{idx}.png")    

        if self.transform:
            image = self.transform(image)

        # invert black background to white
        binary = numpy.array(image) / 255.0
        # print(binary)

        return binary, pose

class PoseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# -----------------------------
# Model
# -----------------------------
class PoseFCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640 * 360, 1024),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(1024, 12),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Feature extractor ----
        # Input: (B, 360, 640)

        self.features = nn.Sequential(
            # 360x640 -> 180x320
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 90x160

            # 90x160 -> 45x80
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 45x80 -> 23x40
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 23x40 -> 12x20
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 12x20 -> 6x10
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # global pooling -> (B, 512, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )

        # ---- Regression head ----
        self.head = nn.Sequential(
            nn.Flatten(),           # (B, 512)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 9)       # 3 translation + 6 rotation
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

# -----------------------------
# Train
# -----------------------------
def train(
    train_dir: str,
    val_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_path: str = "pose_model.pkl",
    load_path: str = None,
):
    torch.manual_seed(69)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((640, 360)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_ds = PoseDataset(train_dir, transform)
    val_ds = PoseDataset(val_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    if load_path:
        model, img_size = load_model(load_path)
    else:
        model = PoseNet()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total = 0
        for imgs, poses in train_loader:
            imgs, poses = imgs.to(device), poses.to(device)

            pred = model(imgs)
            loss = criterion(pred, poses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Epoch {epoch+1}/{epochs} train loss: {total/len(train_loader):.6f}, {total/69:.6f}")

        # validation
        # model.eval()
        # with torch.no_grad():
        #     total = 0
        #     for imgs, poses in val_loader:
        #         imgs, poses = imgs.to(device), poses.to(device)
        #         pred = model(imgs)
        #         loss = criterion(pred, poses)
        #         total += loss.item()

        # print(f"Epoch {epoch+1}/{epochs} val loss: {total/len(val_loader):.6f}")

    # save
    with open(save_path, "wb") as f:
        pickle.dump({
            "model_state": model.state_dict(),
            "img_size": (640, 360),
        }, f)

    print(f"Saved to {save_path}")

# -----------------------------
# Load model
# -----------------------------
def load_model(path: str) -> Tuple[nn.Module, Tuple[int, int]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(path, "rb") as f:
        data = pickle.load(f)

    model = PoseNet().to(device)
    model.load_state_dict(data["model_state"])
    model.eval()

    return model, data["img_size"]


# -----------------------------
# Evaluate on a directory
# -----------------------------
def evaluate(model_path: str, eval_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, img_size = load_model(model_path)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    ds = PoseDataset(eval_dir, transform)
    loader = DataLoader(ds, batch_size=1)

    criterion = nn.MSELoss()
    total = 0

    with torch.no_grad():
        for imgs, poses in loader:
            imgs, poses = imgs.to(device), poses.to(device)
            pred = model(imgs)

            # print("poses", poses.tolist())
            # print("preds", pred.tolist())
            print("delta", (poses - pred).tolist())

            loss = criterion(pred, poses)
            total += loss.item()

    print(f"Eval MSE: {total/len(loader):.6f}")
