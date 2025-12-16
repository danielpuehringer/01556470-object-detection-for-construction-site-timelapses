import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import models, transforms


# -------------------------
# 1) Dataset
# -------------------------
class InterestDataset(Dataset):
    def __init__(self, images_dir: str, csv_path: str, transform=None):
        self.images_dir = images_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Basic sanity checks
        if "filename" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV has wrong format; please check: filename,label")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filename = row["filename"]
        label = int(row["label"])

        img_path = os.path.join(self.images_dir, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Always convert to RGB (important for ResNet)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# -------------------------
# 2) Main training script
# -------------------------
def main():
    # Paths
    images_dir = "../datasets/test-dataset/test-dataset-input"
    csv_path = "../datasets/test-dataset/labels.csv"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Transforms (ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Dataset
    dataset = InterestDataset(images_dir=images_dir, csv_path=csv_path, transform=transform)

    # Train/val split (80/20)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    # Load pretrained ResNet-50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Replace head for 2 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Freeze backbone (everything except fc)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")

    model = model.to(device)

    # Loss + optimizer (only trainable params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # Training
    num_epochs = 10

    for epoch in range(num_epochs):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1:02d}/{num_epochs} "
              f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.3f}")

    # Save trained model weights
    os.makedirs("models", exist_ok=True)
    save_path = "models/resnet50_interest_head_onlyNew.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
