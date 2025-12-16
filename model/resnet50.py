import os
import argparse
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import models, transforms


# -------------------------
# Helpers
# -------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_binary_label(value) -> int:
    """
    Accepts: 0/1, "0"/"1", "interesting"/"not interesting", "yes"/"no", etc.
    Returns: 0 or 1
    """
    if pd.isna(value):
        raise ValueError("Label value is NaN")

    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "interesting", "true", "yes", "y"}:
            return 1
        if s in {"0", "not interesting", "not_interesting", "false", "no", "n"}:
            return 0
        # Try to parse numeric strings like "1.0"
        try:
            v = int(float(s))
            return 1 if v == 1 else 0
        except Exception as e:
            raise ValueError(f"Unrecognized label string: {value!r}") from e

    # numeric (int/float/bool)
    v = int(value)
    return 1 if v == 1 else 0


def _find_image_path(images_dir: str, image_name: str) -> str:
    """
    Finds an image file in images_dir.
    - If image_name already has an extension, tries it directly.
    - Otherwise tries common extensions, then falls back to scanning the folder.
    """
    image_name = str(image_name).strip()
    base, ext = os.path.splitext(image_name)

    candidates = []
    if ext:  # already has extension
        candidates.append(image_name)
    else:
        for e in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            candidates.append(image_name + e)

    for cand in candidates:
        p = os.path.join(images_dir, cand)
        if os.path.exists(p):
            return p

    # Fallback: scan directory for any file that starts with "<image_name>."
    try:
        for fn in os.listdir(images_dir):
            if fn.startswith(image_name + "."):
                p = os.path.join(images_dir, fn)
                if os.path.isfile(p):
                    return p
    except FileNotFoundError:
        pass

    raise FileNotFoundError(
        f"Could not find image for '{image_name}' in '{images_dir}'. "
        f"Tried: {candidates} (and a directory scan fallback)."
    )


# -------------------------
# 1) Dataset
# -------------------------
class InterestDataset(Dataset):
    def __init__(self, images_dir: str, csv_path: str, transform=None):
        self.images_dir = images_dir
        self.df = _normalize_columns(pd.read_csv(csv_path))
        self.transform = transform

        # Support both CSV schemas:
        #  - filename,label
        #  - image_name,interesting/not interesting,...
        if "filename" in self.df.columns:
            self.image_col = "filename"
        elif "image_name" in self.df.columns:
            self.image_col = "image_name"
        else:
            raise ValueError(
                "CSV missing image column. Expected 'filename' or 'image_name'. "
                f"Found columns: {list(self.df.columns)}"
            )

        if "label" in self.df.columns:
            self.label_col = "label"
        elif "interesting/not interesting" in self.df.columns:
            self.label_col = "interesting/not interesting"
        else:
            raise ValueError(
                "CSV missing label column. Expected 'label' or 'interesting/not interesting'. "
                f"Found columns: {list(self.df.columns)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_name = row[self.image_col]
        label = _parse_binary_label(row[self.label_col])

        img_path = _find_image_path(self.images_dir, image_name)

        # Always convert to RGB (important for ResNet)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# -------------------------
# 2) Main training script
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="../datasets/test-dataset/test-dataset-input")
    parser.add_argument("--csv-path", default="../datasets/test-dataset/labels.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = InterestDataset(images_dir=args.images_dir, csv_path=args.csv_path, transform=transform)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    for epoch in range(args.epochs):
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

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

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

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(f"Epoch {epoch+1:02d}/{args.epochs} "
              f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.3f}")

    os.makedirs("models", exist_ok=True)
    save_path = "models/resnet50_interest_head_onlyNew.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
