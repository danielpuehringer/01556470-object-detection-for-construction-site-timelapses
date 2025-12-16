import os
import json
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
def parse_binary_label(value) -> int:
    # Expect 0/1 in CSV, but be robust.
    if pd.isna(value):
        raise ValueError("Label value is NaN")
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "interesting", "true", "yes"}:
            return 1
        if s in {"0", "not interesting", "false", "no"}:
            return 0
        try:
            return 1 if int(float(s)) == 1 else 0
        except Exception as e:
            raise ValueError(f"Unrecognized label: {value!r}") from e
    return 1 if int(value) == 1 else 0


def find_image_path(images_dir: str, image_name: str) -> str:
    image_name = str(image_name).strip()
    base, ext = os.path.splitext(image_name)

    # If CSV already contains extension, try directly.
    candidates = [image_name] if ext else [image_name + e for e in
                                          [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]]

    for cand in candidates:
        p = os.path.join(images_dir, cand)
        if os.path.exists(p):
            return p

    # Fallback scan
    for fn in os.listdir(images_dir):
        if fn.startswith(image_name + "."):
            p = os.path.join(images_dir, fn)
            if os.path.isfile(p):
                return p

    raise FileNotFoundError(
        f"Could not find image for '{image_name}' in '{images_dir}'. Tried: {candidates} + directory scan."
    )


def safe_load_boxes(boxes_str: str):
    """
    bounding_boxes cell is a JSON list. Sometimes it may contain doubled quotes.
    Return list[dict].
    """
    if pd.isna(boxes_str) or str(boxes_str).strip() == "":
        return []

    s = str(boxes_str).strip()

    # Common CSV escaping artifact: "" -> "
    # If json.loads fails, try this normalization.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return json.loads(s.replace('""', '"'))
        except json.JSONDecodeError:
            # Give up gracefully: treat as empty
            return []


def label_to_onehot(lbl: str):
    """
    Encode box label into 3 dims:
      [person, vehicle, other]
    """
    s = (lbl or "").strip().lower()
    if s == "person":
        return (1.0, 0.0, 0.0)
    if s in {"car", "truck", "bus", "motorcycle", "vehicle"}:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def encode_boxes_fixed(boxes, img_w: float, img_h: float, max_boxes: int):
    """
    Fixed-length vector for bounding boxes:
    For top-K boxes (by score desc), each contributes:
      [x1/W, y1/H, x2/W, y2/H, score, onehot_person, onehot_vehicle, onehot_other]  => 8 dims
    Total dims = max_boxes * 8
    """
    if img_w <= 0 or img_h <= 0:
        img_w, img_h = 1.0, 1.0

    # Sort by score descending
    def score_of(b):
        try:
            return float(b.get("score", 0.0))
        except Exception:
            return 0.0

    boxes_sorted = sorted(boxes, key=score_of, reverse=True)[:max_boxes]
    feats = []

    for b in boxes_sorted:
        bbox = b.get("bbox", [0, 0, 0, 0])
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            bbox = [0, 0, 0, 0]

        x1, y1, x2, y2 = [float(v) if v is not None else 0.0 for v in bbox]
        x1 /= img_w
        x2 /= img_w
        y1 /= img_h
        y2 /= img_h

        sc = score_of(b)
        oh = label_to_onehot(b.get("label", ""))

        feats.extend([x1, y1, x2, y2, sc, *oh])

    # Pad
    need = max_boxes * 8 - len(feats)
    if need > 0:
        feats.extend([0.0] * need)

    return feats


# -------------------------
# Dataset (NEW CSV ONLY)
# -------------------------
class InterestMultiModalDataset(Dataset):
    REQUIRED_COLS = [
        "image_name",
        "interesting/not interesting",
        "num_person",
        "num_vehicles",
        "avg_person_conf",
        "total_boxes",
        "bounding_boxes",
    ]

    def __init__(self, images_dir: str, csv_path: str, transform=None, max_boxes: int = 10):
        self.images_dir = images_dir
        self.df = pd.read_csv(csv_path)
        self.df.columns = [str(c).strip() for c in self.df.columns]
        self.transform = transform
        self.max_boxes = max_boxes

        missing = [c for c in self.REQUIRED_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Found: {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_name = row["image_name"]
        y = parse_binary_label(row["interesting/not interesting"])

        img_path = find_image_path(self.images_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Numeric features (use ALL numeric columns explicitly)
        num_person = float(row["num_person"])
        num_vehicles = float(row["num_vehicles"])
        avg_person_conf = float(row["avg_person_conf"])
        total_boxes = float(row["total_boxes"])

        boxes = safe_load_boxes(row["bounding_boxes"])
        box_feats = encode_boxes_fixed(boxes, img_w=float(w), img_h=float(h), max_boxes=self.max_boxes)

        # Final metadata vector
        meta = [num_person, num_vehicles, avg_person_conf, total_boxes] + box_feats
        meta = torch.tensor(meta, dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, meta, torch.tensor(y, dtype=torch.long)


# -------------------------
# Model: ResNet50 + Metadata MLP + Fusion Head
# -------------------------
class ResNet50WithMetadata(nn.Module):
    def __init__(self, meta_dim: int, freeze_backbone: bool = True):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()  # outputs 2048-dim features
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(2048 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x_img, x_meta):
        img_feat = self.backbone(x_img)          # (B, 2048)
        meta_feat = self.meta_net(x_meta)        # (B, 64)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.head(fused)


# -------------------------
# Train / Eval
# -------------------------
def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, meta, labels in loader:
        imgs = imgs.to(device)
        meta = meta.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(imgs, meta)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += imgs.size(0)

    return total_loss / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--csv-path", required=True, help="labels.csv with the NEW schema")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-boxes", type=int, default=10, help="Top-K boxes to encode from bounding_boxes")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ResNet backbone (default: False)")
    parser.add_argument("--lr", type=float, default=1e-3)
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

    dataset = InterestMultiModalDataset(
        images_dir=args.images_dir,
        csv_path=args.csv_path,
        transform=transform,
        max_boxes=args.max_boxes,
    )

    meta_dim = 4 + args.max_boxes * 8  # 4 numeric + (K boxes * 8 dims per box)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ResNet50WithMetadata(meta_dim=meta_dim, freeze_backbone=args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer=optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device, criterion, optimizer=None)

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"Val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

    os.makedirs("models", exist_ok=True)
    save_path = "models/resnet50_fused_image_meta.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta_dim": meta_dim,
            "max_boxes": args.max_boxes,
        },
        save_path,
    )
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
