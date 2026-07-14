import os
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =========================
# CONFIG
# =========================

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "dataset" / "images"
CSV_PATH = ROOT / "dataset" / "labels.csv"

SAVE_DIR = ROOT / "runs" / "crnn"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMG_H = 48
IMG_W = 192

BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use only characters needed for UAE plate code + number
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_IDX = 0

char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}
idx_to_char[BLANK_IDX] = ""


# =========================
# DATASET
# =========================

class PlateDataset(Dataset):
    def __init__(self, csv_path, image_dir, split):
        self.image_dir = Path(image_dir)

        df = pd.read_csv(csv_path, dtype=str).fillna("")
        df["ocr_label"] = df["ocr_label"].str.strip().str.upper()

        if "split" in df.columns:
            df = df[df["split"].str.lower() == split.lower()]
        else:
            raise ValueError("CSV must contain a split column: train / val / test")

        df = df[df["ocr_label"] != ""].reset_index(drop=True)

        self.df = df

        self.transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.df)

    def encode_label(self, text):
        encoded = []
        for ch in text:
            if ch not in char_to_idx:
                raise ValueError(f"Invalid character '{ch}' in label '{text}'")
            encoded.append(char_to_idx[ch])
        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        label = row["ocr_label"]

        image_path = self.image_dir / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        encoded_label = self.encode_label(label)

        return image, encoded_label, label


def collate_fn(batch):
    images, labels, texts = zip(*batch)

    images = torch.stack(images, dim=0)

    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labels = torch.cat(labels)

    return images, labels, label_lengths, texts


# =========================
# MODEL
# =========================

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),      # 48x192 -> 24x96

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),      # 24x96 -> 12x48

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d((2, 1), (2, 1)),  # 12x48 -> 6x48

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.MaxPool2d((2, 1), (2, 1)),  # 6x48 -> 3x48
        )

        self.rnn = nn.LSTM(
            input_size=512 * 3,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        # B, C, H, W

        b, c, h, w = x.size()

        x = x.permute(3, 0, 1, 2)
        # W, B, C, H

        x = x.reshape(w, b, c * h)
        # T, B, features

        x, _ = self.rnn(x)
        x = self.classifier(x)
        # T, B, num_classes

        return x


# =========================
# DECODER
# =========================

def decode_predictions(logits):
    preds = logits.softmax(2).argmax(2)
    # T, B

    results = []

    for b in range(preds.size(1)):
        previous = BLANK_IDX
        text = ""

        for t in range(preds.size(0)):
            current = preds[t, b].item()

            if current != BLANK_IDX and current != previous:
                text += idx_to_char[current]

            previous = current

        results.append(text)

    return results


# =========================
# TRAIN / VALIDATE
# =========================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, labels, label_lengths, _ in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        label_lengths = label_lengths.to(DEVICE)

        logits = model(images)
        log_probs = logits.log_softmax(2)

        batch_size = images.size(0)
        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=logits.size(0),
            dtype=torch.long
        ).to(DEVICE)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels, label_lengths, texts in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        label_lengths = label_lengths.to(DEVICE)

        logits = model(images)
        log_probs = logits.log_softmax(2)

        batch_size = images.size(0)
        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=logits.size(0),
            dtype=torch.long
        ).to(DEVICE)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        total_loss += loss.item()

        predictions = decode_predictions(logits.cpu())

        for pred, gt in zip(predictions, texts):
            if pred == gt:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    avg_loss = total_loss / max(len(loader), 1)

    return avg_loss, acc


def main():
    print("Device:", DEVICE)

    train_dataset = PlateDataset(CSV_PATH, IMAGE_DIR, "train")
    val_dataset = PlateDataset(CSV_PATH, IMAGE_DIR, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    num_classes = len(CHARSET) + 1

    model = CRNN(num_classes=num_classes).to(DEVICE)

    criterion = nn.CTCLoss(
        blank=BLANK_IDX,
        zero_infinity=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "charset": CHARSET,
            "img_h": IMG_H,
            "img_w": IMG_W,
        }

        torch.save(checkpoint, SAVE_DIR / "last.pt")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, SAVE_DIR / "best.pt")
            print("Saved best model.")

    print("Training finished.")
    print("Best validation accuracy:", best_acc)


if __name__ == "__main__":
    main()