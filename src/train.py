import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm

from utils import set_seed, DEVICE
from model import AudioCNN
from dataset import AudioDataset, cache_features

# Paths (user supplies dataset)
DATA_ROOT = "data"
AUDIO_ROOT = os.path.join(DATA_ROOT, "audio")
CACHE_ROOT = "mel_cache"

TRAIN_CSV = os.path.join(DATA_ROOT, "metadata", "kaggle_train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "metadata", "kaggle_test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "metadata", "kaggle_sample_submission.csv")

BATCH_SIZE = 64
EPOCHS = 60
LR = 3e-4

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def main():
    set_seed()

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    cache_features(train_df, AUDIO_ROOT, f"{CACHE_ROOT}/train", train=True)
    cache_features(test_df, AUDIO_ROOT, f"{CACHE_ROOT}/test", train=False)

    train_ds = AudioDataset(train_df, f"{CACHE_ROOT}/train", train=True)
    test_ds = AudioDataset(test_df, f"{CACHE_ROOT}/test", train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = AudioCNN().to(DEVICE)

    counts = Counter(train_df.classID)
    weights = torch.tensor([1.0 / counts[i] for i in range(10)], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

    model.eval()
    preds, ids = [], []
    with torch.no_grad():
        for x, names in test_loader:
            out = model(x.to(DEVICE))
            preds.extend(out.argmax(1).cpu().numpy())
            ids.extend(names)

    sub = pd.read_csv(SAMPLE_SUB)
    sub["classID"] = preds
    sub.to_csv("submission.csv", index=False)

    torch.save(model.state_dict(), "model_weights.pth")

if __name__ == "__main__":
    main()
