import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import RetinaDataset
from model import BuildUnet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time

HEIGHT = 512
WIDTH = 512
SIZE = (HEIGHT, WIDTH)
BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "files/checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model_t, loader, optimizer_t, loss_fn_t):
    epoch_loss = 0.0

    model_t.train()
    for x, y in loader:
        x = x.to(DEVICE, dtype=torch.float32)
        y = y.to(DEVICE, dtype=torch.float32)

        optimizer_t.zero_grad()
        y_pred = model_t(x)
        loss = loss_fn_t(y_pred, y)
        loss.backward()
        optimizer_t.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def evaluate(model_e, loader, loss_fn_e):
    epoch_loss = 0.0

    model_e.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32)

            y_pred = model_e(x)
            loss = loss_fn_e(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
    return epoch_loss


if __name__ == "__main__":
    seeding(42)
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("new_data/train/image/*"))
    train_y = sorted(glob("new_data/train/mask/*"))

    test_x = sorted(glob("new_data/test/image/*"))
    test_y = sorted(glob("new_data/test/mask/*"))

    data_str = f"Dataset SIZE:\nTrain: {len(train_x)} - Test: {len(test_x)}\n"
    print(data_str)

    train_dataset = RetinaDataset(train_x, train_y)
    test_dataset = RetinaDataset(test_x, test_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    model = BuildUnet()
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn)
        valid_loss = evaluate(model, test_loader, loss_fn)

        if valid_loss < best_valid_loss:
            print(f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
