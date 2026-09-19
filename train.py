"""
训练入口。无命令行参数。

  python train.py

多卡（>1）时自动使用 nn.DataParallel。
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from safetensors.torch import save_file

from config import (
    NUM_CLASSES,
    IDX2CHAR,
    IMG_H,
    IMG_W,
    CHANNELS,
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    VAL_RATIO,
    NUM_WORKERS,
    MODEL_DIM,
    MODEL_DEPTH,
    MODEL_HEADS,
    MODEL_DROPOUT,
    LOCAL_PARQUET,
)
from model import CaptchaCNNTransformer
from dataset import CaptchaDataset, ctc_collate_fn


def decode_prediction(logits, idx2char=IDX2CHAR, blank_idx=0):
    pred = logits.argmax(dim=-1)
    prev = blank_idx
    chars = []
    for p in pred:
        p = p.item()
        if p != prev and p != blank_idx:
            chars.append(idx2char.get(p, ""))
        prev = p
    return "".join(chars)


def train():
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_dp = n_gpu > 1
    print(f"Using device: {device} | GPU 数: {n_gpu} | DataParallel: {use_dp}")

    full_dataset = CaptchaDataset(LOCAL_PARQUET)

    val_size = max(1, int(VAL_RATIO * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size:,} | Val: {val_size:,}")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ctc_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ctc_collate_fn,
    )

    model = CaptchaCNNTransformer(
        img_h=IMG_H,
        img_w=IMG_W,
        dim=MODEL_DIM,
        depth=MODEL_DEPTH,
        heads=MODEL_HEADS,
        num_classes=NUM_CLASSES,
        channels=CHANNELS,
        dropout=MODEL_DROPOUT,
    ).to(device)

    if use_dp:
        model = nn.DataParallel(model)
        print(f"已启用 DataParallel，使用 {n_gpu} 张卡")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, labels, label_lengths in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)

            logits = model(imgs)
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)

            input_lengths = torch.full(
                (imgs.size(0),),
                fill_value=logits.size(1),
                dtype=torch.long,
                device=device,
            )

            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels, label_lengths in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                logits = model(imgs)

                for i in range(imgs.size(0)):
                    pred_str = decode_prediction(logits[i])
                    start = int(label_lengths[:i].sum()) if i > 0 else 0
                    end = start + int(label_lengths[i])
                    true_indices = labels[start:end].tolist()
                    true_str = "".join(
                        IDX2CHAR.get(idx, "") for idx in true_indices
                    )
                    if pred_str == true_str:
                        correct += 1
                    total += 1

        acc = correct / total if total else 0.0
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state, "checkpoints/best.pth")
            save_file(state, "checkpoints/model.safetensors")
            print(f"  → 保存最佳模型 (acc={acc:.4f})")

    print("训练完成！最佳准确率:", best_acc)


if __name__ == "__main__":
    train()
