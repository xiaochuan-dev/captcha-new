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
)
from model import CaptchaCNNTransformer
from dataset import CaptchaDataset, ctc_collate_fn


def decode_prediction(
    logits,
    idx2char=IDX2CHAR,
    blank_idx=0,
):
    """
    CTC greedy decode。

    logits:
        [B, T, C]
    """

    pred = logits.argmax(dim=-1)

    prev = blank_idx
    chars = []

    for p in pred:
        p = p.item()

        if p != prev and p != blank_idx:
            chars.append(
                idx2char.get(p, "")
            )

        prev = p

    return "".join(chars)


def train():

    # ============================================================
    # Device
    # ============================================================

    n_gpu = torch.cuda.device_count()

    if n_gpu > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_dp = n_gpu > 1

    print(
        f"Using device: {device} | "
        f"GPU 数: {n_gpu} | "
        f"DataParallel: {use_dp}",
        flush=True,
    )

    # ============================================================
    # Dataset
    # ============================================================
    #
    # 不再传 LOCAL_PARQUET。
    #
    # CaptchaDataset() 会自动：
    #
    # 本地存在
    #     ↓
    # 使用 .npy
    #
    # 本地不存在
    #     ↓
    # 从 Hugging Face 下载
    #
    # 然后使用 mmap 读取。
    #

    full_dataset = CaptchaDataset()

    # ============================================================
    # Train / Validation split
    # ============================================================

    val_size = max(
        1,
        int(VAL_RATIO * len(full_dataset)),
    )

    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(
        f"Train: {train_size:,} | "
        f"Val: {val_size:,}",
        flush=True,
    )

    # ============================================================
    # DataLoader
    # ============================================================

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": ctc_collate_fn,
    }

    # Windows / Linux 都支持。
    #
    # num_workers=0 时不能使用 persistent_workers / prefetch_factor。
    if NUM_WORKERS > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 4,
            }
        )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **loader_kwargs,
    )

    print(
        f"DataLoader workers: {NUM_WORKERS}",
        flush=True,
    )

    print(
        f"Batch size: {BATCH_SIZE}",
        flush=True,
    )

    print(
        f"Train iterations: {len(train_loader):,}",
        flush=True,
    )

    print(
        f"Val iterations: {len(val_loader):,}",
        flush=True,
    )

    # ============================================================
    # Model
    # ============================================================

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

    # ============================================================
    # DataParallel
    # ============================================================

    if use_dp:
        model = nn.DataParallel(model)

        print(
            f"已启用 DataParallel，"
            f"使用 {n_gpu} 张卡",
            flush=True,
        )

    # ============================================================
    # Loss
    # ============================================================

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True,
    )

    # ============================================================
    # Optimizer
    # ============================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ============================================================
    # Scheduler
    # ============================================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    # ============================================================
    # Checkpoints
    # ============================================================

    best_acc = 0.0

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    # ============================================================
    # Training
    # ============================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
        )

        for imgs, labels, label_lengths in pbar:

            # ====================================================
            # CPU -> GPU
            # ====================================================

            imgs = imgs.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            label_lengths = label_lengths.to(
                device,
                non_blocking=True,
            )

            # ====================================================
            # Forward
            # ====================================================

            logits = model(imgs)

            # [B,T,C]
            # ->
            # [T,B,C]

            log_probs = (
                logits
                .log_softmax(dim=-1)
                .permute(1, 0, 2)
            )

            # ====================================================
            # CTC input lengths
            # ====================================================

            input_lengths = torch.full(
                (imgs.size(0),),
                fill_value=logits.size(1),
                dtype=torch.long,
                device=device,
            )

            # ====================================================
            # Loss
            # ====================================================

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths,
            )

            # ====================================================
            # Backward
            # ====================================================

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            # ====================================================
            # Logging
            # ====================================================

            total_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # ========================================================
        # Scheduler
        # ========================================================

        scheduler.step()

        avg_loss = (
            total_loss
            / max(len(train_loader), 1)
        )

        # ========================================================
        # Validation
        # ========================================================

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for imgs, labels, label_lengths in val_loader:

                imgs = imgs.to(
                    device,
                    non_blocking=True,
                )

                logits = model(imgs)

                # ================================================
                # Greedy CTC decode
                # ================================================

                for i in range(imgs.size(0)):

                    pred_str = decode_prediction(
                        logits[i]
                    )

                    start = (
                        int(label_lengths[:i].sum())
                        if i > 0
                        else 0
                    )

                    end = (
                        start
                        + int(label_lengths[i])
                    )

                    true_indices = labels[
                        start:end
                    ].tolist()

                    true_str = "".join(
                        IDX2CHAR.get(
                            idx,
                            "",
                        )
                        for idx in true_indices
                    )

                    if pred_str == true_str:
                        correct += 1

                    total += 1

        # ========================================================
        # Accuracy
        # ========================================================

        acc = (
            correct / total
            if total
            else 0.0
        )

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {acc:.4f}",
            flush=True,
        )

        # ========================================================
        # Save best
        # ========================================================

        if acc > best_acc:

            best_acc = acc

            if isinstance(
                model,
                nn.DataParallel,
            ):
                state = model.module.state_dict()
            else:
                state = model.state_dict()

            torch.save(
                state,
                "checkpoints/best.pth",
            )

            save_file(
                state,
                "checkpoints/model.safetensors",
            )

            print(
                f"  → 保存最佳模型 "
                f"(acc={acc:.4f})",
                flush=True,
            )

    print(
        f"训练完成！最佳准确率: {best_acc:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    train()
