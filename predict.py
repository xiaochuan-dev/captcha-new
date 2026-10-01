import torch
import numpy as np
import os
from PIL import Image

from config import *
from model import CaptchaCNNTransformer
from train import decode_prediction
from utils import resize_keep_ratio_pad


def predict(filepaths, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = CaptchaCNNTransformer(
        dim=MODEL_DIM,
        depth=MODEL_DEPTH,
        heads=MODEL_HEADS,
        num_classes=NUM_CLASSES,
        channels=CHANNELS,          # 1
        dropout=MODEL_DROPOUT,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    # 兼容 DataParallel
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    
    model.eval()

    images = []
    valid_filepaths = []

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            continue

        # 1. 打开图片
        img = Image.open(filepath).convert('RGB')

        # 2. 必须和数据生成走同一个 resize（内部强制变灰度）
        img = resize_keep_ratio_pad(img, IMG_H, IMG_W)   # 输出 mode="L", size=(128, 32)

        # 3. 和 Dataset 完全一致的处理
        img_array = np.array(img, dtype=np.uint8)                # (32, 128)
        img_tensor = torch.from_numpy(img_array).float().div_(255.0)  # [0,1]
        img_tensor = img_tensor.unsqueeze(0)                     # (1, 32, 128)

        images.append(img_tensor)
        valid_filepaths.append(filepath)

    if not images:
        return {}

    imgs = torch.stack(images, dim=0).to(device)   # (B, 1, 32, 128)

    print("输入 shape :", imgs.shape)
    print("数值范围   :", float(imgs.min()), "~", float(imgs.max()))

    with torch.no_grad():
        logits = model(imgs)   # (B, T, C)

    results = {}
    for filepath, logit in zip(valid_filepaths, logits):
        pred_str = decode_prediction(logit, IDX2CHAR)
        results[filepath] = pred_str

    return results


if __name__ == '__main__':
    model_path = './best.pth'          # 注意路径，可能是 ./checkpoints/best.pth
    res = predict(['./data/2.png'], model_path=model_path)
    print(res)