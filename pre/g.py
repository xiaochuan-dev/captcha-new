import io
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from captcha.image import ImageCaptcha
from utils import resize_keep_ratio_pad


# ============================================================
# 配置
# ============================================================

OUTPUT_DIR = Path("./data")

NUM_IMAGES = 60000

# 生成时用稍大尺寸，最后 resize 到模型输入
GEN_WIDTH = 160
GEN_HEIGHT = 60

# 模型输入尺寸（与 config.py 一致）
IMG_H = 32
IMG_W = 128

CAPTCHA_LENGTH = 4

# 生成时允许大小写 + 数字（图片上可以出现大写）
GEN_CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 词表（与 config.py 完全一致）：只保留数字 + 小写
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}   # 1~36

SEED = 42

STYLE_WEIGHTS = {
    "clean": 5,
    "dots": 15,
    "lines": 15,
    "curves": 15,
    "dots_lines": 15,
    "grid": 8,
    "heavy": 12,
    "blur": 5,
    "distorted": 5,
    "extreme": 5,
}


# ============================================================
# 随机数
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 随机验证码（允许大写）
# ============================================================

def random_text():
    # 注意：这里不要 .lower()，让图片可以出现大写
    return "".join(
        random.choice(GEN_CHARSET)
        for _ in range(CAPTCHA_LENGTH)
    )


# ============================================================
# captcha 基础生成
# ============================================================

def create_base_captcha(text):
    generator = ImageCaptcha(
        width=GEN_WIDTH,
        height=GEN_HEIGHT,
        font_sizes=(
            random.randint(32, 40),
            random.randint(34, 42),
        ),
    )
    image = generator.generate_image(text)
    return image.convert("RGB")


# ============================================================
# 随机颜色
# ============================================================

def random_dark_color():
    value = random.randint(20, 130)
    return (value, value, value)


def random_color():
    return (
        random.randint(0, 180),
        random.randint(0, 180),
        random.randint(0, 180),
    )


# ============================================================
# 噪点
# ============================================================

def add_noise(image, amount=None):
    if amount is None:
        amount = random.randint(100, 800)
    draw = ImageDraw.Draw(image)
    for _ in range(amount):
        x = random.randrange(GEN_WIDTH)
        y = random.randrange(GEN_HEIGHT)
        draw.point((x, y), fill=random_dark_color())
    return image


def add_color_noise(image, amount=None):
    if amount is None:
        amount = random.randint(100, 500)
    draw = ImageDraw.Draw(image)
    for _ in range(amount):
        x = random.randrange(GEN_WIDTH)
        y = random.randrange(GEN_HEIGHT)
        draw.point((x, y), fill=random_color())
    return image


# ============================================================
# 干扰直线
# ============================================================

def add_lines(image, count=None):
    if count is None:
        count = random.randint(1, 6)
    draw = ImageDraw.Draw(image)
    for _ in range(count):
        x1 = random.randint(0, GEN_WIDTH)
        y1 = random.randint(0, GEN_HEIGHT)
        x2 = random.randint(0, GEN_WIDTH)
        y2 = random.randint(0, GEN_HEIGHT)
        width = random.randint(1, 6)
        draw.line((x1, y1, x2, y2), fill=random_color(), width=width)
    return image


# ============================================================
# 干扰曲线
# ============================================================

def add_curves(image, count=None):
    if count is None:
        count = random.randint(1, 4)
    draw = ImageDraw.Draw(image)
    for _ in range(count):
        points = []
        start_y = random.randint(0, GEN_HEIGHT)
        amplitude = random.randint(5, 20)
        frequency = random.uniform(0.015, 0.05)
        phase = random.uniform(0, math.pi * 2)
        for x in range(GEN_WIDTH):
            y = start_y + amplitude * math.sin(frequency * x + phase)
            points.append((x, int(y)))
        draw.line(points, fill=random_color(), width=random.randint(1, 2))
    return image


# ============================================================
# 网格
# ============================================================

def add_grid(image):
    draw = ImageDraw.Draw(image)
    spacing_x = random.randint(15, 30)
    spacing_y = random.randint(10, 20)
    color = random_color()
    for x in range(0, GEN_WIDTH, spacing_x):
        draw.line((x, 0, x, GEN_HEIGHT), fill=color, width=1)
    for y in range(0, GEN_HEIGHT, spacing_y):
        draw.line((0, y, GEN_WIDTH, y), fill=color, width=1)
    return image


# ============================================================
# 波纹
# ============================================================

def add_wave(image):
    arr = np.array(image)
    h, w = arr.shape[:2]
    result = np.zeros_like(arr)
    amplitude = random.uniform(1.5, 5.0)
    frequency = random.uniform(0.03, 0.08)
    for y in range(h):
        offset = int(amplitude * math.sin(frequency * y))
        if offset >= 0:
            result[y, offset:] = arr[y, :w - offset]
        else:
            result[y, :w + offset] = arr[y, -offset:]
    return Image.fromarray(result)


# ============================================================
# 随机旋转
# ============================================================

def random_rotate(image):
    angle = random.uniform(-8, 8)
    return image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )


# ============================================================
# 随机模糊
# ============================================================

def random_blur(image):
    radius = random.uniform(0.3, 1.4)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


# ============================================================
# 对比度 / 亮度
# ============================================================

def random_brightness_contrast(image):
    brightness = random.uniform(0.75, 1.25)
    contrast = random.uniform(0.7, 1.4)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


# ============================================================
# JPEG 压缩伪影
# ============================================================

def jpeg_artifact(image):
    quality = random.randint(25, 85)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# ============================================================
# 轻微遮挡
# ============================================================

def add_random_blocks(image):
    draw = ImageDraw.Draw(image)
    for _ in range(random.randint(1, 4)):
        x1 = random.randint(0, GEN_WIDTH - 10)
        y1 = random.randint(0, GEN_HEIGHT - 10)
        w = random.randint(3, 15)
        h = random.randint(2, 8)
        draw.rectangle((x1, y1, x1 + w, y1 + h), fill=random_color())
    return image


# ============================================================
# 轻微像素噪声
# ============================================================

def add_gaussian_noise(image):
    arr = np.asarray(image).astype(np.float32)
    sigma = random.uniform(3, 18)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ============================================================
# 风格选择
# ============================================================

def choose_style():
    styles = list(STYLE_WEIGHTS.keys())
    weights = list(STYLE_WEIGHTS.values())
    return random.choices(styles, weights=weights, k=1)[0]


# ============================================================
# 应用 CAPTCHA 风格
# ============================================================

def apply_style(image, style):
    if style == "clean":
        return image

    if style == "dots":
        return add_noise(image, random.randint(100, 500))

    if style == "lines":
        return add_lines(image, random.randint(2, 5))

    if style == "curves":
        return add_curves(image, random.randint(1, 4))

    if style == "dots_lines":
        image = add_noise(image, random.randint(150, 600))
        return add_lines(image, random.randint(2, 6))

    if style == "grid":
        image = add_grid(image)
        return add_noise(image, random.randint(100, 400))

    if style == "heavy":
        image = add_noise(image, random.randint(300, 1000))
        image = add_lines(image, random.randint(3, 8))
        return add_curves(image, random.randint(2, 5))

    if style == "blur":
        image = add_noise(image, random.randint(100, 500))
        image = add_lines(image, random.randint(1, 4))
        return random_blur(image)

    if style == "distorted":
        image = random_rotate(image)
        image = add_wave(image)
        image = add_noise(image, random.randint(100, 500))
        return add_curves(image, random.randint(1, 3))

    if style == "extreme":
        if random.random() < 0.8:
            image = add_noise(image, random.randint(400, 1500))
        if random.random() < 0.8:
            image = add_lines(image, random.randint(3, 9))
        if random.random() < 0.7:
            image = add_curves(image, random.randint(2, 6))
        if random.random() < 0.4:
            image = add_grid(image)
        if random.random() < 0.3:
            image = add_random_blocks(image)
        if random.random() < 0.5:
            image = random_rotate(image)
        if random.random() < 0.5:
            image = add_wave(image)
        if random.random() < 0.5:
            image = random_blur(image)
        if random.random() < 0.6:
            image = add_gaussian_noise(image)
        return image

    return image


# ============================================================
# 通用随机增强
# ============================================================

def random_postprocess(image):
    if random.random() < 0.35:
        image = random_brightness_contrast(image)
    if random.random() < 0.20:
        image = add_gaussian_noise(image)
    if random.random() < 0.15:
        image = jpeg_artifact(image)
    return image


# ============================================================
# 标签转换：大写 → 小写 → 索引
# ============================================================

def text_to_indices(text: str) -> list[int]:
    text = text.lower()                     # 大写映射成小写
    return [CHAR2IDX[c] for c in text]


# ============================================================
# 生成一张
# ============================================================

def generate_one():
    text = random_text()                    # 可能包含大写

    image = create_base_captcha(text)
    style = choose_style()
    image = apply_style(image, style)
    image = random_postprocess(image)

    # 缩放到模型输入尺寸
    # image = image.resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)
    # image = image.convert("L")              # 灰度
    image = resize_keep_ratio_pad(img=image, target_h=IMG_H, target_w=IMG_W)

    # 标签：大写 → 小写 → 索引
    label_indices = text_to_indices(text)

    return image, label_indices, style


# ============================================================
# 主程序
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {NUM_IMAGES:,} CAPTCHA images...")
    print(f"生成尺寸: {GEN_WIDTH}x{GEN_HEIGHT} → 最终尺寸: {IMG_W}x{IMG_H}")
    print(f"词表大小: {len(CHARSET)} (不含 blank)")

    images = []
    labels = []

    for i in range(NUM_IMAGES):
        image, label_indices, style = generate_one()

        arr = np.asarray(image, dtype=np.uint8)   # (32, 128)
        images.append(arr)
        labels.append(label_indices)              # [int, int, int, int]

        if (i + 1) % 500 == 0:
            print(f"{i + 1:,} / {NUM_IMAGES:,}")

    # 关键：保存为 uint8 数值数组，而不是 object 字符串
    images_arr = np.stack(images, axis=0)         # (N, 32, 128) uint8
    labels_arr = np.array(labels, dtype=np.uint8) # (N, 4) uint8

    images_path = OUTPUT_DIR / "images.npy"
    labels_path = OUTPUT_DIR / "labels.npy"

    np.save(images_path, images_arr)
    np.save(labels_path, labels_arr)

    print()
    print("Finished!")
    print(f"Images shape : {images_arr.shape}  dtype={images_arr.dtype}")
    print(f"Labels shape : {labels_arr.shape}  dtype={labels_arr.dtype}")
    print(f"Saved: {images_path}")
    print(f"Saved: {labels_path}")

    # 简单验证
    print("\n示例标签（前5个）:")
    for i in range(min(5, len(labels_arr))):
        idxs = labels_arr[i]
        chars = "".join(CHARSET[idx - 1] for idx in idxs)
        print(f"  {idxs} → '{chars}'")


if __name__ == "__main__":
    main()