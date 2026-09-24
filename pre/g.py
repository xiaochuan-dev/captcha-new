import io
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from captcha.image import ImageCaptcha


# ============================================================
# 配置
# ============================================================

OUTPUT_DIR = Path("./data")

NUM_IMAGES = 60000

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 60

# 固定 4 字符
CAPTCHA_LENGTH = 4

# 字符集：数字 + 小写 + 大写
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 随机种子
SEED = 42

# 每种风格大致占比
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
# 随机验证码
# ============================================================

def random_text():
    return "".join(
        random.choice(CHARSET)
        for _ in range(CAPTCHA_LENGTH)
    ).lower()


# ============================================================
# captcha 基础生成
# ============================================================

def create_base_captcha(text):
    """
    使用 captcha 包生成基础验证码。
    """

    generator = ImageCaptcha(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        font_sizes=(
            random.randint(32, 40),
            random.randint(34, 42),
        ),
    )

    # captcha 返回 PIL Image
    image = generator.generate_image(text)

    return image.convert("RGB")


# ============================================================
# 随机颜色
# ============================================================

def random_dark_color():
    value = random.randint(20, 130)

    return (
        value,
        value,
        value,
    )


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
    """
    添加随机噪点。
    """

    if amount is None:
        amount = random.randint(100, 800)

    draw = ImageDraw.Draw(image)

    for _ in range(amount):
        x = random.randrange(IMAGE_WIDTH)
        y = random.randrange(IMAGE_HEIGHT)

        color = random_dark_color()

        draw.point(
            (x, y),
            fill=color,
        )

    return image


# ============================================================
# 彩色噪点
# ============================================================

def add_color_noise(image, amount=None):
    if amount is None:
        amount = random.randint(100, 500)

    draw = ImageDraw.Draw(image)

    for _ in range(amount):
        x = random.randrange(IMAGE_WIDTH)
        y = random.randrange(IMAGE_HEIGHT)

        draw.point(
            (x, y),
            fill=random_color(),
        )

    return image


# ============================================================
# 干扰直线
# ============================================================

def add_lines(image, count=None):
    if count is None:
        count = random.randint(1, 6)

    draw = ImageDraw.Draw(image)

    for _ in range(count):

        x1 = random.randint(0, IMAGE_WIDTH)
        y1 = random.randint(0, IMAGE_HEIGHT)

        x2 = random.randint(0, IMAGE_WIDTH)
        y2 = random.randint(0, IMAGE_HEIGHT)

        width = random.randint(1, 6)

        draw.line(
            (x1, y1, x2, y2),
            fill=random_color(),
            width=width,
        )

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

        start_y = random.randint(
            0,
            IMAGE_HEIGHT,
        )

        amplitude = random.randint(
            5,
            20,
        )

        frequency = random.uniform(
            0.015,
            0.05,
        )

        phase = random.uniform(
            0,
            math.pi * 2,
        )

        for x in range(IMAGE_WIDTH):

            y = (
                start_y
                + amplitude
                * math.sin(
                    frequency * x + phase
                )
            )

            points.append(
                (
                    x,
                    int(y),
                )
            )

        draw.line(
            points,
            fill=random_color(),
            width=random.randint(1, 2),
        )

    return image


# ============================================================
# 网格
# ============================================================

def add_grid(image):
    draw = ImageDraw.Draw(image)

    spacing_x = random.randint(15, 30)
    spacing_y = random.randint(10, 20)

    color = random_color()

    for x in range(
        0,
        IMAGE_WIDTH,
        spacing_x,
    ):
        draw.line(
            (x, 0, x, IMAGE_HEIGHT),
            fill=color,
            width=1,
        )

    for y in range(
        0,
        IMAGE_HEIGHT,
        spacing_y,
    ):
        draw.line(
            (0, y, IMAGE_WIDTH, y),
            fill=color,
            width=1,
        )

    return image


# ============================================================
# 波纹
# ============================================================

def add_wave(image):
    """
    使用 displacement 做轻微波纹。
    """

    arr = np.array(image)

    h, w = arr.shape[:2]

    result = np.zeros_like(arr)

    amplitude = random.uniform(1.5, 5.0)
    frequency = random.uniform(
        0.03,
        0.08,
    )

    for y in range(h):

        offset = int(
            amplitude
            * math.sin(
                frequency * y
            )
        )

        if offset >= 0:
            result[
                y,
                offset:,
            ] = arr[
                y,
                :w - offset,
            ]
        else:
            result[
                y,
                :w + offset,
            ] = arr[
                y,
                -offset:,
            ]

    return Image.fromarray(result)


# ============================================================
# 随机旋转
# ============================================================

def random_rotate(image):
    angle = random.uniform(
        -8,
        8,
    )

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
    radius = random.uniform(
        0.3,
        1.4,
    )

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )


# ============================================================
# 对比度 / 亮度
# ============================================================

def random_brightness_contrast(image):

    brightness = random.uniform(
        0.75,
        1.25,
    )

    contrast = random.uniform(
        0.7,
        1.4,
    )

    image = ImageEnhance.Brightness(
        image
    ).enhance(brightness)

    image = ImageEnhance.Contrast(
        image
    ).enhance(contrast)

    return image


# ============================================================
# JPEG 压缩伪影
# ============================================================

def jpeg_artifact(image):

    quality = random.randint(
        25,
        85,
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    return Image.open(buffer).convert(
        "RGB"
    )


# ============================================================
# 轻微遮挡
# ============================================================

def add_random_blocks(image):

    draw = ImageDraw.Draw(image)

    count = random.randint(
        1,
        4,
    )

    for _ in range(count):

        x1 = random.randint(
            0,
            IMAGE_WIDTH - 10,
        )

        y1 = random.randint(
            0,
            IMAGE_HEIGHT - 10,
        )

        width = random.randint(
            3,
            15,
        )

        height = random.randint(
            2,
            8,
        )

        draw.rectangle(
            (
                x1,
                y1,
                x1 + width,
                y1 + height,
            ),
            fill=random_color(),
        )

    return image


# ============================================================
# 轻微像素噪声
# ============================================================

def add_gaussian_noise(image):

    arr = np.asarray(
        image
    ).astype(
        np.float32
    )

    sigma = random.uniform(
        3,
        18,
    )

    noise = np.random.normal(
        0,
        sigma,
        arr.shape,
    )

    arr += noise

    arr = np.clip(
        arr,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        arr
    )


# ============================================================
# 风格选择
# ============================================================

def choose_style():

    styles = list(
        STYLE_WEIGHTS.keys()
    )

    weights = list(
        STYLE_WEIGHTS.values()
    )

    return random.choices(
        styles,
        weights=weights,
        k=1,
    )[0]


# ============================================================
# 应用 CAPTCHA 风格
# ============================================================

def apply_style(image, style):

    if style == "clean":

        return image

    # --------------------------------------------------------
    # 少量噪点
    # --------------------------------------------------------

    if style == "dots":

        image = add_noise(
            image,
            random.randint(
                100,
                500,
            ),
        )

        return image

    # --------------------------------------------------------
    # 干扰线
    # --------------------------------------------------------

    if style == "lines":

        image = add_lines(
            image,
            random.randint(
                2,
                5,
            ),
        )

        return image

    # --------------------------------------------------------
    # 曲线
    # --------------------------------------------------------

    if style == "curves":

        image = add_curves(
            image,
            random.randint(
                1,
                4,
            ),
        )

        return image

    # --------------------------------------------------------
    # 噪点 + 干扰线
    # --------------------------------------------------------

    if style == "dots_lines":

        image = add_noise(
            image,
            random.randint(
                150,
                600,
            ),
        )

        image = add_lines(
            image,
            random.randint(
                2,
                6,
            ),
        )

        return image

    # --------------------------------------------------------
    # 网格
    # --------------------------------------------------------

    if style == "grid":

        image = add_grid(
            image
        )

        image = add_noise(
            image,
            random.randint(
                100,
                400,
            ),
        )

        return image

    # --------------------------------------------------------
    # heavy
    # --------------------------------------------------------

    if style == "heavy":

        image = add_noise(
            image,
            random.randint(
                300,
                1000,
            ),
        )

        image = add_lines(
            image,
            random.randint(
                3,
                8,
            ),
        )

        image = add_curves(
            image,
            random.randint(
                2,
                5,
            ),
        )

        return image

    # --------------------------------------------------------
    # blur
    # --------------------------------------------------------

    if style == "blur":

        image = add_noise(
            image,
            random.randint(
                100,
                500,
            ),
        )

        image = add_lines(
            image,
            random.randint(
                1,
                4,
            ),
        )

        image = random_blur(
            image
        )

        return image

    # --------------------------------------------------------
    # distorted
    # --------------------------------------------------------

    if style == "distorted":

        image = random_rotate(
            image
        )

        image = add_wave(
            image
        )

        image = add_noise(
            image,
            random.randint(
                100,
                500,
            ),
        )

        image = add_curves(
            image,
            random.randint(
                1,
                3,
            ),
        )

        return image

    # --------------------------------------------------------
    # extreme
    # --------------------------------------------------------

    if style == "extreme":

        if random.random() < 0.8:
            image = add_noise(
                image,
                random.randint(
                    400,
                    1500,
                ),
            )

        if random.random() < 0.8:
            image = add_lines(
                image,
                random.randint(
                    3,
                    9,
                ),
            )

        if random.random() < 0.7:
            image = add_curves(
                image,
                random.randint(
                    2,
                    6,
                ),
            )

        if random.random() < 0.4:
            image = add_grid(
                image
            )

        if random.random() < 0.3:
            image = add_random_blocks(
                image
            )

        if random.random() < 0.5:
            image = random_rotate(
                image
            )

        if random.random() < 0.5:
            image = add_wave(
                image
            )

        if random.random() < 0.5:
            image = random_blur(
                image
            )

        if random.random() < 0.6:
            image = add_gaussian_noise(
                image
            )

        return image

    return image


# ============================================================
# 通用随机增强
# ============================================================

def random_postprocess(image):

    # 亮度/对比度
    if random.random() < 0.35:
        image = random_brightness_contrast(
            image
        )

    # 高斯噪声
    if random.random() < 0.20:
        image = add_gaussian_noise(
            image
        )

    # JPEG
    if random.random() < 0.15:
        image = jpeg_artifact(
            image
        )

    return image


# ============================================================
# 生成一张
# ============================================================

def generate_one():

    text = random_text()

    image = create_base_captcha(
        text
    )

    style = choose_style()

    image = apply_style(
        image,
        style
    )

    image = random_postprocess(
        image
    )

    image = image.resize(
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        ),
        Image.Resampling.LANCZOS,
    )

    # 转成灰度图
    image = image.convert("L")

    return image, text, style


# ============================================================
# 主程序
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Generating {NUM_IMAGES:,} CAPTCHA images..."
    )

    images = []
    labels = []

    for i in range(NUM_IMAGES):

        image, text, style = generate_one()

        # 转成 numpy 数组 (H, W)，uint8 灰度
        arr = np.asarray(image, dtype=np.uint8)
        images.append(arr)
        labels.append(text)

        if (i + 1) % 100 == 0:
            print(f"{i + 1:,} / {NUM_IMAGES:,}")

    # 堆叠成 (N, H, W)
    images_arr = np.stack(images, axis=0)
    labels_arr = np.array(labels, dtype=object)   # 字符串数组

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


if __name__ == "__main__":
    main()