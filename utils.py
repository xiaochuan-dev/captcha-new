from PIL import Image

def resize_keep_ratio_pad(
    img: Image.Image,
    target_h: int,
    target_w: int,
    fill: int = 255,
    threshold: int = 128,
):
    src_w, src_h = img.size

    if src_w <= 8 or src_h <= 8:
        return Image.new("L", (target_w, target_h), fill)

    # 1. 裁剪四周各 4px
    img = img.crop((
        4,
        4,
        src_w - 4,
        src_h - 4,
    ))

    # 2. 转换为灰度图
    img = img.convert("L")

    # 3. 强制缩放到目标尺寸
    img = img.resize(
        (target_w, target_h),
        Image.Resampling.BILINEAR,
    )

    # 4. 二值化
    img = img.point(
        lambda pixel: 0 if pixel < threshold else 255
    )

    return img