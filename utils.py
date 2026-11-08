from PIL import Image

def resize_keep_ratio_pad(
    img: Image.Image,
    target_h: int,
    target_w: int,
    fill: int = 255,
):
    src_w, src_h = img.size

    if src_w <= 0 or src_h <= 0:
        return Image.new("L", (target_w, target_h), fill)

    # 1. 转换为灰度图
    img = img.convert("L")

    # 2. 强制缩放到目标尺寸
    img = img.resize(
        (target_w, target_h),
        Image.Resampling.BILINEAR,
    )

    return img