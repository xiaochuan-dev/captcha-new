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

    scale = min(
        target_w / src_w,
        target_h / src_h,
    )

    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    img = img.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR,
    )

    canvas = Image.new(
        "L",
        (target_w, target_h),
        fill,
    )

    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    canvas.paste(img, (offset_x, offset_y))

    return canvas
def resize_crop_border(
    img: Image.Image,
    target_h: int,
    target_w: int,
    border: int = 4,
    fill: int = 255,
):
    """
    1. 先裁掉四周 border 像素
    2. 再直接 resize 到目标大小（不保持比例）
    """
    src_w, src_h = img.size

    # 防止图片太小导致裁剪后宽高为负
    if src_w <= 2 * border or src_h <= 2 * border:
        # 图片太小，直接返回目标尺寸的空白图
        return Image.new("L", (target_w, target_h), fill)

    # 裁掉四周
    left = border
    top = border
    right = src_w - border
    bottom = src_h - border

    img = img.crop((left, top, right, bottom))

    # 直接缩放到目标尺寸
    img = img.resize(
        (target_w, target_h),
        Image.Resampling.BILINEAR,
    )

    return img