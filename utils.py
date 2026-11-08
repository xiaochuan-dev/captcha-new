from PIL import Image, ImageStat


def resize_keep_ratio_pad(
    img: Image.Image,
    target_h: int,
    target_w: int,
    fill: int = 255,
):
    src_w, src_h = img.size

    if src_w <= 0 or src_h <= 0:
        return Image.new("L", (target_w, target_h), fill)

    # 1. 计算原图四周边界像素的平均灰度
    gray = img.convert("L")

    top_edge = gray.crop((0, 0, src_w, 1))
    bottom_edge = gray.crop((0, src_h - 1, src_w, src_h))
    left_edge = gray.crop((0, 1, 1, src_h - 1))
    right_edge = gray.crop((src_w - 1, 1, src_w, src_h - 1))

    total = (
        ImageStat.Stat(top_edge).sum[0]
        + ImageStat.Stat(bottom_edge).sum[0]
        + ImageStat.Stat(left_edge).sum[0]
        + ImageStat.Stat(right_edge).sum[0]
    )

    count = (
        src_w * 2
        + max(0, src_h - 2) * 2
    )

    pad_color = round(total / count) if count else fill

    # 2. 按比例缩放
    scale = min(target_w / src_w, target_h / src_h)

    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    img = img.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR,
    )

    # 3. 计算 padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    # 4. 使用边界平均灰度填充
    canvas = Image.new("L", (target_w, target_h), pad_color)
    canvas.paste(img, (left, top))

    return canvas