def resize_keep_ratio_pad(
    img: Image.Image,
    target_h: int,
    target_w: int,
    fill: int = 255,
):
    src_w, src_h = img.size

    if src_w <= 8 or src_h <= 8:
        return Image.new("L", (target_w, target_h), fill)

    # 1. 裁剪四周各 4 个像素
    img = img.crop((
        4,
        4,
        src_w - 4,
        src_h - 4,
    ))

    src_w, src_h = img.size

    # 2. 按比例缩放
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

    # 3. 创建画布
    canvas = Image.new(
        "L",
        (target_w, target_h),
        fill,
    )

    # 4. 居中 padding
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    canvas.paste(img, (offset_x, offset_y))

    return canvas