import os
from PIL import Image
from huggingface_hub import hf_hub_download

def ensure_file(
    local_path: str,
    repo_id: str,
    filename: str,
) -> str:
    """
    本地没有则从 Hugging Face 自动下载。
    """
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"使用本地数据: {local_path}", flush=True)
        return local_path

    print(f"本地未找到 {local_path}", flush=True)
    print(f"正在从 Hugging Face 下载: {filename}", flush=True)
    print(f"  repo: {repo_id}", flush=True)

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=os.path.dirname(os.path.abspath(local_path)) or ".",
    )

    if os.path.abspath(downloaded) != os.path.abspath(local_path):
        if not os.path.exists(local_path):
            os.replace(downloaded, local_path)

    print(f"下载完成: {local_path}", flush=True)
    return local_path

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