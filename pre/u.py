from huggingface_hub import HfApi, login

api = HfApi()

REPO_ID = "xiaochuan-dev/captcha"
LOCAL_PARQUET = "./data/captcha_fixed_labels.npy"

api.upload_file(
    path_or_fileobj=LOCAL_PARQUET,
    path_in_repo="captcha_fixed_labels.npy",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Upload length-4 captcha dataset (RGB bitmap)",
)

print(f"上传完成 → https://huggingface.co/datasets/{REPO_ID}")