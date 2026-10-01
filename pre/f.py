from huggingface_hub import HfApi

def upload(filename):
    api = HfApi()

    REPO_ID = "xiaochuan-dev/captcha-new"
    LOCAL_PARQUET = f"./{filename}"

    api.upload_file(
        path_or_fileobj=LOCAL_PARQUET,
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Upload length-4 captcha dataset (RGB bitmap)",
    )
