from huggingface_hub import HfApi, login

# Paste your Hugging Face token here (from Step 1)
HF_TOKEN = "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"

# Replace "yourname" with your actual Hugging Face username
REPO_ID = "somanpatrasom/meditrak-demand-model"

login(token=HF_TOKEN)

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj="models/demand_model.pkl",
    path_in_repo="demand_model.pkl",
    repo_id=REPO_ID,
    repo_type="model"
)

print("Upload complete!")
print(f"Anyone can now download it from: https://huggingface.co/{REPO_ID}")