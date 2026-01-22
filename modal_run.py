import modal

app = modal.App("engram-gpt2-wikitext")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "datasets", "tqdm", "huggingface_hub", "wandb")
    .add_local_dir(".", remote_path="/root/engram", ignore=[".venv", "__pycache__", "*.git", "*.pt"])
)
volume = modal.Volume.from_name("engram-checkpoints", create_if_missing=True)

@app.function(
    gpu="A100",
    image=image,
    timeout=3600,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret")
    ]
)
def train(repo_id: str="engram-gpt2-wikitext", push_to_hub: bool=False):
    import subprocess
    cmd = ["python", "/root/engram/train.py"]
    if push_to_hub:
        cmd.extend(["--push-to-hub", "--repo-id", repo_id])

    subprocess.run(cmd, check=True)

@app.local_entrypoint()
def main(repo_id: str="engram-gpt2-wikitext", push_to_hub: bool=False):
    train.remote(repo_id, push_to_hub)