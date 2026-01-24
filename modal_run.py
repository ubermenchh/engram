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
    timeout=7200,
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

@app.function(
    gpu="A100",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def inference(prompt: str):
    import sys
    sys.path.append("/root/engram")

    import torch
    from transformers import AutoTokenizer, GPT2Config

    from compression import VocabCompressor
    from config import EngramConfig
    from model import GPT2WithEngram

    nanogpt_config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=384,
        n_layer=6,
        n_head=6,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1
    )
    config = EngramConfig()

    repo_id = "ubermenchh/nanogpt-engram-wikitext"
    device = "cuda"

    print(f"Loading model from {repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    
    compressor = VocabCompressor(tokenizer)
    vocab_map = compressor.build_mapping().to(device)

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    try:
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = load_file(model_path)
    except Exception:
        # Fallback to pytorch_model.bin
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "") # Remove the compile prefix
        new_state_dict[new_key] = value

    model = GPT2WithEngram(config, vocab_map, nanogpt_config=nanogpt_config)
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval() 
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.gpt2.generate(
            input_ids, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.7,
            top_k=50
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

@app.local_entrypoint()
def run_inference(prompt: str = "The history of science is"):
    result = inference.remote(prompt)
    print("\nGenerated Text:")
    print("-" * 20)
    print(result)