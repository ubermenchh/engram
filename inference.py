import torch
from transformers import AutoTokenizer

from compression import VocabCompressor
from model import GPT2WithEngram

repo_id = "ubermenchh/nanogpt-engram-wikitext" 
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {repo_id}...")
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Rebuild Vocab Map
compressor = VocabCompressor(tokenizer)
vocab_map = compressor.build_mapping().to(device)

# Load Model
model = GPT2WithEngram.from_pretrained(repo_id, vocab_map=vocab_map)
model.to(device)
model.eval()

# Generate
prompt = "The history of science is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

print(f"\nPrompt: {prompt}")
print("-" * 20)

with torch.no_grad():
    output_ids = model.gpt2.generate(
        input_ids, 
        max_new_tokens=50, 
        do_sample=True, 
        temperature=0.7,
        top_k=50
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))