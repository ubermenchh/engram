import argparse

import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from compression import VocabCompressor
from config import EngramConfig
from model import GPT2WithEngram

parser = argparse.ArgumentParser()
parser.add_argument("--push-to-hub", action="store_true", help="Push model to Hugging Face Hub after training")
parser.add_argument("--repo-id", type=str, default="engram-gpt2-wikitext", help="Hugging Face repository ID")
parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
args = parser.parse_args()

bs = 32
lr = 5e-5
epochs = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

config = EngramConfig()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
compressor = VocabCompressor(tokenizer)
vocab_map = compressor.build_mapping().to(device)
model = GPT2WithEngram(config, vocab_map).to(device)
model.gpt2 = torch.compile(model.gpt2)
model.engram = torch.compile(model.engram)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_func(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

tokenized_datasets = dataset.map(tokenize_func, batched=True, remove_columns=["text"])

train_dataset = tokenized_datasets["train"]
valid_dataset = tokenized_datasets["validation"]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=data_collator)
valid_loader = DataLoader(valid_dataset, batch_size=bs, collate_fn=data_collator)

if not args.no_wandb:
    wandb.init(
        project="engram-gpt2-wikitext",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": bs,
            "ngram_orders": config.ngram_orders,
            "engram_dim": config.engram_dim,
            "bucket_size": config.bucket_size
        }
    )

for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, labels=labels, attention_mask=attn_mask)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        if not args.no_wandb:
            wandb.log({"train_loss": loss.item(), "epoch": epoch})

        if step % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} complete. Avg Train Loss: {avg_train_loss:.4f}")

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, labels=labels, attention_mask=attn_mask)
            valid_loss += outputs.loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    perplexity = torch.exp(torch.tensor(avg_valid_loss))

    print(f"Validation Loss: {avg_valid_loss:.4f} | Perplexity: {perplexity:.2f}")

    if not args.no_wandb:
        wandb.log({
            "valid_loss": avg_valid_loss,
            "perplexity": perplexity,
            "epoch": epoch + 1
        })

if not args.no_wandb:
    wandb.finish()

if args.push_to_hub:
    print(f"Pushing model to Hub: {args.repo_id}...")
    model.push_to_hub(args.repo_id)
    tokenizer.push_to_hub(args.repo_id)
    print("Push complete!")