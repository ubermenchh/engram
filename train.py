import argparse

import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from compression import VocabCompressor
from config import EngramConfig
from model import GPT2WithEngram

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument("--push-to-hub", action="store_true", help="Push model to Hugging Face Hub after training")
parser.add_argument("--repo-id", type=str, default="engram-gpt2-wikitext", help="Hugging Face repository ID")
parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
args = parser.parse_args()

bs = 64
lr = 5e-5
epochs = 1
val_iters = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
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
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=data_collator, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=bs, collate_fn=data_collator, num_workers=4, pin_memory=True)

config = EngramConfig()
compressor = VocabCompressor(tokenizer)
vocab_map = compressor.build_mapping().to(device)
model = GPT2WithEngram(config, vocab_map).to(device)
model.gpt2 = torch.compile(model.gpt2)
model.engram = torch.compile(model.engram)

# Freezing the base GPT2
# for param in model.gpt2.parameters():
#     param.requires_grad = False

engram_params =[p for n, p in model.named_parameters() if "engram" in n]
gpt2_params = [p for n, p in model.named_parameters() if "engram" not in n]

optimizer = torch.optim.AdamW([
    {
        "params": gpt2_params,
        "lr": lr,
        "weight_decay": 0.01
    },
    {
        "params": engram_params,
        "lr": lr * 5,
        "weight_decay": 0.0
    }
], fused=True)

num_training_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps
)

if not args.no_wandb:
    wandb.login()
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

def evaluate(model, valid_loader, device, limit=None):
    model.eval()
    valid_loss = 0
    steps_run = 0

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            if limit is not None and i >= limit:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)

            outputs = model(input_ids, labels=labels, attention_mask=attn_mask)
            valid_loss += outputs.loss.item()
            steps_run += 1

    if steps_run == 0: return 0.0, 0.0

    avg_valid_loss = valid_loss / steps_run
    perplexity = torch.exp(torch.tensor(avg_valid_loss))
    return avg_valid_loss, perplexity

for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, labels=labels, attention_mask=attn_mask)
            loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if not args.no_wandb:
            wandb.log({"train_loss": loss.item(), "epoch": epoch})

        if step > 0 and step % val_iters == 0:
            valid_loss, valid_ppl = evaluate(model, valid_loader, device, limit=200)

            print(f"\nStep {step}: Valid Loss: {valid_loss:.4f} | Perplexity: {valid_ppl:.2f}")

            if not args.no_wandb:
                wandb.log({"valid_loss": valid_loss, "perplexity": valid_ppl, "step": step})
            
            model.train()

        if step % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} complete. Avg Train Loss: {avg_train_loss:.4f}")

    avg_valid_loss, perplexity = evaluate(model, valid_loader, device, limit=None)
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