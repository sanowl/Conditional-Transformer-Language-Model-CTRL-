import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CTRLConfig, CTRLModel, CTRLTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

class CTRL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ctrl = CTRLModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.ctrl(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(outputs.last_hidden_state)
        return logits

    def generate(self, input_ids, max_length, temperature=1.0, top_k=50, top_p=0.95):
        for _ in range(max_length - input_ids.shape[1]):
            outputs = self(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

class CTRLDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def train(args, model, train_dataset, val_dataset, tokenizer, device):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_loader) * args.epochs
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), ignore_index=tokenizer.pad_token_id)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, tokenizer, device)
        
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

def evaluate(model, val_loader, tokenizer, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), ignore_index=tokenizer.pad_token_id)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def generate_text(model, tokenizer, prompt, max_length, temperature, top_k, top_p, device):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Train a CTRL model")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset to use for training")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for saving model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_dataset(args.dataset, args.dataset_config)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Initialize tokenizer and config
    tokenizer = CTRLTokenizer.from_pretrained("ctrl")
    config = CTRLConfig(vocab_size=tokenizer.vocab_size)

    # Create model
    model = CTRL(config).to(device)

    # Prepare datasets
    train_dataset = CTRLDataset(train_dataset["text"], tokenizer, args.max_length)
    val_dataset = CTRLDataset(val_dataset["text"], tokenizer, args.max_length)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train model
    train(args, model, train_dataset, val_dataset, tokenizer, device)

    # Generate sample text
    prompt = "Once upon a time"
    generated_text = generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95, device=device)
    logging.info(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main()