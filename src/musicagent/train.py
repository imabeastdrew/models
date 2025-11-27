import argparse
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from musicagent.config import DataConfig, ModelConfig
from musicagent.dataset import MusicAgentDataset, collate_fn
from musicagent.model import OfflineTransformer
from musicagent.utils import seed_everything, setup_logging

logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return LambdaLR(optimizer, lr_lambda)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(
    model, loader, optimizer, scheduler, criterion, device, epoch,
    global_step, use_wandb=True, log_interval=100
):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for i, batch in enumerate(loader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        tgt_input = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)

        loss = criterion(output.reshape(-1, output.size(-1)), tgt_y.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        global_step += 1

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = elapsed * 1000 / log_interval

            logger.info(
                f"| epoch {epoch} | {i}/{len(loader)} batches | "
                f"ms/batch {ms_per_batch:.2f} | "
                f"lr {lr:.6f} | loss {cur_loss:.4f}"
            )

            if use_wandb:
                wandb.log({
                    "train/loss": cur_loss,
                    "train/lr": lr,
                    "train/ms_per_batch": ms_per_batch,
                    "train/epoch": epoch,
                }, step=global_step)

            total_loss = 0.0
            start_time = time.time()

    return global_step


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_y.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train Offline Model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="musicagent")
    parser.add_argument("--run-name", type=str, default=None, help="Custom wandb run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training (may be slower and restrict some CUDA kernels).",
    )
    # Data hyperparameters
    parser.add_argument("--max-len", type=int, help="Maximum input length for the model.")
    parser.add_argument(
        "--storage-len",
        type=int,
        help="Sequence length used on disk (must be >= max-len).",
    )
    parser.add_argument(
        "--max-transpose",
        type=int,
        help="Maximum semitone shift for random transposition augmentation.",
    )
    # Model / training hyperparameters
    parser.add_argument("--d-model", type=int, help="Transformer model dimension.")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads.")
    parser.add_argument("--n-layers", type=int, help="Number of Transformer layers.")
    parser.add_argument("--dropout", type=float, help="Dropout rate.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        help="Number of warmup steps for the learning-rate scheduler.",
    )
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed, deterministic=args.deterministic)
    args.save_dir.mkdir(exist_ok=True)

    d_cfg = DataConfig()
    m_cfg = ModelConfig()

    # Apply CLI overrides to configs (only when explicitly provided).
    if args.max_len is not None:
        d_cfg.max_len = args.max_len
    if args.storage_len is not None:
        d_cfg.storage_len = args.storage_len
    if args.max_transpose is not None:
        d_cfg.max_transpose = args.max_transpose

    if args.d_model is not None:
        m_cfg.d_model = args.d_model
    if args.n_heads is not None:
        m_cfg.n_heads = args.n_heads
    if args.n_layers is not None:
        m_cfg.n_layers = args.n_layers
    if args.dropout is not None:
        m_cfg.dropout = args.dropout
    if args.batch_size is not None:
        m_cfg.batch_size = args.batch_size
    if args.lr is not None:
        m_cfg.lr = args.lr
    if args.warmup_steps is not None:
        m_cfg.warmup_steps = args.warmup_steps

    device = torch.device(m_cfg.device)
    logger.info(f"Using device: {device}")

    try:
        train_ds = MusicAgentDataset(d_cfg, split='train')
        valid_ds = MusicAgentDataset(d_cfg, split='valid')
    except FileNotFoundError as e:
        logger.error(e)
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=m_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=m_cfg.batch_size,
        collate_fn=collate_fn,
    )

    vocab_src_len = len(train_ds.vocab_melody)
    vocab_tgt_len = len(train_ds.vocab_chord)
    logger.info(f"Vocab Size: Src={vocab_src_len}, Tgt={vocab_tgt_len}")

    model = OfflineTransformer(m_cfg, d_cfg, vocab_src_len, vocab_tgt_len).to(device)
    total_params = count_parameters(model)
    logger.info(f"Model Parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=m_cfg.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=d_cfg.pad_id)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, m_cfg.warmup_steps, total_steps)

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.run_name or (
            f"d{m_cfg.d_model}_h{m_cfg.n_heads}_L{m_cfg.n_layers}_bs{m_cfg.batch_size}"
        )

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                # Model config
                **asdict(m_cfg),
                # Data config (excluding paths)
                "max_len": d_cfg.max_len,
                "frame_rate": d_cfg.frame_rate,
                "storage_len": d_cfg.storage_len,
                # Training args
                "epochs": args.epochs,
                "weight_decay": 0.01,
                "grad_clip": 0.5,
                # Derived
                "vocab_src_size": vocab_src_len,
                "vocab_tgt_size": vocab_tgt_len,
                "total_params": total_params,
                "train_samples": len(train_ds),
                "valid_samples": len(valid_ds),
                "total_steps": total_steps,
            },
            tags=["transformer", "melody-chord", "realchords"],
        )

        # Define summary metrics
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("valid/loss", summary="min")
        wandb.define_metric("valid/perplexity", summary="min")

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        global_step = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, global_step, use_wandb
        )
        val_loss = evaluate(model, valid_loader, criterion, device)
        val_ppl = math.exp(val_loss)

        logger.info("-" * 89)
        logger.info(
            f"| End of epoch {epoch} | valid loss {val_loss:.4f} | "
            f"perplexity {val_ppl:.2f}"
        )
        logger.info("-" * 89)

        if use_wandb:
            wandb.log({
                "valid/loss": val_loss,
                "valid/perplexity": val_ppl,
                "valid/epoch": epoch,
            }, step=global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = args.save_dir / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("Saved best model.")

            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_val_perplexity"] = math.exp(best_val_loss)
                wandb.run.summary["best_epoch"] = epoch

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
