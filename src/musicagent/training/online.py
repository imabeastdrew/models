"""Script for online model."""

import argparse
import json
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

from musicagent.config import DataConfig, OnlineConfig
from musicagent.data.online import OnlineDataset, make_online_collate_fn
from musicagent.models import OnlineTransformer
from musicagent.utils import seed_everything, setup_logging

logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create a linear learning rate schedule with warmup."""
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
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    global_step,
    use_wandb: bool = True,
    log_interval: int = 100,
):
    """Train for one epoch.

    For the online model, the DataLoader yields interleaved sequences
    ``[SOS, y₁, x₁, y₂, x₂, ...]`` from :class:`OnlineDataset`. We perform
    next‑token prediction but only train on chord tokens ``y_t``; loss
    contributions from melody tokens ``x_t`` are masked out.
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for i, input_ids in enumerate(loader):
        input_ids = input_ids.to(device)

        # Standard next-token prediction over the interleaved sequence:
        #
        #   full sequence: [SOS, y₁, x₁, ..., yₙ, xₙ]
        #   inputs       = [SOS, y₁, x₁, ..., yₙ]       (drops last token)
        #   targets      = [y₁,  x₁, y₂, ..., yₙ, xₙ]  (shifted left by 1)
        #
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:].clone()

        # Mask out melody positions in the target so we only optimize chords.
        # In `targets`, chord tokens live at even time indices (0, 2, 4, ...),
        # and melody tokens at odd indices (1, 3, 5, ...).
        pad_id = model.pad_id
        targets[:, 1::2] = pad_id

        optimizer.zero_grad()
        output = model(inputs)

        # Flatten for loss computation
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            targets.reshape(-1)
        )
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
    """Evaluate model on validation/test set.

    Note:
        The returned loss is the *mean of per-batch cross-entropies*, i.e.
        effectively averaged per sample, not strictly normalized by the total
        number of (non-ignored) tokens. This is sufficient for tracking
        training progress but is not a calibrated per-token NLL.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids in loader:
            input_ids = input_ids.to(device)

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:].clone()

            # Apply the same chord-only masking used in training.
            pad_id = model.pad_id
            targets[:, 1::2] = pad_id

            output = model(inputs)

            flat_output = output.reshape(-1, output.size(-1))
            flat_targets = targets.reshape(-1)

            loss = criterion(flat_output, flat_targets)

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_tokens += batch_size

    if total_tokens == 0:
        return 0.0

    return total_loss / total_tokens


def train_online(args: argparse.Namespace) -> None:
    """Main training function for online model (MLE pretraining)."""
    setup_logging()
    seed_everything(args.seed, deterministic=args.deterministic)
    args.save_dir.mkdir(exist_ok=True, parents=True)

    d_cfg = DataConfig()
    m_cfg = OnlineConfig()

    # Apply CLI overrides
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

    # Validate configuration
    if m_cfg.d_model % m_cfg.n_heads != 0:
        raise ValueError(
            f"d_model ({m_cfg.d_model}) must be divisible by n_heads ({m_cfg.n_heads})"
        )

    if args.device:
        m_cfg.device = args.device

    device = torch.device(m_cfg.device)
    logger.info(f"Using device: {device}")

    try:
        train_ds = OnlineDataset(d_cfg, split='train')
        valid_ds = OnlineDataset(d_cfg, split='valid')
    except FileNotFoundError as e:
        logger.error(e)
        return

    collate_fn = make_online_collate_fn(pad_id=d_cfg.pad_id)
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

    # Get vocab sizes from dataset
    melody_vocab_size = train_ds.melody_vocab_size
    chord_vocab_size = train_ds.chord_vocab_size
    unified_vocab_size = train_ds.unified_vocab_size
    logger.info(
        f"Vocab Size: Melody={melody_vocab_size}, Chord={chord_vocab_size}, "
        f"Unified={unified_vocab_size}"
    )

    model = OnlineTransformer(
        m_cfg, d_cfg, melody_vocab_size, chord_vocab_size
    ).to(device)
    total_params = count_parameters(model)
    logger.info(f"Model Parameters: {total_params:,}")

    # Optionally resume from checkpoint
    if args.resume_from is not None:
        if args.resume_from.is_file():
            state_dict = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded checkpoint from {args.resume_from}")
        else:
            logger.warning(
                f"--resume-from was set to {args.resume_from}, but file does not exist. "
                "Continuing with randomly initialized weights."
            )

    optimizer = optim.AdamW(model.parameters(), lr=m_cfg.lr, weight_decay=0.01)

    # Use unified vocab's pad_id for loss masking
    # In the unified vocab, pad_id is the same as melody's pad_id (0)
    criterion = nn.CrossEntropyLoss(ignore_index=d_cfg.pad_id)

    # Calculate total steps from epochs
    total_steps = len(train_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, m_cfg.warmup_steps, total_steps)

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.run_name or (
            f"online_d{m_cfg.d_model}_h{m_cfg.n_heads}_L{m_cfg.n_layers}_bs{m_cfg.batch_size}"
        )

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_type": "online",
                **asdict(m_cfg),
                "max_len": d_cfg.max_len,
                "frame_rate": d_cfg.frame_rate,
                "storage_len": d_cfg.storage_len,
                "epochs": args.epochs,
                "weight_decay": 0.01,
                "grad_clip": 0.5,
                "melody_vocab_size": melody_vocab_size,
                "chord_vocab_size": chord_vocab_size,
                "unified_vocab_size": unified_vocab_size,
                "total_params": total_params,
                "train_samples": len(train_ds),
                "valid_samples": len(valid_ds),
                "total_steps": total_steps,
            },
            tags=["transformer", "online", "melody-chord", "realchords"],
        )

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
        try:
            val_ppl = math.exp(min(val_loss, 100))
        except OverflowError:
            val_ppl = float('inf')

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

            # Persist the effective data/model configs alongside the checkpoint
            data_cfg_path = args.save_dir / "data_config.json"
            model_cfg_path = args.save_dir / "model_config.json"
            with data_cfg_path.open("w") as f:
                json.dump(asdict(d_cfg), f, default=str, indent=2)
            with model_cfg_path.open("w") as f:
                json.dump(asdict(m_cfg), f, default=str, indent=2)
            logger.info("Saved data/model configs.")

            if use_wandb and wandb.run is not None:
                wandb.run.summary["best_val_loss"] = best_val_loss
                try:
                    wandb.run.summary["best_val_perplexity"] = math.exp(min(best_val_loss, 100))
                except OverflowError:
                    wandb.run.summary["best_val_perplexity"] = float('inf')
                wandb.run.summary["best_epoch"] = epoch

                artifact = wandb.Artifact(
                    name="best-online-model",
                    type="model",
                    metadata={
                        "model_type": "online",
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl,
                    },
                )
                artifact.add_file(str(checkpoint_path))
                wandb.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])

    if use_wandb:
        wandb.finish()


def main():
    """CLI entry point for online training."""
    parser = argparse.ArgumentParser(description="Train Online Model (MLE Pretraining)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/online"))
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="musicagent")
    parser.add_argument("--run-name", type=str, default=None, help="Custom wandb run name")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a model checkpoint (.pt) to warm-start from.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training.",
    )
    # Data hyperparameters
    parser.add_argument("--max-len", type=int, help="Maximum input length.")
    parser.add_argument("--storage-len", type=int, help="Sequence length on disk.")
    parser.add_argument("--max-transpose", type=int, help="Max semitone shift.")
    # Model / training hyperparameters
    parser.add_argument("--d-model", type=int, help="Model dimension.")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads.")
    parser.add_argument("--n-layers", type=int, help="Number of layers.")
    parser.add_argument("--dropout", type=float, help="Dropout rate.")
    parser.add_argument("--batch-size", type=int, help="Batch size.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--warmup-steps", type=int, help="Warmup steps.")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu).")

    args = parser.parse_args()
    train_online(args)


if __name__ == "__main__":
    main()

