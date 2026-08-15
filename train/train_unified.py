#!/usr/bin/env python
"""
Main training script for Unified Image Restoration models.

Supports:
- Unified V1 and V2 architectures
- Config-based hyperparameter management
- Curriculum learning configuration
- Mixed precision configuration
- Checkpoint management
- Validation
- Early stopping configuration

Usage:
    python train/train_unified.py --config configs/train_v1.yaml --epochs 50
    python train/train_unified.py --config configs/train_v2.yaml --epochs 60
    python train/train_unified.py --config configs/train_v2.yaml --smoke-test
"""

import os
import sys
import argparse
import yaml
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm


# ============================================================
# PATH SETUP
# ============================================================

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from datasets import create_dataloaders, load_paired_images
from models import UnifiedRestorerV1, UnifiedRestorerV2
from models.losses import RestorationLoss


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    """Training orchestrator for restoration models."""

    def __init__(self, config: Dict, model_name: str = "v1"):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary loaded from YAML.
            model_name: "v1" or "v2".
        """

        self.config = config
        self.model_name = model_name.lower()

        if self.model_name not in {"v1", "v2"}:
            raise ValueError(
                f"Unsupported model '{self.model_name}'. "
                f"Expected 'v1' or 'v2'."
            )

        # --------------------------------------------------------
        # Device
        # --------------------------------------------------------

        hardware_cfg = config.get("hardware", {})

        configured_device = hardware_cfg.get("device")

        if configured_device:
            self.device = torch.device(configured_device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # --------------------------------------------------------
        # Reproducibility
        # --------------------------------------------------------

        self.seed = config.get(
            "reproducibility",
            {}
        ).get("seed", 42)

        self._set_seed(self.seed)

        # --------------------------------------------------------
        # Directories
        # --------------------------------------------------------

        self.checkpoint_dir = (
            PROJECT_ROOT / "checkpoints" / self.model_name
        )
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.results_dir = (
            PROJECT_ROOT / "results" / self.model_name
        )
        self.results_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------

        self.log_file = self.results_dir / "training.log"
        self.metrics_file = self.results_dir / "metrics.json"

        self.metrics = {
            "model": self.model_name,
            "train": [],
            "val": []
        }

        print(
            f"[{self._timestamp()}] "
            f"Trainer initialized for {self.model_name} "
            f"on {self.device}"
        )

        self._log(
            f"Trainer initialized for {self.model_name} "
            f"on {self.device}"
        )

    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================

    def _timestamp(self):
        """Get current timestamp."""

        return datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def _log(self, msg: str):
        """Log message to stdout and training log."""

        timestamp = self._timestamp()

        log_msg = (
            f"[{timestamp}] {msg}"
        )

        print(log_msg)

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as f:
            f.write(log_msg + "\n")

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""

        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ============================================================
    # MODEL
    # ============================================================

    def build_model(self) -> nn.Module:
        """Build model based on configuration."""

        model_cfg = self.config.get(
            "model",
            {}
        )

        # --------------------------------------------------------
        # V1
        # --------------------------------------------------------

        if self.model_name == "v1":

            num_features = model_cfg.get(
                "num_features",
                model_cfg.get("channels", 64)
            )

            num_blocks = model_cfg.get(
                "num_blocks",
                model_cfg.get("blocks", 12)
            )

            model = UnifiedRestorerV1(
                num_features=num_features,
                num_blocks=num_blocks
            )

        # --------------------------------------------------------
        # V2
        # --------------------------------------------------------

        elif self.model_name == "v2":

            num_features = model_cfg.get(
                "num_features",
                model_cfg.get("channels", 64)
            )

            num_blocks = model_cfg.get(
                "num_blocks",
                model_cfg.get("blocks", 16)
            )

            use_hf_branch = model_cfg.get(
                "use_hf_branch",
                model_cfg.get(
                    "use_high_frequency_branch",
                    True
                )
            )

            use_edge_branch = model_cfg.get(
                "use_edge_branch",
                True
            )

            hf_strength = model_cfg.get(
                "hf_strength",
                model_cfg.get(
                    "hf_branch_strength",
                    0.5
                )
            )

            edge_strength = model_cfg.get(
                "edge_strength",
                model_cfg.get(
                    "edge_branch_strength",
                    0.3
                )
            )

            model = UnifiedRestorerV2(
                num_features=num_features,
                num_blocks=num_blocks,
                use_hf_branch=use_hf_branch,
                use_edge_branch=use_edge_branch,
                hf_strength=hf_strength,
                edge_strength=edge_strength
            )

        else:
            raise ValueError(
                f"Unknown model: {self.model_name}"
            )

        # --------------------------------------------------------
        # Move model to device
        # --------------------------------------------------------

        model = model.to(self.device)

        params = sum(
            p.numel()
            for p in model.parameters()
        )

        self._log(
            f"Model {self.model_name}: "
            f"{params:,} parameters"
        )

        return model

    # ============================================================
    # LOSS
    # ============================================================

    def build_loss(self) -> nn.Module:
        """Build restoration loss from configuration."""

        loss_cfg = self.config.get(
            "loss",
            {}
        )

        loss_fn = RestorationLoss(
            use_charbonnier=loss_cfg.get(
                "use_charbonnier",
                True
            ),

            l1_weight=loss_cfg.get(
                "l1_weight",
                0.65
            ),

            mse_weight=loss_cfg.get(
                "mse_weight",
                0.15
            ),

            gradient_weight=loss_cfg.get(
                "gradient_weight",
                0.10
            ),

            frequency_weight=loss_cfg.get(
                "frequency_weight",
                0.05
            ),

            perceptual_weight=loss_cfg.get(
                "perceptual_weight",
                0.0
            )
        )

        return loss_fn

    # ============================================================
    # OPTIMIZER
    # ============================================================

    def build_optimizer(
        self,
        model: nn.Module
    ) -> Tuple[
        torch.optim.Optimizer,
        Optional[torch.optim.lr_scheduler._LRScheduler]
    ]:
        """Build optimizer and scheduler."""

        train_cfg = self.config.get(
            "training",
            {}
        )

        opt_type = train_cfg.get(
            "optimizer",
            "adamw"
        ).lower()

        lr = float(
            train_cfg.get(
                "learning_rate",
                1e-3
            )
        )

        weight_decay = float(
            train_cfg.get(
                "weight_decay",
                1e-5
            )
        )

        # --------------------------------------------------------
        # Optimizer
        # --------------------------------------------------------

        if opt_type == "adamw":

            optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        elif opt_type == "adam":

            optimizer = Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        else:

            raise ValueError(
                f"Unknown optimizer: {opt_type}"
            )

        # --------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------

        sched_type = train_cfg.get(
            "scheduler",
            "cosine"
        ).lower()

        scheduler = None

        if sched_type == "cosine":

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=train_cfg.get(
                    "epochs",
                    50
                ),
                eta_min=lr * 0.01
            )

        elif sched_type == "step":

            scheduler = StepLR(
                optimizer,
                step_size=train_cfg.get(
                    "step_size",
                    10
                ),
                gamma=0.5
            )

        elif sched_type in {
            "none",
            "null"
        }:

            scheduler = None

        else:

            raise ValueError(
                f"Unknown scheduler: {sched_type}"
            )

        self._log(
            f"Optimizer: {opt_type} "
            f"(lr={lr}, wd={weight_decay})"
        )

        self._log(
            f"Scheduler: {sched_type}"
        )

        return optimizer, scheduler

    # ============================================================
    # DATA
    # ============================================================

    def build_dataloaders(
        self
    ) -> Tuple[
        DataLoader,
        DataLoader,
        DataLoader
    ]:
        """Build train/validation/test dataloaders."""

        data_cfg = self.config.get(
            "dataset",
            {}
        )

        train_cfg = self.config.get(
            "training",
            {}
        )

        gt_dir = data_cfg.get(
            "gt_dir",
            "train/GT"
        )

        noisy_dir = data_cfg.get(
            "noisy_dir",
            "train/NoisyLR"
        )

        batch_size = int(
            train_cfg.get(
                "batch_size",
                16
            )
        )

        num_workers = int(
            train_cfg.get(
                "num_workers",
                0
            )
        )

        # Resolve paths relative to project root
        gt_path = PROJECT_ROOT / gt_dir
        noisy_path = PROJECT_ROOT / noisy_dir

        train_loader, val_loader, test_loader = (
            create_dataloaders(
                str(gt_path),
                str(noisy_path),
                batch_size=batch_size,
                num_workers=num_workers
            )
        )

        self._log(
            "Dataloaders created:"
        )

        self._log(
            f"  Train: "
            f"{len(train_loader)} batches "
            f"of {batch_size}"
        )

        self._log(
            f"  Val:   "
            f"{len(val_loader)} batches "
            f"of {batch_size}"
        )

        self._log(
            f"  Test:  "
            f"{len(test_loader)} batches "
            f"of {batch_size}"
        )

        return (
            train_loader,
            val_loader,
            test_loader
        )

    # ============================================================
    # TRAIN ONE EPOCH
    # ============================================================

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> float:
        """Train one epoch."""

        model.train()

        total_loss = 0.0

        train_cfg = self.config.get(
            "training",
            {}
        )

        use_amp = (
            train_cfg.get(
                "use_amp",
                False
            )
            and self.device.type == "cuda"
        )

        use_gradient_clipping = train_cfg.get(
            "use_gradient_clipping",
            False
        )

        max_grad_norm = float(
            train_cfg.get(
                "max_grad_norm",
                1.0
            )
        )

        scaler = getattr(
            self,
            "_scaler",
            None
        )

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} Train"
        )

        for batch_idx, batch in enumerate(pbar):

            # ----------------------------------------------------
            # Move data
            # ----------------------------------------------------

            noisy = batch["noisy"].to(
                self.device,
                non_blocking=True
            )

            gt = batch["gt"].to(
                self.device,
                non_blocking=True
            )

            # ----------------------------------------------------
            # Forward / backward
            # ----------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            if use_amp:

                with autocast():

                    pred = model(noisy)

                    loss = loss_fn(
                        pred,
                        gt
                    )

                scaler.scale(
                    loss
                ).backward()

                if use_gradient_clipping:

                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm
                    )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                pred = model(noisy)

                loss = loss_fn(
                    pred,
                    gt
                )

                loss.backward()

                if use_gradient_clipping:

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm
                    )

                optimizer.step()

            # ----------------------------------------------------
            # Accumulate
            # ----------------------------------------------------

            loss_value = float(
                loss.detach().item()
            )

            total_loss += loss_value

            pbar.set_postfix(
                {
                    "loss": f"{loss_value:.6f}"
                }
            )

        avg_loss = (
            total_loss /
            max(len(train_loader), 1)
        )

        return avg_loss

    # ============================================================
    # VALIDATION
    # ============================================================

    @torch.no_grad()
    def val_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        epoch: int
    ) -> float:
        """Validate one epoch."""

        model.eval()

        total_loss = 0.0

        pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1} Val"
        )

        for batch in pbar:

            noisy = batch["noisy"].to(
                self.device,
                non_blocking=True
            )

            gt = batch["gt"].to(
                self.device,
                non_blocking=True
            )

            pred = model(noisy)

            loss = loss_fn(
                pred,
                gt
            )

            loss_value = float(
                loss.detach().item()
            )

            total_loss += loss_value

            pbar.set_postfix(
                {
                    "loss": f"{loss_value:.6f}"
                }
            )

        avg_loss = (
            total_loss /
            max(len(val_loader), 1)
        )

        return avg_loss

    # ============================================================
    # CHECKPOINT
    # ============================================================

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save latest and best model checkpoints."""

        checkpoint = {
            "epoch": epoch,
            "model_name": self.model_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": float(val_loss),
            "config": self.config
        }

        # --------------------------------------------------------
        # Latest
        # --------------------------------------------------------

        latest_path = (
            self.checkpoint_dir /
            "latest.pth"
        )

        torch.save(
            checkpoint,
            latest_path
        )

        # --------------------------------------------------------
        # Best
        # --------------------------------------------------------

        if is_best:

            best_path = (
                self.checkpoint_dir /
                f"best_{self.model_name}.pth"
            )

            torch.save(
                checkpoint,
                best_path
            )

            self._log(
                f"Saved best checkpoint: "
                f"{best_path}"
            )

    # ============================================================
    # TRAINING
    # ============================================================

    def train(
        self,
        epochs: Optional[int] = None,
        resume_from: Optional[str] = None
    ):
        """Main training loop."""

        if epochs is None:

            epochs = self.config.get(
                "training",
                {}
            ).get(
                "epochs",
                50
            )

        epochs = int(epochs)

        self._log(
            f"Starting training: "
            f"{epochs} epochs, "
            f"model={self.model_name}"
        )

        start_time = time.time()

        # --------------------------------------------------------
        # Build components
        # --------------------------------------------------------

        model = self.build_model()

        loss_fn = self.build_loss()

        optimizer, scheduler = (
            self.build_optimizer(model)
        )

        train_loader, val_loader, test_loader = (
            self.build_dataloaders()
        )

        # --------------------------------------------------------
        # AMP
        # --------------------------------------------------------

        train_cfg = self.config.get(
            "training",
            {}
        )

        use_amp = (
            train_cfg.get(
                "use_amp",
                False
            )
            and self.device.type == "cuda"
        )

        self._scaler = GradScaler(
            enabled=use_amp
        )

        # --------------------------------------------------------
        # Resume
        # --------------------------------------------------------

        start_epoch = 0

        best_val_loss = float("inf")

        if resume_from:

            resume_path = Path(
                resume_from
            )

            if not resume_path.is_absolute():

                resume_path = (
                    PROJECT_ROOT /
                    resume_path
                )

            if resume_path.exists():

                self._log(
                    f"Resuming from checkpoint: "
                    f"{resume_path}"
                )

                checkpoint = torch.load(
                    resume_path,
                    map_location=self.device
                )

                model.load_state_dict(
                    checkpoint["model_state"]
                )

                optimizer.load_state_dict(
                    checkpoint["optimizer_state"]
                )

                start_epoch = (
                    int(
                        checkpoint.get(
                            "epoch",
                            -1
                        )
                    ) + 1
                )

                best_val_loss = float(
                    checkpoint.get(
                        "val_loss",
                        float("inf")
                    )
                )

                self._log(
                    f"Resumed at epoch "
                    f"{start_epoch}"
                )

            else:

                self._log(
                    f"WARNING: Resume checkpoint "
                    f"not found: {resume_path}"
                )

        # --------------------------------------------------------
        # Training loop
        # --------------------------------------------------------

        self._log(
            f"Training {self.model_name} "
            f"for {epochs} epochs"
        )

        try:

            for epoch in range(
                start_epoch,
                epochs
            ):

                # ------------------------------------------------
                # Train
                # ------------------------------------------------

                train_loss = self.train_epoch(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    epoch
                )

                # ------------------------------------------------
                # Validation
                # ------------------------------------------------

                val_loss = self.val_epoch(
                    model,
                    val_loader,
                    loss_fn,
                    epoch
                )

                # ------------------------------------------------
                # Scheduler
                # ------------------------------------------------

                if scheduler is not None:

                    scheduler.step()

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------

                self.metrics["train"].append(
                    float(train_loss)
                )

                self.metrics["val"].append(
                    float(val_loss)
                )

                # ------------------------------------------------
                # Best checkpoint
                # ------------------------------------------------

                is_best = (
                    val_loss <
                    best_val_loss
                )

                if is_best:

                    best_val_loss = (
                        float(val_loss)
                    )

                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    is_best=is_best
                )

                # ------------------------------------------------
                # Log
                # ------------------------------------------------

                self._log(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_loss:.6f}, "
                    f"val_loss={val_loss:.6f}, "
                    f"best_val_loss={best_val_loss:.6f}"
                )

                # ------------------------------------------------
                # Save metrics
                # ------------------------------------------------

                with open(
                    self.metrics_file,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        self.metrics,
                        f,
                        indent=2
                    )

        except KeyboardInterrupt:

            self._log(
                "Training interrupted by user"
            )

        # --------------------------------------------------------
        # Completion
        # --------------------------------------------------------

        elapsed = (
            time.time() -
            start_time
        )

        self._log(
            f"Training complete! "
            f"Elapsed: {elapsed / 3600:.2f}h, "
            f"Best val loss: "
            f"{best_val_loss:.6f}"
        )

        return model, best_val_loss


# ============================================================
# MODEL NAME DETECTION
# ============================================================

def detect_model_name(
    config: Dict,
    config_path: str
) -> str:
    """
    Determine model name.

    Priority:
    1. Explicit model.name in YAML
    2. Configuration filename
    3. Default to V1
    """

    model_cfg = config.get(
        "model",
        {}
    )

    # --------------------------------------------------------
    # Explicit model name
    # --------------------------------------------------------

    explicit_name = model_cfg.get(
        "name"
    )

    if explicit_name:

        model_name = str(
            explicit_name
        ).lower().strip()

        if model_name in {
            "v1",
            "v2"
        }:

            return model_name

    # --------------------------------------------------------
    # Infer from config filename
    # --------------------------------------------------------

    config_name = Path(
        config_path
    ).stem.lower()

    if "v2" in config_name:

        return "v2"

    if "v1" in config_name:

        return "v1"

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return "v1"


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Train Unified Image "
            "Restoration Models"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Number of epochs "
            "(overrides config)"
        )
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to checkpoint "
            "to resume from"
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run smoke test "
            "(2 epochs only)"
        )
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    config_path = Path(
        args.config
    )

    if not config_path.exists():

        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    if config is None:

        raise ValueError(
            f"Configuration file is empty: "
            f"{config_path}"
        )

    # ========================================================
    # DETERMINE MODEL
    # ========================================================

    model_name = detect_model_name(
        config,
        str(config_path)
    )

    print(
        "\n" +
        "=" * 70
    )

    print(
        f"CONFIG: {config_path}"
    )

    print(
        f"SELECTED MODEL: "
        f"Unified Restorer {model_name.upper()}"
    )

    print(
        "=" * 70 +
        "\n"
    )

    # ========================================================
    # EPOCHS
    # ========================================================

    if args.smoke_test:

        epochs = 2

        print(
            "\n" +
            "=" * 70
        )

        print(
            "SMOKE TEST MODE (2 epochs only)"
        )

        print(
            "=" * 70 +
            "\n"
        )

    else:

        epochs = (
            args.epochs
            if args.epochs is not None
            else config.get(
                "training",
                {}
            ).get(
                "epochs",
                50
            )
        )

    # ========================================================
    # CREATE TRAINER
    # ========================================================

    trainer = Trainer(
        config,
        model_name=model_name
    )

    # ========================================================
    # TRAIN
    # ========================================================

    model, best_val_loss = trainer.train(
        epochs=epochs,
        resume_from=args.resume
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" +
        "=" * 70
    )

    print(
        "TRAINING SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Model:             {model_name}"
    )

    print(
        f"Epochs:            {epochs}"
    )

    print(
        f"Best Val Loss:     "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Checkpoint Dir:    "
        f"{trainer.checkpoint_dir}"
    )

    print(
        f"Results Dir:       "
        f"{trainer.results_dir}"
    )

    print(
        f"Log File:          "
        f"{trainer.log_file}"
    )

    print(
        f"Metrics File:      "
        f"{trainer.metrics_file}"
    )

    print(
        "=" * 70 +
        "\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()