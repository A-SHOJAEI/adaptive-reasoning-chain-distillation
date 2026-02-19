"""
Training loop with curriculum learning for reasoning chain distillation.

Implements a standard PyTorch training loop enhanced with:
- Adaptive curriculum sampling (easy -> hard progression)
- Gradient accumulation for effective larger batch sizes
- Mixed precision training (FP16) for memory efficiency
- Periodic evaluation and checkpoint saving
- Comprehensive metric logging
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ..models.distiller import ReasoningDistiller
from ..data.mmlu_loader import MMLUReasoningDataset, CurriculumSampler, DifficultyCalibrator
from ..evaluation.metrics import MetricsComputer

logger = logging.getLogger(__name__)


def _get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine learning rate schedule with linear warmup.

    During warmup: LR increases linearly from 0 to base LR.
    After warmup: LR follows a cosine decay to 0.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class DistillationTrainer:
    """
    Manages the end-to-end training process for reasoning chain distillation.

    Handles:
    - Optimizer and scheduler setup
    - Curriculum-aware data loading
    - Training loop with gradient accumulation
    - Mixed precision training
    - Evaluation and checkpointing
    - Metric logging
    """

    def __init__(
        self,
        distiller: ReasoningDistiller,
        dataset: MMLUReasoningDataset,
        calibrator: DifficultyCalibrator,
        config: dict,
    ) -> None:
        self.distiller = distiller
        self.dataset = dataset
        self.calibrator = calibrator
        self.config = config
        self.device = distiller.student.device

        # Training hyperparameters
        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.batch_size = train_cfg["batch_size"]
        self.grad_accum_steps = train_cfg["gradient_accumulation_steps"]
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.use_fp16 = train_cfg.get("fp16", True) and torch.cuda.is_available()

        # Curriculum sampler
        curriculum_cfg = config["curriculum"]
        self.sampler = CurriculumSampler(
            calibrator=calibrator,
            initial_fraction=curriculum_cfg["initial_fraction"],
            final_fraction=curriculum_cfg["final_fraction"],
            strategy=curriculum_cfg["strategy"],
            total_epochs=self.epochs,
            warmup_epochs=curriculum_cfg["warmup_epochs"],
        )

        # Optimizer (only student parameters)
        self.optimizer = AdamW(
            distiller.student.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Estimate total training steps for scheduler
        # (conservative: use full dataset size; curriculum may use fewer)
        steps_per_epoch = max(1, len(dataset) // (self.batch_size * self.grad_accum_steps))
        self.total_steps = steps_per_epoch * self.epochs
        warmup_steps = int(self.total_steps * train_cfg.get("warmup_ratio", 0.06))

        self.scheduler = _get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

        # Mixed precision scaler
        self.scaler = GradScaler("cuda", enabled=self.use_fp16)

        # Checkpointing
        ckpt_cfg = config["checkpoint"]
        self.checkpoint_dir = Path(ckpt_cfg["dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_epoch = ckpt_cfg.get("save_every_epoch", True)
        self.save_best = ckpt_cfg.get("save_best", True)
        self.best_metric_name = ckpt_cfg.get("metric", "difficulty_calibration")
        self.best_metric_value = -float("inf")

        # Evaluation
        eval_cfg = config["evaluation"]
        self.eval_every_steps = eval_cfg.get("eval_every_steps", 500)
        self.num_eval_samples = eval_cfg.get("num_eval_samples", 1000)

        # Logging
        log_cfg = config["logging"]
        self.log_every_steps = log_cfg.get("log_every_steps", 50)

        # Metrics computer
        self.metrics_computer = MetricsComputer()

        # State tracking
        self.global_step = 0
        self.epoch_losses: list[float] = []

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Dictionary of final training metrics.
        """
        logger.info("=" * 70)
        logger.info("Starting Reasoning Chain Distillation Training")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch size: {self.batch_size} (effective: {self.batch_size * self.grad_accum_steps})")
        logger.info(f"  Total steps (est.): {self.total_steps}")
        logger.info(f"  FP16: {self.use_fp16}")
        logger.info(f"  Device: {self.device}")
        logger.info("=" * 70)

        training_start = time.time()
        all_metrics: list[dict] = []

        for epoch in range(self.epochs):
            epoch_metrics = self._train_epoch(epoch)
            all_metrics.append(epoch_metrics)

            # Checkpoint
            if self.save_every_epoch:
                self._save_checkpoint(epoch, epoch_metrics)

            # Best model tracking
            if self.save_best and self.best_metric_name in epoch_metrics:
                current_val = epoch_metrics[self.best_metric_name]
                if current_val > self.best_metric_value:
                    self.best_metric_value = current_val
                    self._save_checkpoint(epoch, epoch_metrics, is_best=True)
                    logger.info(
                        f"New best {self.best_metric_name}: {current_val:.4f}"
                    )

        total_time = time.time() - training_start
        logger.info("=" * 70)
        logger.info(f"Training complete in {total_time / 3600:.2f} hours")
        logger.info(f"Best {self.best_metric_name}: {self.best_metric_value:.4f}")
        logger.info("=" * 70)

        return {
            "total_time_hours": total_time / 3600,
            "best_metric": self.best_metric_value,
            "epoch_metrics": all_metrics,
        }

    def _train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch with curriculum sampling.

        Args:
            epoch: Current epoch index.

        Returns:
            Dictionary of epoch-level metrics.
        """
        self.sampler.set_epoch(epoch)

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=min(self.config["data"].get("num_workers", 4), 4),
            pin_memory=True,
            drop_last=True,
        )

        self.distiller.student.model.train()
        self.distiller.teacher.model.eval()

        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_ce_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{self.epochs}",
            leave=True,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            loss, kl_loss, ce_loss = self._training_step(batch)

            # Accumulate for logging
            epoch_loss += loss
            epoch_kl_loss += kl_loss
            epoch_ce_loss += ce_loss
            num_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.distiller.student.model.parameters(),
                    self.max_grad_norm,
                )

                if self.use_fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.log_every_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_kl = epoch_kl_loss / num_batches
                    avg_ce = epoch_ce_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "kl": f"{avg_kl:.4f}",
                        "ce": f"{avg_ce:.4f}",
                        "lr": f"{lr:.2e}",
                        "step": self.global_step,
                    })

                # Periodic evaluation
                if self.global_step % self.eval_every_steps == 0:
                    eval_metrics = self._evaluate()
                    logger.info(
                        f"  [Step {self.global_step}] Eval: {eval_metrics}"
                    )
                    self.distiller.student.model.train()

        # End-of-epoch evaluation
        eval_metrics = self._evaluate()
        avg_loss = epoch_loss / max(num_batches, 1)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_kl_loss": epoch_kl_loss / max(num_batches, 1),
            "train_ce_loss": epoch_ce_loss / max(num_batches, 1),
            "curriculum_samples": len(self.sampler),
            **eval_metrics,
        }

        logger.info(
            f"Epoch {epoch + 1}/{self.epochs} complete | "
            f"Loss: {avg_loss:.4f} | "
            f"Difficulty cal.: {eval_metrics.get('difficulty_calibration', 0):.4f} | "
            f"Samples: {len(self.sampler)}"
        )

        return epoch_metrics

    def _training_step(self, batch: dict) -> tuple[float, float, float]:
        """
        Execute a single forward/backward pass.

        Args:
            batch: Dictionary from the DataLoader.

        Returns:
            Tuple of (total_loss, kl_loss, ce_loss) as Python floats.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        with autocast("cuda", enabled=self.use_fp16):
            output = self.distiller(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # Scale loss by gradient accumulation steps
            scaled_loss = output.loss / self.grad_accum_steps

        if self.use_fp16:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return (
            output.loss.item(),
            output.kl_loss.item(),
            output.ce_loss.item(),
        )

    @torch.no_grad()
    def _evaluate(self) -> dict:
        """
        Run evaluation on a subset of the dataset.

        Computes difficulty calibration (Spearman correlation between
        predicted and actual difficulty) and other metrics.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.distiller.student.model.eval()

        # Use a fixed subset for evaluation
        eval_size = min(self.num_eval_samples, len(self.dataset))
        eval_indices = list(range(eval_size))

        eval_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size * 2,  # Can use larger batch for eval
            sampler=eval_indices,
            num_workers=0,
            pin_memory=True,
        )

        all_student_losses = []
        all_difficulties = []
        total_eval_loss = 0.0
        num_eval_batches = 0

        for batch in eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            difficulties = batch["difficulty"]

            with autocast("cuda", enabled=self.use_fp16):
                output = self.distiller(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_eval_loss += output.loss.item()
            num_eval_batches += 1

            # Per-sample loss for difficulty calibration
            # We approximate per-sample loss using the CE component
            student_logits = output.student_logits
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, :student_logits.size(1)][:, 1:].contiguous()

            for i in range(input_ids.size(0)):
                sample_logits = shift_logits[i]
                sample_labels = shift_labels[i]
                valid_mask = sample_labels != -100

                if valid_mask.sum() > 0:
                    sample_loss = torch.nn.functional.cross_entropy(
                        sample_logits[valid_mask],
                        sample_labels[valid_mask],
                        reduction="mean",
                    ).item()
                else:
                    sample_loss = 0.0

                all_student_losses.append(sample_loss)
                all_difficulties.append(difficulties[i].item())

        # Compute metrics
        metrics = self.metrics_computer.compute_all(
            student_losses=np.array(all_student_losses),
            difficulty_scores=np.array(all_difficulties),
        )
        metrics["eval_loss"] = total_eval_loss / max(num_eval_batches, 1)

        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict,
        is_best: bool = False,
    ) -> None:
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch.
            metrics: Metrics dictionary to include in checkpoint.
            is_best: If True, save as the best model.
        """
        if is_best:
            save_dir = self.checkpoint_dir / "best"
        else:
            save_dir = self.checkpoint_dir / f"epoch_{epoch}"

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save student model
        self.distiller.student.save_pretrained(str(save_dir / "student_model"))

        # Save training state
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_fp16 else None,
            "metrics": metrics,
            "best_metric_value": self.best_metric_value,
        }
        torch.save(state, save_dir / "training_state.pt")

        tag = " (best)" if is_best else ""
        logger.info(f"Checkpoint saved: {save_dir}{tag}")
