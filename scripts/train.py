#!/usr/bin/env python3
"""
Training script for Adaptive Reasoning Chain Distillation.

Runs the full pipeline:
1. Load teacher and student models
2. Download/load MMLU dataset
3. Generate reasoning chains from teacher
4. Score question difficulty
5. Train student with curriculum learning
6. Save checkpoints and metrics

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --max-samples 1000  # debug run
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Ensure the project root is on the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from adaptive_reasoning_chain_distillation.models.teacher import TeacherModel
from adaptive_reasoning_chain_distillation.models.student import StudentModel
from adaptive_reasoning_chain_distillation.models.distiller import ReasoningDistiller
from adaptive_reasoning_chain_distillation.data.mmlu_loader import (
    MMLUReasoningDataset,
    DifficultyCalibrator,
)
from adaptive_reasoning_chain_distillation.training.trainer import DistillationTrainer


def setup_logging(verbose: bool = True) -> None:
    """Configure logging with timestamps and levels."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress noisy library loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_config(config_path: str) -> dict:
    """Load and return YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Adaptive Reasoning Chain Distillation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of dataset samples (for debugging)",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Override config with CLI args
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if args.no_fp16:
        config["training"]["fp16"] = False

    setup_logging(config.get("logging", {}).get("verbose", True))
    logger = logging.getLogger(__name__)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU detected -- training will be very slow")

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # =========================================================================
    # Step 1: Load models
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Loading teacher and student models")
    logger.info("=" * 60)

    teacher_cfg = config["model"]["teacher"]
    teacher = TeacherModel(
        model_name=teacher_cfg["name"],
        max_new_tokens=teacher_cfg["max_new_tokens"],
        temperature=teacher_cfg["temperature"],
        top_p=teacher_cfg["top_p"],
        device=device,
    )

    student = StudentModel(
        model_name=config["model"]["student"]["name"],
        device=device,
    )

    # =========================================================================
    # Step 2: Build distiller
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 2: Building distillation module")
    logger.info("=" * 60)

    loss_cfg = config["loss"]
    distiller = ReasoningDistiller(
        teacher=teacher,
        student=student,
        kl_temperature=loss_cfg["kl_temperature"],
        kl_weight=loss_cfg["kl_weight"],
        ce_weight=loss_cfg["ce_weight"],
    )

    # =========================================================================
    # Step 3: Load MMLU dataset
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 3: Loading MMLU dataset")
    logger.info("=" * 60)

    data_cfg = config["data"]
    # Use the teacher's tokenizer (shared vocab with student)
    tokenizer = teacher.tokenizer

    dataset = MMLUReasoningDataset(
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_seq_length"],
        split=data_cfg["split"],
        dataset_name=data_cfg["dataset_name"],
        cache_dir=data_cfg.get("cache_dir"),
        reasoning_cache_dir=data_cfg.get("reasoning_cache_dir"),
        max_samples=data_cfg.get("max_samples"),
    )

    # =========================================================================
    # Step 4: Generate reasoning chains
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 4: Generating reasoning chains from teacher")
    logger.info("=" * 60)

    chain_start = time.time()
    dataset.generate_reasoning_chains(teacher_model=teacher, batch_size=8)
    chain_time = time.time() - chain_start
    logger.info(f"Reasoning chain generation took {chain_time / 60:.1f} minutes")

    # =========================================================================
    # Step 5: Score difficulty
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 5: Scoring question difficulty")
    logger.info("=" * 60)

    calibrator = DifficultyCalibrator(num_bins=data_cfg.get("difficulty_bins", 10))
    difficulty_scores = calibrator.score_difficulties(
        teacher_model=teacher,
        tokenizer=tokenizer,
        prompts=dataset.prompts,
        batch_size=8,
        max_seq_length=config["model"]["max_seq_length"],
    )
    dataset.set_difficulty_scores(difficulty_scores)

    # =========================================================================
    # Step 6: Train with curriculum
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 6: Starting curriculum-based distillation training")
    logger.info("=" * 60)

    trainer = DistillationTrainer(
        distiller=distiller,
        dataset=dataset,
        calibrator=calibrator,
        config=config,
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        state_path = Path(args.resume_from) / "training_state.pt"
        if state_path.exists():
            logger.info(f"Resuming from {args.resume_from}")
            state = torch.load(state_path, map_location=device)
            trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
            trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
            if state.get("scaler_state_dict") and trainer.use_fp16:
                trainer.scaler.load_state_dict(state["scaler_state_dict"])
            trainer.global_step = state["global_step"]
            trainer.best_metric_value = state.get("best_metric_value", -float("inf"))
            logger.info(
                f"Resumed at step {trainer.global_step}, "
                f"best metric: {trainer.best_metric_value:.4f}"
            )
        else:
            logger.warning(f"No checkpoint found at {state_path}")

    results = trainer.train()

    # =========================================================================
    # Step 7: Final summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Total training time: {results['total_time_hours']:.2f} hours")
    logger.info(f"Best difficulty_calibration: {results['best_metric']:.4f}")

    for i, epoch_metrics in enumerate(results["epoch_metrics"]):
        logger.info(
            f"  Epoch {i + 1}: loss={epoch_metrics['train_loss']:.4f}, "
            f"kl={epoch_metrics['train_kl_loss']:.4f}, "
            f"ce={epoch_metrics['train_ce_loss']:.4f}, "
            f"diff_cal={epoch_metrics.get('difficulty_calibration', 0):.4f}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
