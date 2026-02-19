"""
MMLU dataset loader with reasoning chain generation and difficulty scoring.

Loads the MMLU dataset from HuggingFace, generates chain-of-thought reasoning
using the teacher model, scores question difficulty based on teacher entropy,
and provides a curriculum sampler for progressive difficulty training.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

CHOICE_LABELS = ["A", "B", "C", "D"]


class DifficultyCalibrator:
    """
    Scores and bins MMLU questions by difficulty using teacher model entropy.

    Higher teacher entropy on a question indicates greater uncertainty, which
    correlates with question difficulty. Questions are sorted by entropy and
    assigned to difficulty bins for curriculum learning.
    """

    def __init__(self, num_bins: int = 10) -> None:
        self.num_bins = num_bins
        self.entropies: list[float] = []
        self.difficulty_bins: Optional[np.ndarray] = None
        self.sorted_indices: Optional[np.ndarray] = None

    def score_difficulties(
        self,
        teacher_model: "TeacherModel",
        tokenizer: PreTrainedTokenizer,
        prompts: list[str],
        batch_size: int = 8,
        max_seq_length: int = 512,
    ) -> np.ndarray:
        """
        Compute difficulty scores for a list of prompts using teacher entropy.

        Args:
            teacher_model: Frozen teacher model.
            tokenizer: Shared tokenizer.
            prompts: List of formatted question prompts.
            batch_size: Batch size for entropy computation.
            max_seq_length: Max tokens per prompt.

        Returns:
            Array of entropy scores, one per prompt.
        """
        logger.info(f"Scoring difficulty for {len(prompts)} questions...")
        all_entropies = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Scoring difficulty"):
            batch_prompts = prompts[i : i + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            entropies = teacher_model.compute_entropy(
                encoded["input_ids"],
                encoded["attention_mask"],
            )
            all_entropies.extend(entropies.cpu().tolist())

        self.entropies = all_entropies
        entropy_array = np.array(all_entropies)

        # Sort by entropy (ascending = easy to hard)
        self.sorted_indices = np.argsort(entropy_array)

        # Assign bin labels
        self.difficulty_bins = np.zeros(len(entropy_array), dtype=np.int64)
        bin_size = len(entropy_array) // self.num_bins
        for b in range(self.num_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < self.num_bins - 1 else len(entropy_array)
            indices_in_bin = self.sorted_indices[start:end]
            self.difficulty_bins[indices_in_bin] = b

        logger.info(
            f"Difficulty scoring complete. Entropy range: "
            f"[{entropy_array.min():.3f}, {entropy_array.max():.3f}], "
            f"mean: {entropy_array.mean():.3f}"
        )
        return entropy_array

    def get_curriculum_indices(self, fraction: float) -> np.ndarray:
        """
        Get indices of samples up to a given difficulty fraction.

        Args:
            fraction: Fraction of total samples (0.0 to 1.0), ordered by
                      difficulty from easiest to hardest.

        Returns:
            Array of sample indices, sorted easiest-first.
        """
        if self.sorted_indices is None:
            raise RuntimeError("Must call score_difficulties() before get_curriculum_indices()")

        n_samples = max(1, int(len(self.sorted_indices) * fraction))
        return self.sorted_indices[:n_samples].copy()


class CurriculumSampler(Sampler):
    """
    A PyTorch Sampler that yields indices according to curriculum difficulty.

    At each epoch, the scheduler determines what fraction of the dataset
    (by difficulty) should be included. Only those indices are sampled,
    in shuffled order within the allowed difficulty range.
    """

    def __init__(
        self,
        calibrator: DifficultyCalibrator,
        initial_fraction: float = 0.3,
        final_fraction: float = 1.0,
        strategy: str = "linear",
        total_epochs: int = 10,
        warmup_epochs: int = 1,
    ) -> None:
        self.calibrator = calibrator
        self.initial_fraction = initial_fraction
        self.final_fraction = final_fraction
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0
        self._current_indices: Optional[np.ndarray] = None

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for curriculum progression."""
        self._current_epoch = epoch
        fraction = self._compute_fraction(epoch)
        self._current_indices = self.calibrator.get_curriculum_indices(fraction)
        logger.info(
            f"Epoch {epoch}: curriculum fraction={fraction:.3f}, "
            f"samples={len(self._current_indices)}"
        )

    def _compute_fraction(self, epoch: int) -> float:
        """
        Compute the dataset fraction to include at a given epoch.

        During warmup, only the initial fraction is used. After warmup,
        the fraction increases according to the chosen strategy.
        """
        if epoch < self.warmup_epochs:
            return self.initial_fraction

        # Progress from 0 to 1 over remaining epochs
        remaining = self.total_epochs - self.warmup_epochs
        if remaining <= 0:
            return self.final_fraction

        progress = min(1.0, (epoch - self.warmup_epochs) / remaining)

        if self.strategy == "linear":
            fraction = self.initial_fraction + progress * (
                self.final_fraction - self.initial_fraction
            )
        elif self.strategy == "exponential":
            # Exponential ramp: slow start, fast finish
            fraction = self.initial_fraction + (
                (self.final_fraction - self.initial_fraction) * (progress ** 2)
            )
        elif self.strategy == "step":
            # Step function: jump at each third of training
            if progress < 0.33:
                fraction = self.initial_fraction
            elif progress < 0.66:
                fraction = self.initial_fraction + 0.5 * (
                    self.final_fraction - self.initial_fraction
                )
            else:
                fraction = self.final_fraction
        else:
            raise ValueError(f"Unknown curriculum strategy: {self.strategy}")

        return min(fraction, self.final_fraction)

    def __iter__(self):
        if self._current_indices is None:
            self.set_epoch(0)

        # Shuffle within the allowed indices
        rng = np.random.default_rng(seed=self._current_epoch + 42)
        shuffled = rng.permutation(self._current_indices)
        return iter(shuffled.tolist())

    def __len__(self) -> int:
        if self._current_indices is None:
            self.set_epoch(0)
        return len(self._current_indices)


class MMLUReasoningDataset(Dataset):
    """
    MMLU dataset augmented with teacher-generated reasoning chains.

    Each item contains:
    - The original MMLU question, choices, and answer
    - A chain-of-thought reasoning generated by the teacher model
    - Token IDs and labels for the combined prompt + reasoning sequence
    - A difficulty score (teacher entropy)

    Reasoning chains are cached to disk to avoid redundant generation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        split: str = "auxiliary_train",
        dataset_name: str = "cais/mmlu",
        cache_dir: Optional[str] = None,
        reasoning_cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.reasoning_cache_dir = reasoning_cache_dir

        if reasoning_cache_dir:
            os.makedirs(reasoning_cache_dir, exist_ok=True)

        # Load MMLU dataset from HuggingFace
        logger.info(f"Loading MMLU dataset: {dataset_name} / {split}")
        ds = load_dataset(dataset_name, "all", split=split, cache_dir=cache_dir)

        if max_samples is not None and max_samples < len(ds):
            logger.info(f"Limiting dataset to {max_samples} samples (from {len(ds)})")
            ds = ds.select(range(max_samples))

        self.raw_data = ds
        logger.info(f"Loaded {len(self.raw_data)} MMLU samples")

        # Storage for reasoning chains and difficulty scores
        self.reasoning_chains: list[Optional[str]] = [None] * len(self.raw_data)
        self.difficulty_scores: Optional[np.ndarray] = None
        self.prompts: list[str] = []

        # Pre-compute formatted prompts
        for idx in range(len(self.raw_data)):
            self.prompts.append(self._format_prompt(idx))

    def _format_prompt(self, idx: int) -> str:
        """Format an MMLU sample into a prompt string."""
        item = self.raw_data[idx]
        question = item["question"]
        choices = item["choices"]
        subject = item.get("subject", "")

        formatted_choices = "\n".join(
            f"  ({label}) {choice}"
            for label, choice in zip(CHOICE_LABELS, choices)
        )
        subject_str = f" ({subject.replace('_', ' ')})" if subject else ""

        return (
            f"Question{subject_str}: {question}\n"
            f"Choices:\n{formatted_choices}\n\n"
            f"Let's think step by step:\n"
        )

    def generate_reasoning_chains(
        self,
        teacher_model: "TeacherModel",
        batch_size: int = 8,
    ) -> None:
        """
        Generate reasoning chains for all samples using the teacher model.
        Results are cached to disk to avoid re-generation. Uses batched
        generation for significant speedup.

        Args:
            teacher_model: The frozen teacher model.
            batch_size: Number of samples to process at once.
        """
        n_cached = 0
        n_generated = 0

        # First pass: load all cached chains
        uncached_indices = []
        for idx in range(len(self.raw_data)):
            cached = self._load_cached_reasoning(idx)
            if cached is not None:
                self.reasoning_chains[idx] = cached
                n_cached += 1
            else:
                uncached_indices.append(idx)

        logger.info(
            f"Reasoning chains: {n_cached} loaded from cache, "
            f"{len(uncached_indices)} need generation"
        )

        if not uncached_indices:
            return

        # Generate in batches for uncached samples
        for batch_start in tqdm(
            range(0, len(uncached_indices), batch_size),
            desc="Generating reasoning chains",
            total=(len(uncached_indices) + batch_size - 1) // batch_size,
        ):
            batch_indices = uncached_indices[batch_start : batch_start + batch_size]
            questions = []
            choices_list = []
            subjects = []

            for idx in batch_indices:
                item = self.raw_data[idx]
                questions.append(item["question"])
                choices_list.append(item["choices"])
                subjects.append(item.get("subject", ""))

            results = teacher_model.generate_reasoning_batch(
                questions=questions,
                choices_list=choices_list,
                subjects=subjects,
            )

            for idx, result in zip(batch_indices, results):
                self.reasoning_chains[idx] = result["reasoning"]
                self._save_cached_reasoning(idx, result["reasoning"])
                n_generated += 1

        logger.info(
            f"Reasoning chains: {n_generated} generated, {n_cached} loaded from cache"
        )

    def _cache_key(self, idx: int) -> str:
        """Generate a deterministic cache key for a sample."""
        item = self.raw_data[idx]
        content = f"{item['question']}|{'|'.join(item['choices'])}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cached_reasoning(self, idx: int) -> Optional[str]:
        """Load a cached reasoning chain from disk."""
        if not self.reasoning_cache_dir:
            return None
        cache_path = Path(self.reasoning_cache_dir) / f"{self._cache_key(idx)}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                return data.get("reasoning")
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _save_cached_reasoning(self, idx: int, reasoning: str) -> None:
        """Save a reasoning chain to disk cache."""
        if not self.reasoning_cache_dir:
            return
        cache_path = Path(self.reasoning_cache_dir) / f"{self._cache_key(idx)}.json"
        with open(cache_path, "w") as f:
            json.dump({"reasoning": reasoning, "index": idx}, f)

    def set_difficulty_scores(self, scores: np.ndarray) -> None:
        """Set pre-computed difficulty scores for all samples."""
        assert len(scores) == len(self.raw_data), (
            f"Score count ({len(scores)}) must match dataset size ({len(self.raw_data)})"
        )
        self.difficulty_scores = scores

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single tokenized sample for training.

        Returns a dict with:
            - input_ids: Token IDs for prompt + reasoning (max_seq_length,)
            - attention_mask: 1 for real tokens, 0 for padding (max_seq_length,)
            - labels: Same as input_ids but with prompt tokens set to -100
                      so CE loss only applies to reasoning tokens
            - difficulty: Scalar difficulty score (or 0.0 if not scored)
            - answer_idx: Integer index of the correct answer (0-3)
        """
        item = self.raw_data[idx]
        prompt = self.prompts[idx]
        reasoning = self.reasoning_chains[idx] or ""

        # Build the full sequence: prompt + reasoning + answer indication
        answer_idx = item["answer"]
        answer_label = CHOICE_LABELS[answer_idx]
        full_text = f"{prompt}{reasoning}\n\nThe answer is ({answer_label})."

        # Tokenize the full sequence
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create labels: mask the prompt portion with -100 so the CE loss
        # is only computed on reasoning + answer tokens
        prompt_encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        prompt_length = prompt_encoded["input_ids"].size(1)

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Ignore prompt tokens in CE loss
        labels[attention_mask == 0] = -100  # Ignore padding

        difficulty = 0.0
        if self.difficulty_scores is not None:
            difficulty = float(self.difficulty_scores[idx])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "difficulty": torch.tensor(difficulty, dtype=torch.float32),
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long),
        }
