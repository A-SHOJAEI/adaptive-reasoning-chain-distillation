"""
Evaluation metrics for reasoning chain distillation.

The primary metric is difficulty_calibration: the Spearman rank correlation
between the teacher's predicted difficulty (entropy) and the student's actual
difficulty (per-sample loss). High correlation means the curriculum ordering
is well-calibrated -- questions the teacher finds hard are also the ones the
student struggles with.

Additional metrics track training health and convergence.
"""

import logging
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class MetricsComputer:
    """
    Computes evaluation metrics for reasoning chain distillation.

    Tracks:
    - difficulty_calibration: Spearman correlation between predicted
      (teacher entropy) and actual (student loss) difficulty rankings
    - mean_student_loss: Average per-sample student loss
    - loss_by_difficulty_bin: Mean student loss in each difficulty quintile
    - curriculum_effectiveness: Ratio of loss improvement on easy vs hard samples
    """

    def __init__(self, num_difficulty_bins: int = 5) -> None:
        self.num_difficulty_bins = num_difficulty_bins
        self.history: list[dict] = []

    def compute_difficulty_calibration(
        self,
        student_losses: np.ndarray,
        difficulty_scores: np.ndarray,
    ) -> float:
        """
        Compute Spearman rank correlation between predicted difficulty
        (teacher entropy scores) and actual difficulty (student losses).

        A value near 1.0 means the teacher's difficulty ordering perfectly
        predicts which questions the student finds hard. A value near 0.0
        means the ordering is essentially random.

        Args:
            student_losses: Per-sample student loss values.
            difficulty_scores: Per-sample teacher entropy scores.

        Returns:
            Spearman rho correlation coefficient in [-1, 1].
        """
        if len(student_losses) < 3 or len(difficulty_scores) < 3:
            logger.warning("Too few samples for difficulty calibration")
            return 0.0

        # Filter out samples where difficulty was not scored (score == 0)
        valid_mask = difficulty_scores > 0
        if valid_mask.sum() < 3:
            logger.warning("Too few scored samples for difficulty calibration")
            return 0.0

        valid_losses = student_losses[valid_mask]
        valid_difficulties = difficulty_scores[valid_mask]

        # Spearman rank correlation
        rho, p_value = stats.spearmanr(valid_difficulties, valid_losses)

        if np.isnan(rho):
            return 0.0

        return float(rho)

    def compute_loss_by_difficulty(
        self,
        student_losses: np.ndarray,
        difficulty_scores: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute mean student loss within each difficulty quintile.

        Args:
            student_losses: Per-sample student loss values.
            difficulty_scores: Per-sample teacher entropy scores.

        Returns:
            Dictionary mapping quintile names to mean losses.
        """
        valid_mask = difficulty_scores > 0
        if valid_mask.sum() < self.num_difficulty_bins:
            return {}

        valid_losses = student_losses[valid_mask]
        valid_difficulties = difficulty_scores[valid_mask]

        # Bin by difficulty percentile
        percentiles = np.percentile(
            valid_difficulties,
            np.linspace(0, 100, self.num_difficulty_bins + 1),
        )

        result = {}
        for i in range(self.num_difficulty_bins):
            low, high = percentiles[i], percentiles[i + 1]
            if i == self.num_difficulty_bins - 1:
                mask = (valid_difficulties >= low) & (valid_difficulties <= high)
            else:
                mask = (valid_difficulties >= low) & (valid_difficulties < high)

            if mask.sum() > 0:
                label = f"loss_bin_{i}"
                result[label] = float(valid_losses[mask].mean())

        return result

    def compute_curriculum_effectiveness(
        self,
        student_losses: np.ndarray,
        difficulty_scores: np.ndarray,
    ) -> float:
        """
        Measure how much better the student performs on easy vs hard samples.

        Computed as: (mean_loss_hard - mean_loss_easy) / mean_loss_overall

        A positive value means the student does better on easy questions
        (as expected), and the magnitude indicates how much the curriculum
        ordering matters.

        Args:
            student_losses: Per-sample student loss values.
            difficulty_scores: Per-sample teacher entropy scores.

        Returns:
            Curriculum effectiveness score (higher = more separation).
        """
        valid_mask = difficulty_scores > 0
        if valid_mask.sum() < 10:
            return 0.0

        valid_losses = student_losses[valid_mask]
        valid_difficulties = difficulty_scores[valid_mask]

        median_difficulty = np.median(valid_difficulties)
        easy_mask = valid_difficulties <= median_difficulty
        hard_mask = valid_difficulties > median_difficulty

        if easy_mask.sum() == 0 or hard_mask.sum() == 0:
            return 0.0

        mean_easy = valid_losses[easy_mask].mean()
        mean_hard = valid_losses[hard_mask].mean()
        mean_overall = valid_losses.mean()

        if mean_overall < 1e-8:
            return 0.0

        return float((mean_hard - mean_easy) / mean_overall)

    def compute_reasoning_quality(
        self,
        student_losses: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute statistics on student loss distribution as a proxy for
        reasoning quality.

        Args:
            student_losses: Per-sample student loss values.

        Returns:
            Dictionary with loss statistics.
        """
        if len(student_losses) == 0:
            return {"mean_loss": 0.0, "median_loss": 0.0, "std_loss": 0.0}

        return {
            "mean_student_loss": float(np.mean(student_losses)),
            "median_student_loss": float(np.median(student_losses)),
            "std_student_loss": float(np.std(student_losses)),
            "min_student_loss": float(np.min(student_losses)),
            "max_student_loss": float(np.max(student_losses)),
        }

    def compute_all(
        self,
        student_losses: np.ndarray,
        difficulty_scores: np.ndarray,
    ) -> dict:
        """
        Compute all metrics.

        Args:
            student_losses: Per-sample student loss values.
            difficulty_scores: Per-sample teacher entropy scores.

        Returns:
            Dictionary of all computed metrics.
        """
        metrics = {}

        # Primary metric: difficulty calibration
        metrics["difficulty_calibration"] = self.compute_difficulty_calibration(
            student_losses, difficulty_scores
        )

        # Loss by difficulty bin
        bin_losses = self.compute_loss_by_difficulty(student_losses, difficulty_scores)
        metrics.update(bin_losses)

        # Curriculum effectiveness
        metrics["curriculum_effectiveness"] = self.compute_curriculum_effectiveness(
            student_losses, difficulty_scores
        )

        # Reasoning quality statistics
        quality = self.compute_reasoning_quality(student_losses)
        metrics.update(quality)

        # Store history
        self.history.append(metrics.copy())

        return metrics

    def get_history(self) -> list[dict]:
        """Return the full history of computed metrics."""
        return self.history

    def get_best_epoch(self, metric: str = "difficulty_calibration") -> Optional[int]:
        """
        Find the epoch with the best value of a given metric.

        Args:
            metric: Metric name to optimize.

        Returns:
            Epoch index with the best metric value, or None if no history.
        """
        if not self.history:
            return None

        values = [h.get(metric, -float("inf")) for h in self.history]
        return int(np.argmax(values))
