"""
Reasoning chain distillation logic.

Implements the core distillation objective: KL-divergence between teacher
and student logits (soft targets) combined with cross-entropy on reasoning
tokens (hard targets). The combined loss trains the student to approximate
both the teacher's output distribution and the actual reasoning chain text.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .teacher import TeacherModel
from .student import StudentModel

logger = logging.getLogger(__name__)


@dataclass
class DistillationOutput:
    """Container for distillation forward pass results."""
    loss: torch.Tensor
    kl_loss: torch.Tensor
    ce_loss: torch.Tensor
    student_logits: torch.Tensor
    teacher_logits: torch.Tensor


class ReasoningDistiller(nn.Module):
    """
    Orchestrates knowledge distillation from teacher to student.

    Combines two loss signals:
    1. KL-divergence: Aligns student's output distribution with teacher's
       soft probability distribution (temperature-scaled).
    2. Cross-entropy: Trains student to produce correct next-token
       predictions on the reasoning chain text.

    The KL loss captures the teacher's "dark knowledge" -- the relative
    probabilities it assigns to incorrect tokens, which encode useful
    information about token similarity and reasoning patterns.
    """

    def __init__(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        kl_temperature: float = 2.0,
        kl_weight: float = 0.7,
        ce_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.kl_temperature = kl_temperature
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight

        # Validate vocab sizes match (both GPT-2 family)
        teacher_vocab = teacher.model.config.vocab_size
        student_vocab = student.model.config.vocab_size
        if teacher_vocab != student_vocab:
            raise ValueError(
                f"Vocabulary size mismatch: teacher={teacher_vocab}, student={student_vocab}. "
                f"Both models must share the same tokenizer vocabulary."
            )

        logger.info(
            f"Distiller initialized: KL weight={kl_weight}, CE weight={ce_weight}, "
            f"temperature={kl_temperature}"
        )

    def compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temperature-scaled KL-divergence loss between student and
        teacher logit distributions.

        The temperature softens the probability distributions, making the
        teacher's "dark knowledge" (relative probabilities of non-top tokens)
        more visible to the student.

        Args:
            student_logits: Student output logits (batch, seq_len, vocab).
            teacher_logits: Teacher output logits (batch, seq_len, vocab).
            attention_mask: Mask to exclude padding (batch, seq_len).

        Returns:
            Scalar KL-divergence loss.
        """
        T = self.kl_temperature

        # Temperature-scaled log-softmax for student, softmax for teacher
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        # KL(teacher || student) per token
        # Using F.kl_div with log_target=False: expects input=log_probs, target=probs
        kl_per_token = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)  # Sum over vocab -> (batch, seq_len)

        # Mask padding and average
        mask = attention_mask.float().to(kl_per_token.device)
        masked_kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1.0)

        # Scale by T^2 as per Hinton et al. (2015) to ensure gradient
        # magnitudes are appropriately balanced with CE loss
        return masked_kl * (T ** 2)

    def compute_ce_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss on reasoning chain tokens.

        This loss grounds the student's predictions in the actual reasoning
        text, preventing mode collapse that could occur with pure KL
        distillation.

        Args:
            student_logits: Student output logits (batch, seq_len, vocab).
            labels: Ground-truth token IDs (batch, seq_len). Positions with
                    value -100 are ignored.
            attention_mask: Mask to exclude padding (batch, seq_len).

        Returns:
            Scalar cross-entropy loss.
        """
        # Shift logits and labels for next-token prediction
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Flatten for cross-entropy
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        # Compute per-token CE loss
        ce_per_token = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            ignore_index=-100,
        )

        # Reshape and apply mask
        ce_per_token = ce_per_token.view(shift_labels.size())
        flat_mask = shift_mask.float().to(ce_per_token.device)

        # Only count non-ignored, non-padded positions
        valid = (shift_labels != -100).float().to(ce_per_token.device) * flat_mask
        masked_ce = (ce_per_token * valid).sum() / valid.sum().clamp(min=1.0)

        return masked_ce

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> DistillationOutput:
        """
        Compute the combined distillation loss.

        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).
            labels: Ground-truth token IDs for CE loss (batch, seq_len).
                    If None, only KL loss is computed.

        Returns:
            DistillationOutput with loss components and logits.
        """
        # Move all inputs to the student's device
        device = self.student.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Teacher forward (no gradient)
        with torch.no_grad():
            teacher_logits = self.teacher.get_logits(input_ids, attention_mask)

        # Student forward
        student_logits = self.student.get_logits(input_ids, attention_mask)

        # Ensure shapes match (truncate to shorter if teacher/student differ)
        min_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits_aligned = student_logits[:, :min_len, :]
        teacher_logits_aligned = teacher_logits[:, :min_len, :]
        mask_aligned = attention_mask[:, :min_len]

        # KL divergence loss
        kl_loss = self.compute_kl_loss(
            student_logits_aligned,
            teacher_logits_aligned,
            mask_aligned,
        )

        # Cross-entropy loss on reasoning tokens
        if labels is not None:
            labels_aligned = labels[:, :min_len]
            ce_loss = self.compute_ce_loss(
                student_logits_aligned,
                labels_aligned,
                mask_aligned,
            )
        else:
            ce_loss = torch.tensor(0.0, device=student_logits.device)

        # Combined loss
        total_loss = self.kl_weight * kl_loss + self.ce_weight * ce_loss

        return DistillationOutput(
            loss=total_loss,
            kl_loss=kl_loss,
            ce_loss=ce_loss,
            student_logits=student_logits_aligned,
            teacher_logits=teacher_logits_aligned,
        )
