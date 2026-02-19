"""
Student model for reasoning chain distillation.

The student is a smaller language model (e.g., DistilGPT-2 82M) that learns
to approximate the teacher's reasoning capabilities through distillation.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class StudentModel(nn.Module):
    """
    Trainable student model that learns to reproduce chain-of-thought
    reasoning from the teacher's soft targets.

    The student shares the same tokenizer as the teacher (both are GPT-2
    family) to ensure vocabulary alignment for logit-level distillation.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading student model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Student model loaded: {total_params / 1e6:.1f}M total params, "
            f"{trainable_params / 1e6:.1f}M trainable"
        )

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute student logits for a batch of token sequences.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.get_logits(input_ids, attention_mask)

    def save_pretrained(self, save_path: str) -> None:
        """
        Save the student model and tokenizer to disk.

        Args:
            save_path: Directory to save model files.
        """
        logger.info(f"Saving student model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        device: Optional[torch.device] = None,
    ) -> "StudentModel":
        """
        Load a previously saved student model.

        Args:
            load_path: Directory containing saved model files.
            device: Device to load the model onto.

        Returns:
            StudentModel instance with loaded weights.
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.model_name = load_path

        logger.info(f"Loading student model from {load_path}")
        instance.tokenizer = AutoTokenizer.from_pretrained(load_path)
        instance.model = AutoModelForCausalLM.from_pretrained(load_path)

        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
            instance.model.config.pad_token_id = instance.tokenizer.eos_token_id

        instance.model.to(instance.device)
        return instance
