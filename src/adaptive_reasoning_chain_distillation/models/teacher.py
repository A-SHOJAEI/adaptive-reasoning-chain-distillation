"""
Teacher model wrapper for reasoning chain generation.

The teacher is a frozen pre-trained language model (e.g., GPT-2 Medium) that
generates chain-of-thought reasoning for MMLU questions. Its logits serve as
soft targets for the student model during distillation.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class TeacherModel(nn.Module):
    """
    Frozen teacher model that generates reasoning chains and provides
    soft logit targets for knowledge distillation.

    The teacher processes MMLU questions with a chain-of-thought prompt
    and returns both generated reasoning text and token-level logits
    that the student model learns to approximate.
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading teacher model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # GPT-2 tokenizer does not have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        # Use left-padding for decoder-only batch generation
        self.tokenizer.padding_side = "left"

        # Freeze all parameters -- teacher is never trained
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Teacher model loaded: {param_count / 1e6:.1f}M parameters (frozen)")

    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute teacher logits for a batch of token sequences.

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

    @torch.no_grad()
    def compute_entropy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample entropy of teacher predictions. Higher entropy
        indicates the teacher is more uncertain, implying a harder question.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Per-sample mean entropy of shape (batch_size,).
        """
        logits = self.get_logits(input_ids, attention_mask)
        # Compute entropy over vocabulary dimension
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)

        # Mask out padding tokens and average
        mask = attention_mask.to(token_entropy.device).float()
        masked_entropy = (token_entropy * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        return masked_entropy

    @torch.no_grad()
    def generate_reasoning(
        self,
        question: str,
        choices: list[str],
        subject: str = "",
    ) -> dict:
        """
        Generate a chain-of-thought reasoning for an MMLU question.

        Constructs a prompt that encourages step-by-step reasoning, then
        generates a completion from the teacher model.

        Args:
            question: The MMLU question text.
            choices: List of answer choices (typically 4).
            subject: The MMLU subject/category.

        Returns:
            Dictionary containing:
                - "prompt": The full prompt sent to the teacher.
                - "reasoning": The generated reasoning chain text.
                - "full_text": Prompt + reasoning concatenated.
                - "entropy": Scalar entropy of teacher on the prompt.
        """
        choice_labels = ["A", "B", "C", "D"]
        formatted_choices = "\n".join(
            f"  ({label}) {choice}" for label, choice in zip(choice_labels, choices)
        )
        subject_str = f" ({subject})" if subject else ""

        prompt = (
            f"Question{subject_str}: {question}\n"
            f"Choices:\n{formatted_choices}\n\n"
            f"Let's think step by step:\n"
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512 - self.max_new_tokens,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Compute entropy on the prompt (used for difficulty scoring)
        entropy = self.compute_entropy(input_ids, attention_mask).item()

        # Generate reasoning chain
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reasoning = full_text[len(prompt):]

        return {
            "prompt": prompt,
            "reasoning": reasoning.strip(),
            "full_text": full_text,
            "entropy": entropy,
        }

    @torch.no_grad()
    def generate_reasoning_batch(
        self,
        questions: list[str],
        choices_list: list[list[str]],
        subjects: list[str],
    ) -> list[dict]:
        """
        Generate reasoning chains for a batch of MMLU questions.

        Args:
            questions: List of question texts.
            choices_list: List of choice lists (each has 4 choices).
            subjects: List of subjects.

        Returns:
            List of dicts with prompt, reasoning, full_text, entropy.
        """
        choice_labels = ["A", "B", "C", "D"]
        prompts = []
        for question, choices, subject in zip(questions, choices_list, subjects):
            formatted_choices = "\n".join(
                f"  ({label}) {choice}" for label, choice in zip(choice_labels, choices)
            )
            subject_str = f" ({subject})" if subject else ""
            prompt = (
                f"Question{subject_str}: {question}\n"
                f"Choices:\n{formatted_choices}\n\n"
                f"Let's think step by step:\n"
            )
            prompts.append(prompt)

        # Tokenize batch
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512 - self.max_new_tokens,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Compute entropy on prompts
        entropies = self.compute_entropy(input_ids, attention_mask).cpu().tolist()

        # Generate reasoning chains for the batch
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        results = []
        for i in range(len(prompts)):
            full_text = self.tokenizer.decode(output_ids[i], skip_special_tokens=True)
            reasoning = full_text[len(prompts[i]):]
            results.append({
                "prompt": prompts[i],
                "reasoning": reasoning.strip(),
                "full_text": full_text,
                "entropy": entropies[i],
            })

        return results

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning logits. Kept for nn.Module compatibility."""
        return self.get_logits(input_ids, attention_mask)
