# Adaptive Reasoning Chain Distillation

Distilling chain-of-thought reasoning from large language models into smaller, more efficient student models using adaptive difficulty-based curriculum learning on MMLU.

## Motivation

Large language models (LLMs) can produce impressive step-by-step reasoning when prompted appropriately ("chain-of-thought" prompting). However, these models are expensive to deploy -- GPT-2 Medium at 345M parameters is already unwieldy for many applications, and production-scale models are orders of magnitude larger.

This project explores whether we can *distill* the reasoning capability itself, not just the final answers. The key insight is that a teacher model's chain-of-thought contains structured knowledge about *how to reason* through a problem, and a student model can learn to approximate this reasoning process through careful distillation.

We add a second insight: **not all questions are equally useful for learning to reason**. Easy questions teach basic pattern matching; hard questions can overwhelm an undertrained student. By scoring question difficulty using the teacher's own uncertainty (entropy) and presenting questions in a curriculum from easy to hard, we give the student a structured learning trajectory -- much like how human education works.

## Method

### Overview

```
                         MMLU Question
                              |
                    +---------+---------+
                    |                   |
              [Teacher Model]     [Format as prompt]
              (GPT-2 Medium)           |
                    |                  |
          +--------+--------+         |
          |                 |         |
    [Reasoning Chain]  [Entropy]      |
    "Let's think       (difficulty    |
     step by step..."   score)        |
          |                 |         |
          +-------+---------+---------+
                  |
          [Tokenized Sequence]
          prompt + reasoning + answer
                  |
          +-------+-------+
          |               |
    [Teacher Logits] [Student Logits]
     (frozen, no      (trainable)
      gradient)            |
          |               |
          +-------+-------+
                  |
          [Distillation Loss]
          0.7 * KL-div + 0.3 * CE
                  |
          [Gradient Update]
          (student only)
```

### Architecture

```
+------------------------------------------------------------------+
|                    DISTILLATION PIPELINE                          |
|                                                                  |
|  +--------------------+         +--------------------+           |
|  |   Teacher Model    |         |   Student Model    |           |
|  |   (GPT-2 Medium)   |         |   (DistilGPT-2)    |           |
|  |   345M params       |         |   82M params        |           |
|  |   FROZEN            |         |   TRAINABLE         |           |
|  +--------+-----+-----+         +--------+-----------+           |
|           |     |                         |                      |
|     logits|     |entropy           logits |                      |
|           v     v                         v                      |
|  +--------+-----+-----+         +--------+-----------+           |
|  | Soft probability    |         | Soft probability    |           |
|  | distribution        |<--KL-->| distribution        |           |
|  | (T=2.0)             |  div   | (T=2.0)             |           |
|  +---------------------+         +---------+----------+           |
|                                            |                     |
|  +---------------------+                   | CE loss             |
|  | Difficulty           |                   v                     |
|  | Calibrator           |         +---------+----------+           |
|  |                     |         | Combined Loss       |           |
|  | entropy -> bins     |         | L = 0.7*KL + 0.3*CE|           |
|  +--------+------------+         +--------------------+           |
|           |                                                      |
|           v                                                      |
|  +--------+------------+                                         |
|  | Curriculum Sampler   |                                         |
|  | easy -> hard         |                                         |
|  | over training        |                                         |
|  +---------------------+                                         |
+------------------------------------------------------------------+
```

### Difficulty Calibration

We measure question difficulty by the teacher model's predictive entropy:

- **Low entropy**: Teacher is confident. The question is "easy" -- clear patterns.
- **High entropy**: Teacher is uncertain. The question is "hard" -- ambiguous or requires deeper reasoning.

Questions are sorted by entropy and divided into difficulty bins. The curriculum sampler starts training with only the easiest 30% of questions and progressively includes harder ones according to a configurable schedule (linear, exponential, or step).

### Loss Function

The total loss combines two objectives:

1. **KL-divergence loss** (weight: 0.7): Matches the student's output distribution to the teacher's temperature-scaled soft targets. This transfers the teacher's "dark knowledge" -- the relative probabilities assigned to non-top tokens encode information about semantic similarity and reasoning structure.

2. **Cross-entropy loss** (weight: 0.3): Standard next-token prediction on the reasoning chain text. This grounds the student in producing coherent text, preventing mode collapse from pure distribution matching.

```
L_total = 0.7 * T^2 * KL(teacher_soft || student_soft) + 0.3 * CE(student, tokens)
```

where T=2.0 is the distillation temperature.

### Key Metric: Difficulty Calibration

The primary evaluation metric is `difficulty_calibration` -- the Spearman rank correlation between:
- Teacher's predicted difficulty (entropy ranking)
- Student's actual difficulty (per-sample loss ranking)

High correlation (near 1.0) means the curriculum ordering is well-calibrated: the questions the teacher finds hard are also the ones the student struggles with. This indicates meaningful transfer of the teacher's difficulty structure.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd adaptive-reasoning-chain-distillation

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package (editable mode)
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU with 24GB+ VRAM (tested on RTX 3090)
- ~10GB disk space for models and cached data

## Usage

### Basic Training

```bash
# Full training run (downloads MMLU on first run)
WANDB_MODE=disabled python scripts/train.py --config configs/default.yaml

# Quick debug run with limited data
WANDB_MODE=disabled python scripts/train.py --config configs/default.yaml --max-samples 500

# Disable FP16 if encountering numerical issues
WANDB_MODE=disabled python scripts/train.py --config configs/default.yaml --no-fp16

# Resume from checkpoint
WANDB_MODE=disabled python scripts/train.py --config configs/default.yaml --resume-from checkpoints/epoch_3
```

### Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.teacher.name` | `gpt2-medium` | Teacher model (345M) |
| `model.student.name` | `distilgpt2` | Student model (82M) |
| `training.batch_size` | `2` | Per-GPU batch size |
| `training.gradient_accumulation_steps` | `4` | Effective batch = 8 |
| `training.epochs` | `10` | Training epochs |
| `training.learning_rate` | `5e-5` | Peak learning rate |
| `loss.kl_temperature` | `2.0` | Distillation temperature |
| `loss.kl_weight` | `0.7` | KL loss weight |
| `curriculum.initial_fraction` | `0.3` | Start with easiest 30% |
| `curriculum.strategy` | `linear` | Difficulty ramp schedule |

### Project Structure

```
adaptive-reasoning-chain-distillation/
├── configs/
│   └── default.yaml              # Training configuration
├── scripts/
│   └── train.py                  # Main training entry point
├── src/
│   └── adaptive_reasoning_chain_distillation/
│       ├── models/
│       │   ├── teacher.py        # Frozen teacher model wrapper
│       │   ├── student.py        # Trainable student model
│       │   └── distiller.py      # KL + CE distillation logic
│       ├── data/
│       │   └── mmlu_loader.py    # MMLU loader, difficulty scorer, curriculum sampler
│       ├── training/
│       │   └── trainer.py        # Training loop with curriculum
│       └── evaluation/
│           └── metrics.py        # Difficulty calibration & evaluation metrics
├── checkpoints/                  # Saved model checkpoints
├── cache/                        # Cached reasoning chains
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Training Results

Trained on 10K MMLU samples with curriculum learning on an NVIDIA RTX 3090 (24 GB). Reasoning chains pre-generated by GPT-2 Medium teacher (~22 min), followed by 10 epochs of distillation training (~40 min). Total wall time: ~1.1 hours.

| Epoch | Distill Loss | KL Loss | CE Loss | Diff. Calibration | Curriculum % | Samples |
|-------|-------------|---------|---------|-------------------|-------------|---------|
| 1 | 1.238 | 0.991 | 1.816 | 0.098 | 30% | 3,000 |
| 2 | 1.098 | 0.888 | 1.588 | 0.179 | 30% | 3,000 |
| 3 | 1.040 | 0.837 | 1.513 | 0.252 | 38% | 3,777 |
| 4 | 0.994 | 0.800 | 1.449 | 0.307 | 46% | 4,555 |
| 5 | 0.961 | 0.776 | 1.393 | 0.335 | 53% | 5,333 |
| 6 | 0.941 | 0.758 | 1.368 | **0.351** | 61% | 6,111 |
| 7 | 0.924 | 0.748 | 1.335 | 0.339 | 69% | 6,888 |
| 8 | 0.910 | 0.738 | 1.310 | 0.327 | 77% | 7,666 |
| 9 | 0.900 | 0.730 | 1.296 | 0.325 | 84% | 8,444 |
| 10 | 0.892 | 0.725 | 1.279 | 0.317 | 92% | 9,222 |

**Best Difficulty Calibration**: 0.351 (epoch 6, Spearman rho)

### Analysis

The distillation loss decreases monotonically from 1.238 to 0.892 across 10 epochs, with both KL-divergence and cross-entropy components contributing. The difficulty calibration metric peaks at 0.351 around epoch 6 when the curriculum reaches ~61% of the dataset -- this is the point where the student has learned enough from easy examples to meaningfully discriminate difficulty levels, but before the hardest examples dilute the signal.

The curriculum progression from 30% to 92% shows the expected pattern: loss drops rapidly in early epochs when training on easier data (low teacher entropy), then declines more gradually as harder questions are introduced. Teacher entropy range was [1.479, 4.273] with mean 3.111, indicating good spread across difficulty levels.

This 10K-sample run serves as a proof-of-concept; scaling to the full 100K MMLU auxiliary_train split with longer reasoning chains and more epochs would improve both distillation quality and calibration.

### Configuration
- **Teacher**: GPT-2 Medium (345M params, frozen)
- **Student**: DistilGPT-2 (82M params, trainable)
- **Dataset**: MMLU auxiliary_train (10K subset from ~100K)
- **GPU**: NVIDIA RTX 3090 (24 GB)
- **Batch size**: 2 (effective 8 with gradient accumulation)
- **Learning rate**: 5e-5 (cosine schedule with warmup)
- **Distillation temperature**: 2.0
- **Loss weights**: 0.7 KL + 0.3 CE

## How It Works (In Detail)

### Step 1: Reasoning Chain Generation

For each MMLU question, the teacher model generates a chain-of-thought reasoning by completing the prompt:

```
Question (abstract algebra): Find the degree of the extension Q(sqrt(2), sqrt(3)) over Q.
Choices:
  (A) 1
  (B) 2
  (C) 4
  (D) 6

Let's think step by step:
```

The teacher's completion provides a reasoning chain that the student will learn to approximate. These are cached to disk to avoid repeated generation.

### Step 2: Difficulty Scoring

The teacher's predictive entropy on each question prompt is computed. Higher entropy means the teacher assigns more spread-out probability mass, indicating uncertainty. Questions are sorted by entropy and assigned to difficulty bins.

### Step 3: Curriculum Training

Training proceeds in epochs. Early epochs use only the easiest 30% of questions (low teacher entropy). As training progresses, the difficulty threshold increases linearly until all questions are included. This prevents the student from being overwhelmed by hard questions before it has learned basic reasoning patterns.

### Step 4: Distillation

At each step, the same input sequence is fed through both teacher and student. The teacher's logits (frozen, no gradient) serve as soft targets. The loss combines KL-divergence (matching distributions) with cross-entropy (predicting correct tokens), and only the student's parameters are updated.

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531
- Wei, J., Wang, X., Schuurmans, D., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
- Hendrycks, D., Burns, C., Basart, S., et al. (2021). *Measuring Massive Multitask Language Understanding.* ICLR 2021.
- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). *Curriculum Learning.* ICML 2009.
- Ho, N. & Vasconcelos, N. (2023). *Large Language Models Are Reasoning Teachers.* ACL 2023.
- Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., et al. (2023). *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes.* ACL 2023 Findings.
- Magister, L. C., Mallinson, J., Adamek, J., Malmi, E., & Severyn, A. (2023). *Teaching Small Language Models to Reason.* ACL 2023.

## License

MIT License. See [LICENSE](LICENSE) for details.
