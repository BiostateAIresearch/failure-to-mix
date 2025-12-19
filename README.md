# Failure to Mix: LLM Probability Calibration

**Code repository for the paper:** *"Failure to Mix: A Fundamental Limitation of Large Language Models in Executing Probabilistic Tasks"*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code to reproduce all experiments in the "Failure to Mix" paper, which demonstrates that large language models exhibit a characteristic "step-like" response pattern when asked to execute probabilistic tasks like simulating biased coin flips.

### Key Finding

When asked to flip a biased coin returning "1" with probability p%, LLMs do not output "1" at rate r ≈ p%. Instead, they exhibit a sigmoid/step-like response:
- For p < 50%: r ≈ 0% (almost always output "0")
- For p > 50%: r ≈ 100% (almost always output "1")
- Transition occurs sharply around p = 50%

We quantify this with the **Step-likeness metric S**:
- S = 0: Perfect calibration (r = p for all p)
- S = 1: Perfect step function
- Observed: S ≈ 0.8-0.95 for most models

## Installation

```bash
# Clone repository
git clone https://github.com/BiostateAIresearch/failure-to-mix.git
cd failure-to-mix

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENROUTER_API_KEY="your-key-here"
```

## Quick Start

```bash
# Run single flip experiment on Gemini
python scripts/run_experiments.py --model google/gemini-2.5-pro --exp 1

# Run all experiments on GPT-5
python scripts/run_experiments.py --model openai/gpt-5 --all

# Run with custom trial count
python scripts/run_experiments.py --model anthropic/claude-4.5-sonnet --exp 1 2 --n 500
```

## Experiments

| Experiment | Figure | Description |
|------------|--------|-------------|
| `exp1` | Fig 1 | Single coin flip with varying p |
| `exp2` | Fig 2 | Two sequential flips (compensation analysis) |
| `exp3` | Fig 3 | D=10 flip ensemble (binomial distribution) |
| `exp4` | Fig 4 | Ternary distribution (0/1/2) |
| `exp5` | Fig 5 | Word choice and position bias |
| `exp6` | Fig 6 | Game theory mixed strategies |

## Repository Structure

```
failure-to-mix/
├── src/                      # Core library
│   ├── api_caller.py        # Async API with rate limiting
│   ├── config.py            # Configuration and prompts
│   ├── metrics.py           # S metric, KL divergence
│   ├── parsers.py           # Response extraction
│   ├── plotting.py          # Visualization utilities
│   └── drive_uploader.py    # Google Drive integration
├── experiments/              # Experiment implementations
│   ├── exp1_single_flip.py  # Figure 1
│   ├── exp2_two_flips.py    # Figure 2
│   ├── exp4_ternary.py      # Figure 4
│   ├── exp5_word_bias.py    # Figure 5
│   └── exp6_game_theory.py  # Figure 6
├── scripts/
│   ├── run_experiments.py   # Main CLI
│   └── generate_figures.py  # Paper figure generation
├── data/                     # Raw experimental data
├── figures/                  # Generated figures
└── notebooks/                # Jupyter notebooks
```

## Step-likeness Metric (S)

The key metric for measuring calibration quality:

```python
from src.metrics import compute_S

# p_values: target probabilities (%)
# r_values: observed response rates (%)
S = compute_S(p_values, r_values)

# S = 0: Perfect calibration
# S = 1: Perfect step function
```

**Formula:**
```
S = 4 × ∫₀¹ |r(p) - p| dp
```

## Models Tested

| Model | S Value | Calibration Score |
|-------|---------|-------------------|
| Gemini 2.5 Pro | 0.96 | 0.4/10 |
| GPT-5 | 0.85 | 1.5/10 |
| Claude 4.5 Sonnet | 0.82 | 1.8/10 |
| Kimi K2 | 0.78 | 2.2/10 |
| Qwen 3 | 0.91 | 0.9/10 |

## API Configuration

The code uses OpenRouter for unified API access. Set your key:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Or pass multiple keys for rate limit handling:
```bash
export OPENROUTER_API_KEY_0="sk-or-v1-..."
export OPENROUTER_API_KEY_1="sk-or-v1-..."
```

## Prompts Used

**Single Flip:**
```
I'd like you to flip a biased coin that returns "1" {p}% of the time 
and "0" {100-p}% of the time. Please return a 1 token response 
simulating a single flip.
```

**Two Flip (Turn 2):**
```
Please do it one more time.
```

## Citation

```bibtex
@article{failure-to-mix-2025,
  title={Failure to Mix: A Fundamental Limitation of Large Language Models 
         in Executing Probabilistic Tasks},
  author={...},
  journal={Nature Methods & Informatics},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- **Biostate AI Research**: research@biostate.ai
- **Issues**: Please open a GitHub issue for bugs or questions
