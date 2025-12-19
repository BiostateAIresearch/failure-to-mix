# Failure to Mix: Large Language Models Struggle to Answer According to Desired Probability Distributions

This repository contains the code and data for reproducing all experiments and figures in the paper:

> **Failure to Mix: Large Language Models Struggle to Answer According to Desired Probability Distributions**  
> Ivy Yuqian Yang, David Yu Zhang  
> Biostate AI, Houston, TX

## Abstract

Scientific idea generation and selection requires exploration following a target probability distribution. In contrast, current AI benchmarks have objectively correct answers, and training large language models (LLMs) via reinforcement learning against these benchmarks discourages probabilistic exploration. Here, we conducted systematic experiments requesting LLMs to produce outputs following simple probabilistic distributions, and found that all modern LLMs tested grossly fail to follow the distributions. For example, requesting a binary output of "1" 49% of the time produces an answer of "0" nearly 100% of the time. This step function-like behavior of near-exclusively generating the output with marginally highest probability even overrules strong in-built LLM biases.

## Repository Structure

```
failure-to-mix/
├── data/                          # Raw experimental data
│   ├── fig1_single_flip.csv       # Figure 1: Single flip (8 models)
│   ├── fig2_*.csv                 # Figure 2: Sequential flips D=2,3
│   ├── fig3_*.csv                 # Figure 3: D=10 ensemble
│   ├── fig4_ternary.csv           # Figure 4: Ternary distribution
│   ├── fig5_word_bias.csv         # Figure 5: Word bias experiments
│   └── fig6_*.csv                 # Figure 6: Game theory scenarios
├── src/                           # Core library
│   ├── config.py                  # Configuration and model colors
│   ├── api_caller.py              # Async API caller with retry logic
│   ├── parsers.py                 # Response parsing utilities
│   ├── metrics.py                 # S metric and analysis functions
│   ├── plotting.py                # Plotting utilities
│   └── drive_uploader.py          # Google Drive integration
├── experiments/                   # Experiment modules
│   ├── exp1_single_flip.py        # Figure 1 experiments
│   ├── exp2_two_flips.py          # Figure 2 experiments
│   ├── exp3_multi_flip.py         # Figure 3 experiments
│   ├── exp4_ternary.py            # Figure 4 experiments
│   ├── exp5_word_bias.py          # Figure 5 experiments
│   └── exp6_game_theory.py        # Figure 6 experiments
├── scripts/                       # CLI tools
│   ├── run_experiments.py         # Run experiments
│   └── generate_figures.py        # Generate paper figures
├── notebooks/                     # Jupyter notebooks
│   └── run_all_experiments.ipynb  # Complete Colab notebook
├── figures/                       # Generated figures
├── requirements.txt
├── setup.py
└── LICENSE
```

## Models Tested

| Model | Provider | Color |
|-------|----------|-------|
| Gemini 2.5 Pro | Google | #1f77b4 (Blue) |
| GPT-5 | OpenAI | #2ca02c (Green) |
| GPT-5 Nano | OpenAI | #d62a28 (Red) |
| Claude 4.5 Sonnet | Anthropic | #8c564b (Brown) |
| Kimi K2 | Moonshot | #e377c2 (Pink) |
| Qwen 3 | Alibaba | #662d91 (Purple) |
| Grok-4 Fast | xAI | #c2b59b (Tan) |
| DeepSeek V3.2 | DeepSeek | #9467bd (Purple) |

## Key Metric: Step-likeness S

The S metric quantifies how much an observed response curve deviates from the ideal r = p diagonal:

$$S = 4 \times \sum_{a} (p_{a+1} - p_a) \times \frac{|r(p_a) - p_a| + |r(p_{a+1}) - p_{a+1}|}{2}$$

- **S = 0**: Perfect linear response (r = p)
- **S = 1**: Perfect step function at p = 50%

## Quick Start

### Installation

```bash
git clone https://github.com/BiostateAIresearch/failure-to-mix.git
cd failure-to-mix
pip install -r requirements.txt
```

### Set API Keys

```bash
export OPENROUTER_API_KEY="your-api-key"
```

### Run Experiments

```bash
# Run single flip experiment (Figure 1)
python scripts/run_experiments.py --model google/gemini-2.5-pro --exp 1 --n 100

# Run all experiments for a model
python scripts/run_experiments.py --model google/gemini-2.5-pro --all

# Generate figures from data
python scripts/generate_figures.py --data-dir ./data --output-dir ./figures
```

### Google Colab

Open `notebooks/run_all_experiments.ipynb` in Google Colab for a complete interactive experience.

## Experiments Overview

### Figure 1: Single Coin Flip
- **Prompt**: "Flip a biased coin that returns '1' p% of the time"
- **Result**: All 8 models show step-function behavior at p=50%
- **Data**: `data/fig1_single_flip.csv`

### Figure 2: Multiple Sequential Decisions (D=2, D=3)
- **Finding**: j=1 remains step function, j=2 shows "zigzag" compensation
- **Mean(r)** approaches linear as D increases
- **Data**: `data/fig2_*.csv`

### Figure 3: D=10 Ensemble Analysis
- Individual decisions still show non-linear behavior even at j=10
- Histogram shows much tighter distribution than expected i.i.d.
- **Data**: `data/fig3_*.csv`

### Figure 4: Ternary Distribution (0/1/2)
- Fixed p(1) = 40%, varying q = p(2)
- Model-dependent behavior, not purely step-function
- **Data**: `data/fig4_ternary.csv`

### Figure 5: Word Bias
- 11 word pairs tested (luxury/affordable, sun/moon, human/alien, etc.)
- Position bias exists but is overruled by even 1% probability difference
- **Data**: `data/fig5_word_bias.csv`

### Figure 6: Game Theory Applications
- **Bioinformatics**: NGS read alignment
- **Matching Pennies**: Asymmetric payoffs
- **Business Positioning**: Luxury vs Affordable
- LLMs correctly reason about mixed strategies but fail to execute them
- **Data**: `data/fig6_*.csv`

## Data Format

### Single Flip (fig1)
```csv
model,p,answer,status
google/gemini-2.5-pro,45,0,success
google/gemini-2.5-pro,45,0,success
...
```

### Word Bias (fig5)
```csv
model,p,word1,word2,raw_text,answer
google/gemini-2.5-pro,50,luxury,affordable,luxury,luxury
...
```

### Game Theory (fig6)
```csv
model,optimal_p,prompt,raw_text,answer
openai/gpt-5,40.0,"You are assisting with a bioinformatics task...",B,0
...
```

## Citation

```bibtex
@article{yang2025failure,
  title={Failure to Mix: Large Language Models Struggle to Answer According to Desired Probability Distributions},
  author={Yang, Ivy Yuqian and Zhang, David Yu},
  journal={Nature Methods & Intelligence},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- Ivy Yang: ivy.yang@biostate.ai
- Dave Zhang: dave.zhang@biostate.ai
