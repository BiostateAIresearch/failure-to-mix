"""
Experiment 5: Word Choice and Position Bias (Figure 5)

Tests semantic word pairs at 50% probability to detect:
1. Word bias: Preference for certain words regardless of position
2. Position bias: Preference for first or second position
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

import sys
sys.path.insert(0, '..')
from src.api_caller import APICaller
from src.parsers import extract_word_choice
from src.config import ExperimentConfig
from src.plotting import plot_word_bias


# Word pairs to test (word1, word2)
DEFAULT_WORD_PAIRS = [
    ("luxury", "affordable"),
    ("1", "0"),
    ("sun", "moon"),
    ("black", "white"),
    ("day", "night"),
    ("human", "alien"),
    ("rich", "poor"),
    ("woman", "man"),
    ("Heads", "Tails"),
    ("A", "B"),
]


def create_word_flip_prompt(word1: str, word2: str, p: float, reversed: bool = False) -> str:
    """
    Create word flip prompt.
    
    Args:
        word1: First word option
        word2: Second word option
        p: Probability for word1 (%)
        reversed: If True, swap word order in prompt
        
    Returns:
        Formatted prompt string
    """
    if reversed:
        word1, word2 = word2, word1
        p = 100 - p
    
    return (
        f'I\'d like you to flip a biased coin that returns "{word1}" {p}% of the time '
        f'and "{word2}" {100-p}% of the time. Please return a 1 token response simulating a single flip.'
    )


async def run_exp5(
    model: str,
    config: ExperimentConfig,
    word_pairs: Optional[List[Tuple[str, str]]] = None,
    p_values: Optional[List[float]] = None,
    n_trials: int = 100
) -> pd.DataFrame:
    """
    Run word bias experiment.
    
    For each word pair, tests both orderings at specified p values.
    
    Args:
        model: Model identifier
        config: Experiment configuration
        word_pairs: List of (word1, word2) tuples
        p_values: List of p values to test (default: 49, 50, 51)
        n_trials: Number of trials per condition
        
    Returns:
        DataFrame with word pair info and responses
    """
    if word_pairs is None:
        word_pairs = DEFAULT_WORD_PAIRS
    
    if p_values is None:
        p_values = [49, 50, 51]
    
    caller = APICaller(
        api_keys=config.api_keys,
        api_url=config.api_url,
        max_retries=config.max_retries,
        timeout=config.timeout
    )
    
    # Build requests for both orderings
    requests = []
    for word1, word2 in word_pairs:
        for p in p_values:
            for reversed_order in [False, True]:
                prompt = create_word_flip_prompt(word1, word2, p, reversed=reversed_order)
                
                for trial in range(n_trials):
                    requests.append({
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                        "metadata": {
                            "word1": word1,
                            "word2": word2,
                            "p": p,
                            "reversed": reversed_order,
                            "trial": trial,
                            "prompt_first": word2 if reversed_order else word1,
                            "prompt_second": word1 if reversed_order else word2
                        }
                    })
    
    print(f"🧪 Running Experiment 5: Word Bias")
    print(f"   Model: {model}")
    print(f"   Word pairs: {len(word_pairs)}")
    print(f"   P values: {p_values}")
    print(f"   Trials per condition: {n_trials}")
    print(f"   Total calls: {len(requests)}")
    
    results = []
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(requests), config.batch_size), desc="📊 Progress"):
            batch = requests[i:i + config.batch_size]
            tasks = []
            
            for j, req in enumerate(batch):
                api_key = config.api_keys[(i + j) % len(config.api_keys)]
                tasks.append(
                    caller.call(
                        session=session,
                        model=req["model"],
                        messages=req["messages"],
                        max_tokens=req["max_tokens"],
                        api_key=api_key
                    )
                )
            
            responses = await asyncio.gather(*tasks)
            
            for req, resp in zip(batch, responses):
                meta = req["metadata"]
                
                # Extract answer relative to original word1/word2
                answer = extract_word_choice(
                    resp.content,
                    meta["word1"],
                    meta["word2"]
                )
                
                # Also check if they chose first position word
                first_word = meta["prompt_first"]
                chose_first = extract_word_choice(resp.content, first_word, meta["prompt_second"]) == first_word
                
                results.append({
                    "word1": meta["word1"],
                    "word2": meta["word2"],
                    "p": meta["p"],
                    "reversed": meta["reversed"],
                    "prompt_first": meta["prompt_first"],
                    "prompt_second": meta["prompt_second"],
                    "trial": meta["trial"],
                    "raw_response": resp.content,
                    "answer": answer,
                    "chose_word1": answer == meta["word1"],
                    "chose_first_position": chose_first if answer != "error" else None,
                    "status": resp.status
                })
            
            if i + config.batch_size < len(requests):
                await asyncio.sleep(config.initial_wait)
    
    return pd.DataFrame(results)


def analyze_exp5(df: pd.DataFrame) -> Dict:
    """
    Analyze word bias results.
    
    Args:
        df: Results DataFrame from run_exp5
        
    Returns:
        Dictionary with word bias and position bias for each pair
    """
    valid = df[df["answer"] != "error"].copy()
    
    if valid.empty:
        return {"error": "No valid responses"}
    
    # Analyze each word pair
    results = []
    for (w1, w2), group in valid.groupby(["word1", "word2"]):
        # Word bias: rate of choosing word1 (regardless of position)
        word1_rate = group["chose_word1"].mean()
        
        # Position bias: rate of choosing first position (regardless of word)
        position_first_rate = group["chose_first_position"].mean()
        
        # By ordering
        normal = group[~group["reversed"]]
        reversed_df = group[group["reversed"]]
        
        r_normal = normal["chose_word1"].mean() if len(normal) > 0 else np.nan
        r_reversed = reversed_df["chose_word1"].mean() if len(reversed_df) > 0 else np.nan
        
        results.append({
            "word1": w1,
            "word2": w2,
            "word1_rate": word1_rate,
            "position_first_rate": position_first_rate,
            "r_when_first": r_normal,  # word1 rate when word1 is first in prompt
            "r_when_second": r_reversed,  # word1 rate when word1 is second in prompt
            "n_trials": len(group)
        })
    
    summary = pd.DataFrame(results)
    
    # Overall biases
    word_biases = summary["word1_rate"].values
    position_biases = summary["position_first_rate"].values
    
    return {
        "pairs": list(zip(summary["word1"], summary["word2"])),
        "word_bias": word_biases,
        "position_bias": position_biases,
        "r_when_first": summary["r_when_first"].values,
        "r_when_second": summary["r_when_second"].values,
        "mean_word_bias_deviation": np.mean(np.abs(word_biases - 0.5)),
        "mean_position_bias_deviation": np.mean(np.abs(position_biases - 0.5)),
        "summary": summary
    }


def plot_exp5(
    analysis: Dict,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot word bias results.
    
    Args:
        analysis: Output from analyze_exp5
        model_name: Model name for title
        save_path: Path to save figure
    """
    fig = plot_word_bias(
        word_pairs=analysis["pairs"],
        word_bias=analysis["word_bias"],
        position_bias=analysis["position_bias"],
        save_path=save_path
    )
    
    print(f"\n📊 Word Bias Analysis for {model_name}:")
    print(f"   Mean word bias deviation from 0.5: {analysis['mean_word_bias_deviation']:.4f}")
    print(f"   Mean position bias deviation from 0.5: {analysis['mean_position_bias_deviation']:.4f}")
    
    return fig


def main():
    """Command line entry point."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run word bias experiment")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-pro")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", type=str, default="./results/exp5")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    
    config = ExperimentConfig()
    os.makedirs(args.output, exist_ok=True)
    
    df = asyncio.run(run_exp5(
        model=args.model,
        config=config,
        n_trials=args.n
    ))
    
    raw_path = os.path.join(args.output, "raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"💾 Saved: {raw_path}")
    
    analysis = analyze_exp5(df)
    
    if not args.no_plot:
        plot_path = os.path.join(args.output, "word_bias.png")
        plot_exp5(analysis, args.model, save_path=plot_path)


if __name__ == "__main__":
    main()
