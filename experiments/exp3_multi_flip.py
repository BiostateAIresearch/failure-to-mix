"""
Experiment 3: Multi-Flip Ensemble (Figure 3)

Tests D=10 sequential decisions to analyze:
1. Individual response curves for j=1,2,9,10
2. Mean response curve across all D decisions
3. Distribution of total "1" counts vs expected binomial
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
from scipy.stats import binom, chisquare

import sys
sys.path.insert(0, '..')
from src.api_caller import APICaller
from src.parsers import extract_multi_digits
from src.metrics import compute_S
from src.config import ExperimentConfig, get_model_color, EDGE_COLOR


# Prompt template for D sequential flips
PROMPT_MULTI_FLIP = '''I'd like you to flip a biased coin that returns "1" {p}% of the time and "0" {100_p}% of the time. Please return a {D} token response simulating {D} flips. Please return exactly {D} tokens (0 or 1) separated by a single space.'''


async def run_exp3(
    model: str,
    config: ExperimentConfig,
    p_values: Optional[List[float]] = None,
    D: int = 10,
    n_trials: int = 100
) -> pd.DataFrame:
    """
    Run D=10 multi-flip experiment.
    
    Args:
        model: Model identifier
        config: Experiment configuration
        p_values: List of p values to test (%)
        D: Number of flips per trial
        n_trials: Number of trials per p value
        
    Returns:
        DataFrame with columns: p, trial, j, answer, status
    """
    if p_values is None:
        p_values = list(range(0, 101, 5))
    
    caller = APICaller(
        api_keys=config.api_keys,
        api_url=config.api_url,
        max_retries=config.max_retries,
        timeout=config.timeout
    )
    
    print(f"🧪 Running Experiment 3: Multi-Flip (D={D})")
    print(f"   Model: {model}")
    print(f"   P values: {len(p_values)} values from {min(p_values)} to {max(p_values)}")
    print(f"   Trials per p: {n_trials}")
    
    results = []
    async with aiohttp.ClientSession() as session:
        for p in tqdm(p_values, desc="📊 Progress"):
            prompt = PROMPT_MULTI_FLIP.format(p=p, D=D, **{"100_p": 100-p})
            
            for trial in range(n_trials):
                api_key = config.api_keys[(p * n_trials + trial) % len(config.api_keys)]
                
                resp = await caller.call(
                    session=session,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=D * 2 + 10,
                    api_key=api_key
                )
                
                # Parse D digits
                text = resp.content
                digits = extract_multi_digits(text, D)
                
                # Store each position separately
                for j, digit in enumerate(digits):
                    results.append({
                        "p": p,
                        "trial": trial,
                        "j": j + 1,  # 1-indexed
                        "answer": str(digit) if digit is not None else "error",
                        "status": resp.status
                    })
                
                await asyncio.sleep(config.initial_wait / config.batch_size)
    
    return pd.DataFrame(results)


def analyze_exp3_by_j(df: pd.DataFrame, j: int) -> Dict:
    """
    Analyze response curve for specific position j.
    
    Args:
        df: Results DataFrame from run_exp3
        j: Position index (1-indexed)
        
    Returns:
        Dictionary with p, r values and S metric
    """
    sub = df[df["j"] == j]
    valid = sub[sub["answer"].isin(["0", "1"])]
    
    if valid.empty:
        return {"p": [], "r": [], "S": np.nan}
    
    summary = valid.groupby("p").apply(
        lambda x: (x["answer"] == "1").mean() * 100
    ).reset_index()
    summary.columns = ["p", "r"]
    
    S = compute_S(summary["p"].values, summary["r"].values)
    
    return {
        "p": summary["p"].values,
        "r": summary["r"].values,
        "S": S
    }


def analyze_exp3_mean(df: pd.DataFrame) -> Dict:
    """
    Analyze mean response curve across all j positions.
    
    Args:
        df: Results DataFrame from run_exp3
        
    Returns:
        Dictionary with p, mean_r values and S metric
    """
    valid = df[df["answer"].isin(["0", "1"])]
    
    if valid.empty:
        return {"p": [], "r": [], "S": np.nan}
    
    # First compute r for each (p, trial, j), then average across j for each p
    summary = valid.groupby("p").apply(
        lambda x: (x["answer"] == "1").mean() * 100
    ).reset_index()
    summary.columns = ["p", "r"]
    
    S = compute_S(summary["p"].values, summary["r"].values)
    
    return {
        "p": summary["p"].values,
        "r": summary["r"].values,
        "S": S
    }


def analyze_exp3_histogram(df: pd.DataFrame, p_target: float, D: int = 10) -> Dict:
    """
    Analyze distribution of total "1" counts for specific p.
    
    Args:
        df: Results DataFrame from run_exp3
        p_target: Target p value to analyze
        D: Number of flips per trial
        
    Returns:
        Dictionary with observed and expected histograms
    """
    sub = df[df["p"] == p_target]
    valid = sub[sub["answer"].isin(["0", "1"])]
    
    if valid.empty:
        return {"observed": [], "expected": [], "chi2": np.nan, "p_value": np.nan}
    
    # Count "1"s per trial
    trial_sums = valid.groupby("trial").apply(
        lambda x: (x["answer"] == "1").sum()
    )
    
    n_trials = len(trial_sums)
    
    # Observed histogram
    observed = np.zeros(D + 1)
    for s in trial_sums:
        if 0 <= s <= D:
            observed[int(s)] += 1
    
    # Expected binomial
    expected = binom.pmf(np.arange(D + 1), D, p_target / 100) * n_trials
    
    # Chi-square test (only where expected > 0)
    mask = expected > 0
    if mask.sum() > 1:
        chi2, pval = chisquare(observed[mask], expected[mask])
    else:
        chi2, pval = np.nan, np.nan
    
    return {
        "observed": observed,
        "expected": expected,
        "chi2": chi2,
        "p_value": pval,
        "n_trials": n_trials,
        "mean_sum": trial_sums.mean(),
        "expected_mean": D * p_target / 100
    }


def plot_fig3a(df: pd.DataFrame, model_name: str, D: int = 10, save_path: Optional[str] = None):
    """
    Plot Figure 3a: Individual j response curves for D=10.
    Shows j=1, j=2, j=9, j=10
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    positions = [1, 2, D-1, D]  # j=1, 2, 9, 10 for D=10
    
    color = get_model_color(model_name)
    
    for ax, j in zip(axes.flatten(), positions):
        analysis = analyze_exp3_by_j(df, j)
        
        ax.plot(analysis["p"], analysis["r"], 'o-', color=color, markersize=4)
        ax.plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, linewidth=0.8)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("p (%)")
        ax.set_ylabel("r (%)")
        ax.set_title(f"D = {D}, j = {j}\nS = {analysis['S']:.3f}", fontsize=11)
        
        for spine in ax.spines.values():
            spine.set_color(EDGE_COLOR)
    
    plt.suptitle(f"Individual Decision Curves - {model_name}", fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    return fig


def plot_fig3b(df: pd.DataFrame, model_name: str, D: int = 10, save_path: Optional[str] = None):
    """
    Plot Figure 3b: Mean response curve across all D decisions.
    """
    import matplotlib.pyplot as plt
    
    analysis = analyze_exp3_mean(df)
    color = get_model_color(model_name)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(analysis["p"], analysis["r"], 'o-', color=color, markersize=5)
    ax.plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, linewidth=0.8)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("p (%)", fontweight='bold')
    ax.set_ylabel("r (%)", fontweight='bold')
    ax.set_title(f"Mean(r), D = {D}\n{model_name}\nS = {analysis['S']:.3f}", fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_color(EDGE_COLOR)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    return fig


def plot_fig3c(all_model_dfs: Dict[str, pd.DataFrame], p_values: List[float] = [15, 45], 
               D: int = 10, save_path: Optional[str] = None):
    """
    Plot Figure 3c: Histogram comparison for multiple models.
    
    Args:
        all_model_dfs: Dict mapping model name to DataFrame
        p_values: List of p values to plot (default: [15, 45] as in paper)
        D: Number of flips
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    n_models = len(all_model_dfs)
    n_p = len(p_values)
    
    fig, axes = plt.subplots(n_p, n_models, figsize=(4 * n_models, 4 * n_p))
    if n_models == 1:
        axes = axes.reshape(n_p, 1)
    if n_p == 1:
        axes = axes.reshape(1, n_models)
    
    x = np.arange(D + 1)
    width = 0.35
    
    for row, p in enumerate(p_values):
        for col, (model, df) in enumerate(all_model_dfs.items()):
            ax = axes[row, col]
            analysis = analyze_exp3_histogram(df, p, D)
            
            # Plot expected (gray) and observed (model color)
            ax.bar(x - width/2, analysis["expected"], width, label='Expected i.i.d.', 
                   color='gray', alpha=0.6)
            ax.bar(x + width/2, analysis["observed"], width, label='Observed',
                   color=get_model_color(model), alpha=0.8)
            
            ax.set_xlabel("Number of '1' outcomes")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{model.split('/')[-1]}\np = {p/100:.2f}")
            ax.set_xticks(x[::2])
            
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    
    plt.suptitle(f"Distribution of '1' Counts (D={D}, N={int(analysis['n_trials'])})", fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    return fig


def main():
    """Command line entry point."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run multi-flip experiment (Figure 3)")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-pro")
    parser.add_argument("--D", type=int, default=10, help="Number of flips per trial")
    parser.add_argument("--n", type=int, default=100, help="Trials per p value (use 1000 for histogram)")
    parser.add_argument("--output", type=str, default="./results/exp3")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    
    config = ExperimentConfig()
    os.makedirs(args.output, exist_ok=True)
    
    # Run experiment
    df = asyncio.run(run_exp3(
        model=args.model,
        config=config,
        D=args.D,
        n_trials=args.n
    ))
    
    # Save raw results
    raw_path = os.path.join(args.output, "raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"💾 Saved: {raw_path}")
    
    if not args.no_plot:
        # Plot Figure 3a
        plot_fig3a(df, args.model, D=args.D, 
                   save_path=os.path.join(args.output, "fig3a_individual.png"))
        
        # Plot Figure 3b
        plot_fig3b(df, args.model, D=args.D,
                   save_path=os.path.join(args.output, "fig3b_mean.png"))
        
        # For Figure 3c, need multiple models - print analysis instead
        for p in [15, 45]:
            analysis = analyze_exp3_histogram(df, p, D=args.D)
            print(f"\n📊 Histogram analysis for p={p}:")
            print(f"   Mean sum: {analysis['mean_sum']:.2f} (expected: {analysis['expected_mean']:.2f})")
            print(f"   Chi-square: {analysis['chi2']:.2f}, p-value: {analysis['p_value']:.4f}")


if __name__ == "__main__":
    main()
