"""
Experiment 1: Single Coin Flip (Figure 1)

Tests LLM's ability to simulate a biased coin flip with varying probabilities.
This is the core experiment that reveals the step-like response pattern.
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

import sys
sys.path.insert(0, '..')
from src.api_caller import APICaller
from src.parsers import extract_binary
from src.metrics import compute_S, compute_calibration_score
from src.config import ExperimentConfig, PROMPTS
from src.plotting import plot_pr_curve, plot_multi_model_pr


async def run_exp1(
    model: str,
    config: ExperimentConfig,
    p_values: Optional[List[float]] = None,
    n_trials: int = 100,
    use_system_prompt: bool = True
) -> pd.DataFrame:
    """
    Run single flip experiment.
    
    Args:
        model: Model identifier (e.g., "google/gemini-2.5-pro")
        config: Experiment configuration
        p_values: List of p values to test (default: 0, 5, 10, ..., 100)
        n_trials: Number of trials per p value
        use_system_prompt: Whether to include system prompt for strict output
        
    Returns:
        DataFrame with columns: p, trial, raw_response, answer, status
    """
    if p_values is None:
        p_values = list(range(0, 101, 5))
    
    caller = APICaller(
        api_keys=config.api_keys,
        api_url=config.api_url,
        max_retries=config.max_retries,
        timeout=config.timeout,
        initial_wait=config.initial_wait
    )
    
    # Build request list
    requests = []
    for p in p_values:
        prompt = PROMPTS["single_flip"].format(p=p, **{"100-p": 100 - p})
        
        for trial in range(n_trials):
            messages = []
            if use_system_prompt:
                messages.append({
                    "role": "system",
                    "content": PROMPTS["single_flip_system"]
                })
            messages.append({"role": "user", "content": prompt})
            
            requests.append({
                "model": model,
                "messages": messages,
                "max_tokens": 2,
                "temperature": config.temperature,
                "metadata": {"p": p, "trial": trial}
            })
    
    print(f"🧪 Running Experiment 1: Single Flip")
    print(f"   Model: {model}")
    print(f"   P values: {len(p_values)} ({min(p_values)}-{max(p_values)})")
    print(f"   Trials per p: {n_trials}")
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
                        temperature=req.get("temperature"),
                        api_key=api_key
                    )
                )
            
            responses = await asyncio.gather(*tasks)
            
            for req, resp in zip(batch, responses):
                meta = req["metadata"]
                answer = extract_binary(resp.content)
                
                results.append({
                    "p": meta["p"],
                    "trial": meta["trial"],
                    "raw_response": resp.content,
                    "answer": answer,
                    "status": resp.status
                })
            
            if i + config.batch_size < len(requests):
                await asyncio.sleep(config.initial_wait)
    
    df = pd.DataFrame(results)
    return df


def analyze_exp1(df: pd.DataFrame) -> Dict:
    """
    Analyze single flip results.
    
    Args:
        df: Results DataFrame from run_exp1
        
    Returns:
        Dictionary with p values, r values, S metric, and score
    """
    # Filter valid responses
    valid = df[df["answer"].isin(["0", "1"])].copy()
    
    if valid.empty:
        return {"error": "No valid responses"}
    
    # Compute response rate for each p
    summary = valid.groupby("p").agg({
        "answer": lambda x: (x == "1").sum() / len(x) * 100
    }).reset_index()
    summary.columns = ["p", "r"]
    
    p_values = summary["p"].values
    r_values = summary["r"].values
    
    # Compute metrics
    S = compute_S(p_values, r_values)
    score = compute_calibration_score(S)
    
    return {
        "p": p_values,
        "r": r_values,
        "S": S,
        "score": score,
        "n_valid": len(valid),
        "n_total": len(df),
        "error_rate": 1 - len(valid) / len(df)
    }


def plot_exp1(
    analysis: Dict,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot single flip results.
    
    Args:
        analysis: Output from analyze_exp1
        model_name: Model name for title
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    plot_pr_curve(
        p_values=analysis["p"],
        r_values=analysis["r"],
        model_name=model_name,
        S_value=analysis["S"],
        ax=ax
    )
    
    ax.set_title(f"Single Flip Response - {model_name}", fontweight='bold')
    ax.legend(frameon=False, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        svg_path = save_path.rsplit('.', 1)[0] + '.svg'
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"💾 Saved: {save_path} and {svg_path}")
    
    plt.show()
    return fig


# Convenience function for running from command line
def main():
    """Command line entry point."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run single flip experiment")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-pro")
    parser.add_argument("--n", type=int, default=100, help="Trials per p value")
    parser.add_argument("--output", type=str, default="./results/exp1")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    
    config = ExperimentConfig()
    os.makedirs(args.output, exist_ok=True)
    
    # Run experiment
    df = asyncio.run(run_exp1(
        model=args.model,
        config=config,
        n_trials=args.n
    ))
    
    # Save raw results
    raw_path = os.path.join(args.output, "raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"💾 Saved raw results: {raw_path}")
    
    # Analyze
    analysis = analyze_exp1(df)
    print(f"\n📊 Results for {args.model}:")
    print(f"   S = {analysis['S']:.4f}")
    print(f"   Score = {analysis['score']:.2f}/10")
    print(f"   Error rate = {analysis['error_rate']*100:.1f}%")
    
    # Plot
    if not args.no_plot:
        plot_path = os.path.join(args.output, "single_flip.png")
        plot_exp1(analysis, args.model, save_path=plot_path)


if __name__ == "__main__":
    main()
