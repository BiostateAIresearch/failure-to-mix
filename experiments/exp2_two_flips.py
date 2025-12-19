"""
Experiment 2: Two Sequential Flips with Conversation Context (Figure 2)

Tests whether LLMs can maintain probability calibration across multiple turns
and whether they compensate for previous deviations.
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
from src.metrics import compute_S, analyze_compensation_effect
from src.config import ExperimentConfig, PROMPTS
from src.plotting import plot_two_flip_comparison


async def run_exp2(
    model: str,
    config: ExperimentConfig,
    p_values: Optional[List[float]] = None,
    n_trials: int = 100
) -> pd.DataFrame:
    """
    Run two-flip sequential experiment.
    
    Turn 1: Initial flip request
    Turn 2: "Please do it one more time" with conversation history
    
    Args:
        model: Model identifier
        config: Experiment configuration
        p_values: List of p values to test
        n_trials: Number of trials per p value
        
    Returns:
        DataFrame with columns: p, trial, turn, raw_response, answer, status
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
    
    print(f"🧪 Running Experiment 2: Two Sequential Flips")
    print(f"   Model: {model}")
    print(f"   P values: {len(p_values)}")
    print(f"   Trials per p: {n_trials}")
    
    results = []
    total = len(p_values) * n_trials
    
    async with aiohttp.ClientSession() as session:
        with tqdm(total=total, desc="📊 Progress") as pbar:
            for p in p_values:
                prompt1 = PROMPTS["single_flip"].format(p=p, **{"100-p": 100 - p})
                prompt2 = PROMPTS["two_flip_followup"]
                
                for trial in range(n_trials):
                    # Turn 1
                    messages = [
                        {"role": "system", "content": PROMPTS["single_flip_system"]},
                        {"role": "user", "content": prompt1}
                    ]
                    
                    api_key = config.api_keys[(p * n_trials + trial) % len(config.api_keys)]
                    
                    resp1 = await caller.call(
                        session=session,
                        model=model,
                        messages=messages,
                        max_tokens=2,
                        api_key=api_key
                    )
                    
                    answer1 = extract_binary(resp1.content)
                    results.append({
                        "p": p,
                        "trial": trial,
                        "turn": 1,
                        "raw_response": resp1.content,
                        "answer": answer1,
                        "status": resp1.status
                    })
                    
                    # Turn 2 with conversation context
                    messages.append({"role": "assistant", "content": resp1.content or answer1})
                    messages.append({"role": "user", "content": prompt2})
                    
                    resp2 = await caller.call(
                        session=session,
                        model=model,
                        messages=messages,
                        max_tokens=2,
                        api_key=api_key
                    )
                    
                    answer2 = extract_binary(resp2.content)
                    results.append({
                        "p": p,
                        "trial": trial,
                        "turn": 2,
                        "raw_response": resp2.content,
                        "answer": answer2,
                        "status": resp2.status
                    })
                    
                    pbar.update(1)
                    
                    # Rate limiting
                    await asyncio.sleep(config.initial_wait / config.batch_size)
    
    df = pd.DataFrame(results)
    return df


def analyze_exp2(df: pd.DataFrame) -> Dict:
    """
    Analyze two-flip results.
    
    Args:
        df: Results DataFrame from run_exp2
        
    Returns:
        Dictionary with S values for each turn and compensation analysis
    """
    valid = df[df["answer"].isin(["0", "1"])].copy()
    
    if valid.empty:
        return {"error": "No valid responses"}
    
    # Separate turns
    t1 = valid[valid["turn"] == 1]
    t2 = valid[valid["turn"] == 2]
    
    # Compute r for each turn and p
    def compute_r(data):
        return data.groupby("p").agg({
            "answer": lambda x: (x == "1").sum() / len(x) * 100
        }).reset_index()
    
    r1 = compute_r(t1)
    r2 = compute_r(t2)
    
    # Merge on p
    merged = r1.merge(r2, on="p", suffixes=("_t1", "_t2"))
    merged["r_mean"] = (merged["answer_t1"] + merged["answer_t2"]) / 2
    
    p_values = merged["p"].values
    s1_values = merged["answer_t1"].values
    s2_values = merged["answer_t2"].values
    s_mean = merged["r_mean"].values
    
    # Compute S for each
    S1 = compute_S(p_values, s1_values)
    S2 = compute_S(p_values, s2_values)
    S_mean = compute_S(p_values, s_mean)
    
    # Compensation analysis
    compensation = analyze_compensation_effect(
        s1_values / 100, s2_values / 100, p_values
    )
    
    return {
        "p": p_values,
        "r_t1": s1_values,
        "r_t2": s2_values,
        "r_mean": s_mean,
        "S1": S1,
        "S2": S2,
        "S_mean": S_mean,
        "compensation": compensation
    }


def plot_exp2(
    analysis: Dict,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot two-flip results.
    
    Args:
        analysis: Output from analyze_exp2
        model_name: Model name for title
        save_path: Path to save figure
    """
    fig = plot_two_flip_comparison(
        p_values=analysis["p"],
        s1_values=analysis["r_t1"],
        s2_values=analysis["r_t2"],
        s_mean_values=analysis["r_mean"],
        model_name=model_name,
        save_path=save_path
    )
    
    print(f"\n📊 Two-Flip Analysis for {model_name}:")
    print(f"   S (Turn 1) = {analysis['S1']:.4f}")
    print(f"   S (Turn 2) = {analysis['S2']:.4f}")
    print(f"   S (Mean)   = {analysis['S_mean']:.4f}")
    print(f"   Compensation correlation = {analysis['compensation']['compensation_correlation']:.4f}")
    
    return fig


def main():
    """Command line entry point."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run two-flip experiment")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-pro")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", type=str, default="./results/exp2")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    
    config = ExperimentConfig()
    os.makedirs(args.output, exist_ok=True)
    
    df = asyncio.run(run_exp2(
        model=args.model,
        config=config,
        n_trials=args.n
    ))
    
    raw_path = os.path.join(args.output, "raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"💾 Saved: {raw_path}")
    
    analysis = analyze_exp2(df)
    
    if not args.no_plot:
        plot_path = os.path.join(args.output, "two_flips.png")
        plot_exp2(analysis, args.model, save_path=plot_path)


if __name__ == "__main__":
    main()
