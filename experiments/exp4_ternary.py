"""
Experiment 4: Ternary Distribution (Figure 4)

Tests two-parameter discrete distribution (0/1/2) with fixed p(1) and varying p(2).
Examines if probability mass is correctly distributed across three outcomes.
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
from src.parsers import extract_ternary
from src.metrics import compute_kl_divergence
from src.config import ExperimentConfig, PROMPTS
from src.plotting import plot_ternary_results


async def run_exp4(
    model: str,
    config: ExperimentConfig,
    p1_fixed: float = 40,
    q_values: Optional[List[float]] = None,
    n_trials: int = 100
) -> pd.DataFrame:
    """
    Run ternary distribution experiment.
    
    Args:
        model: Model identifier
        config: Experiment configuration
        p1_fixed: Fixed probability for outcome "1" (%)
        q_values: List of p(2) values to test (%)
        n_trials: Number of trials per q value
        
    Returns:
        DataFrame with columns: q, trial, raw_response, answer, status
    """
    if q_values is None:
        q_values = list(range(0, 61, 5))  # 0 to 60 in steps of 5
    
    caller = APICaller(
        api_keys=config.api_keys,
        api_url=config.api_url,
        max_retries=config.max_retries,
        timeout=config.timeout
    )
    
    # Build requests
    requests = []
    for q in q_values:
        remainder = 100 - p1_fixed - q
        if remainder < 0:
            continue  # Skip invalid combinations
        
        prompt = PROMPTS["ternary_draw"].format(
            p=p1_fixed,
            q=q,
            remainder=remainder
        )
        
        for trial in range(n_trials):
            requests.append({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2,
                "metadata": {"q": q, "p1": p1_fixed, "p0": remainder, "trial": trial}
            })
    
    print(f"🧪 Running Experiment 4: Ternary Distribution")
    print(f"   Model: {model}")
    print(f"   p(1) fixed at: {p1_fixed}%")
    print(f"   q = p(2) values: {q_values}")
    print(f"   Trials per q: {n_trials}")
    
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
                answer = extract_ternary(resp.content)
                
                results.append({
                    "q": meta["q"],
                    "p1_expected": meta["p1"],
                    "p0_expected": meta["p0"],
                    "trial": meta["trial"],
                    "raw_response": resp.content,
                    "answer": answer,
                    "status": resp.status
                })
            
            if i + config.batch_size < len(requests):
                await asyncio.sleep(config.initial_wait)
    
    return pd.DataFrame(results)


def analyze_exp4(df: pd.DataFrame, p1_fixed: float = 40) -> Dict:
    """
    Analyze ternary distribution results.
    
    Args:
        df: Results DataFrame from run_exp4
        p1_fixed: The fixed p(1) value used
        
    Returns:
        Dictionary with observed rates and KL divergences
    """
    valid = df[df["answer"].isin(["0", "1", "2"])].copy()
    
    if valid.empty:
        return {"error": "No valid responses"}
    
    # Compute rates for each q value
    summary = valid.groupby("q").apply(
        lambda x: pd.Series({
            "r0": (x["answer"] == "0").mean() * 100,
            "r1": (x["answer"] == "1").mean() * 100,
            "r2": (x["answer"] == "2").mean() * 100,
            "n": len(x)
        })
    ).reset_index()
    
    # Expected values
    summary["expected_r0"] = 100 - p1_fixed - summary["q"]
    summary["expected_r1"] = p1_fixed
    summary["expected_r2"] = summary["q"]
    
    # KL divergence for each q
    kl_values = []
    for _, row in summary.iterrows():
        observed = np.array([row["r0"], row["r1"], row["r2"]]) / 100
        expected = np.array([row["expected_r0"], row["expected_r1"], row["expected_r2"]]) / 100
        kl = compute_kl_divergence(observed, expected)
        kl_values.append(kl)
    
    summary["kl_divergence"] = kl_values
    
    return {
        "q": summary["q"].values,
        "r0": summary["r0"].values,
        "r1": summary["r1"].values,
        "r2": summary["r2"].values,
        "expected_r0": summary["expected_r0"].values,
        "expected_r1": summary["expected_r1"].values,
        "expected_r2": summary["expected_r2"].values,
        "kl_divergence": summary["kl_divergence"].values,
        "mean_kl": np.mean(kl_values),
        "p1_fixed": p1_fixed
    }


def plot_exp4(
    analysis: Dict,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot ternary distribution results.
    
    Args:
        analysis: Output from analyze_exp4
        model_name: Model name for title
        save_path: Path to save figure
    """
    fig = plot_ternary_results(
        q_values=analysis["q"],
        r0_values=analysis["r0"],
        r1_values=analysis["r1"],
        r2_values=analysis["r2"],
        p1_fixed=analysis["p1_fixed"],
        model_name=model_name,
        save_path=save_path
    )
    
    print(f"\n📊 Ternary Analysis for {model_name}:")
    print(f"   Mean KL divergence = {analysis['mean_kl']:.4f}")
    
    return fig


def main():
    """Command line entry point."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run ternary distribution experiment")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-pro")
    parser.add_argument("--p1", type=float, default=40, help="Fixed p(1) value (%)")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", type=str, default="./results/exp4")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    
    config = ExperimentConfig()
    os.makedirs(args.output, exist_ok=True)
    
    df = asyncio.run(run_exp4(
        model=args.model,
        config=config,
        p1_fixed=args.p1,
        n_trials=args.n
    ))
    
    raw_path = os.path.join(args.output, "raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"💾 Saved: {raw_path}")
    
    analysis = analyze_exp4(df, p1_fixed=args.p1)
    
    if not args.no_plot:
        plot_path = os.path.join(args.output, "ternary.png")
        plot_exp4(analysis, args.model, save_path=plot_path)


if __name__ == "__main__":
    main()
