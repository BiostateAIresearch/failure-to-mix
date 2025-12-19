"""
Experiment 6: Mixed Strategy Game Theory Problems (Figure 6)

Tests whether LLMs can implement optimal mixed strategies in game-theoretic scenarios:
1. Bioinformatics: NGS read alignment with varying read counts
2. Asymmetric Matching Pennies: Varying payoffs
3. Business Positioning: Luxury vs Affordable with different market conditions
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
from src.parsers import extract_game_choice
from src.config import ExperimentConfig, PROMPTS
from src.plotting import plot_pr_curve


# Game scenarios (matching paper Figure 6)
GAME_SCENARIOS = {
    "bioinformatics": {
        "description": "NGS read alignment with varying A/B read counts (Fig 6b)",
        "options": ["A", "B"],
        # Paper: varies ratio of uniquely mapped reads
        "conditions": [
            {"a_count": 100, "b_count": 2400, "optimal_A": 4.0},   # p=4%
            {"a_count": 250, "b_count": 2250, "optimal_A": 10.0},  # p=10%
            {"a_count": 1000, "b_count": 1500, "optimal_A": 40.0}, # p=40% (paper example)
            {"a_count": 1250, "b_count": 1250, "optimal_A": 50.0}, # p=50%
            {"a_count": 1500, "b_count": 1000, "optimal_A": 60.0}, # p=60%
            {"a_count": 2475, "b_count": 25, "optimal_A": 99.0},   # p=99%
        ]
    },
    "matching_pennies": {
        "description": "Asymmetric Matching Pennies with varying payoffs (Fig 6d)",
        "options": ["Heads", "Tails"],
        # Paper: HH=+$1.50, TT=+$1.00, HT=-$1.50, TH=-$1.00
        # Optimal p(Heads) = tt / (hh + tt) for matcher
        # We vary the payoffs to get different optimal p values
        "conditions": [
            {"hh": "1.50", "tt": "1.00", "ht": "1.50", "th": "1.00", "optimal_H": 40.0},  # 1.0/(1.5+1.0)=40%
            {"hh": "1.00", "tt": "1.00", "ht": "1.00", "th": "1.00", "optimal_H": 50.0},  # 50%
            {"hh": "1.00", "tt": "1.50", "ht": "1.00", "th": "1.50", "optimal_H": 60.0},  # 1.5/(1.0+1.5)=60%
            {"hh": "0.50", "tt": "1.50", "ht": "0.50", "th": "1.50", "optimal_H": 75.0},  # 1.5/(0.5+1.5)=75%
            {"hh": "1.50", "tt": "0.50", "ht": "1.50", "th": "0.50", "optimal_H": 25.0},  # 0.5/(1.5+0.5)=25%
        ]
    },
    "business": {
        "description": "Business positioning Luxury vs Affordable (Fig 6efg)",
        "options": ["Luxury", "Affordable"],
        # Paper: varies payoffs to get different optimal p(Luxury) values
        # From paper Figure 6f: different affordable_payoff values map to optimal p
        "conditions": [
            {"luxury_win": "10M", "affordable_win": "0.1M", "optimal_L": 1.0},
            {"luxury_win": "10M", "affordable_win": "1M", "optimal_L": 10.0},
            {"luxury_win": "10M", "affordable_win": "6.7M", "optimal_L": 40.0},
            {"luxury_win": "10M", "affordable_win": "10M", "optimal_L": 50.0},
            {"luxury_win": "10M", "affordable_win": "15M", "optimal_L": 60.0},
            {"luxury_win": "10M", "affordable_win": "90M", "optimal_L": 90.0},
            {"luxury_win": "10M", "affordable_win": "990M", "optimal_L": 99.0},
        ]
    }
}


def create_game_prompt(scenario: str, condition: Dict) -> Tuple[str, List[str]]:
    """
    Create prompt for a game theory scenario.
    
    Args:
        scenario: Scenario name
        condition: Condition parameters
        
    Returns:
        Tuple of (prompt, valid_options)
    """
    if scenario == "bioinformatics":
        prompt = PROMPTS["bioinformatics"].format(
            a_count=condition["a_count"],
            b_count=condition["b_count"]
        )
        options = ["A", "B"]
    elif scenario == "matching_pennies":
        prompt = PROMPTS["matching_pennies"].format(
            hh=condition["hh"],
            tt=condition["tt"],
            ht=condition["ht"],
            th=condition["th"]
        )
        options = ["Heads", "Tails"]
    elif scenario == "business":
        prompt = PROMPTS["business"].format(
            luxury_win=condition["luxury_win"],
            affordable_win=condition["affordable_win"]
        )
        options = ["Luxury", "Affordable"]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return prompt, options


async def run_exp6(
    model: str,
    config: ExperimentConfig,
    scenarios: Optional[List[str]] = None,
    n_trials: int = 100
) -> pd.DataFrame:
    """
    Run game theory experiment.
    
    Args:
        model: Model identifier
        config: Experiment configuration
        scenarios: List of scenario names (default: all)
        n_trials: Number of trials per condition
        
    Returns:
        DataFrame with game choices and optimal comparisons
    """
    if scenarios is None:
        scenarios = list(GAME_SCENARIOS.keys())
    
    caller = APICaller(
        api_keys=config.api_keys,
        api_url=config.api_url,
        max_retries=config.max_retries,
        timeout=config.timeout
    )
    
    # Build requests
    requests = []
    for scenario_name in scenarios:
        scenario = GAME_SCENARIOS[scenario_name]
        
        for condition in scenario["conditions"]:
            prompt, options = create_game_prompt(scenario_name, condition)
            
            for trial in range(n_trials):
                requests.append({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "metadata": {
                        "scenario": scenario_name,
                        "condition": condition,
                        "options": options,
                        "trial": trial
                    }
                })
    
    print(f"🧪 Running Experiment 6: Game Theory")
    print(f"   Model: {model}")
    print(f"   Scenarios: {scenarios}")
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
                answer = extract_game_choice(resp.content, meta["options"])
                
                # Get optimal probability for first option
                condition = meta["condition"]
                if meta["scenario"] == "bioinformatics":
                    optimal = condition["optimal_A"]
                    chose_first = answer == "A"
                elif meta["scenario"] == "matching_pennies":
                    optimal = condition["optimal_H"]
                    chose_first = answer == "Heads"
                elif meta["scenario"] == "business":
                    optimal = condition["optimal_L"]
                    chose_first = answer == "Luxury"
                else:
                    optimal = 50.0
                    chose_first = False
                
                results.append({
                    "scenario": meta["scenario"],
                    "condition_str": str(condition),
                    "optimal_first": optimal,  # Already in percentage
                    "trial": meta["trial"],
                    "raw_response": resp.content,
                    "answer": answer,
                    "chose_first": chose_first if answer != "error" else None,
                    "status": resp.status
                })
            
            if i + config.batch_size < len(requests):
                await asyncio.sleep(config.initial_wait)
    
    return pd.DataFrame(results)


def analyze_exp6(df: pd.DataFrame) -> Dict:
    """
    Analyze game theory results.
    
    Args:
        df: Results DataFrame from run_exp6
        
    Returns:
        Dictionary with observed vs optimal rates for each scenario
    """
    valid = df[df["answer"] != "error"].copy()
    
    if valid.empty:
        return {"error": "No valid responses"}
    
    results = {}
    for scenario in valid["scenario"].unique():
        scenario_data = valid[valid["scenario"] == scenario]
        
        # Group by condition
        summary = scenario_data.groupby(["optimal_first"]).agg({
            "chose_first": "mean"
        }).reset_index()
        summary.columns = ["optimal", "observed"]
        summary["observed"] = summary["observed"] * 100  # Convert to percentage
        
        # Calculate error
        summary["error"] = np.abs(summary["observed"] - summary["optimal"])
        
        results[scenario] = {
            "optimal": summary["optimal"].values,
            "observed": summary["observed"].values,
            "mean_error": summary["error"].mean(),
            "max_error": summary["error"].max()
        }
    
    return results


def plot_exp6(
    analysis: Dict,
    model_name: str,
    save_path: Optional[str] = None
):
    """
    Plot game theory results.
    
    Args:
        analysis: Output from analyze_exp6
        model_name: Model name for title
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    n_scenarios = len(analysis)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4))
    if n_scenarios == 1:
        axes = [axes]
    
    for ax, (scenario, data) in zip(axes, analysis.items()):
        # Sort by optimal
        order = np.argsort(data["optimal"])
        optimal = data["optimal"][order]
        observed = data["observed"][order]
        
        # Plot
        ax.plot(optimal, observed, 'o-', linewidth=1.5, markersize=8)
        ax.plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, label='Optimal')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Optimal Rate (%)")
        ax.set_ylabel("Observed Rate (%)")
        ax.set_title(f"{scenario.replace('_', ' ').title()}\nMean Error: {data['mean_error']:.1f}%")
        ax.legend()
    
    fig.suptitle(f"Game Theory Results - {model_name}", fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        svg_path = save_path.rsplit('.', 1)[0] + '.svg'
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()
    
    print(f"\n📊 Game Theory Analysis for {model_name}:")
    for scenario, data in analysis.items():
        print(f"   {scenario}: Mean error = {data['mean_error']:.1f}%, Max error = {data['max_error']:.1f}%")
    
    return fig


def main():
    """Command line entry point."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run game theory experiment")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-pro")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", type=str, default="./results/exp6")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    
    config = ExperimentConfig()
    os.makedirs(args.output, exist_ok=True)
    
    df = asyncio.run(run_exp6(
        model=args.model,
        config=config,
        n_trials=args.n
    ))
    
    raw_path = os.path.join(args.output, "raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"💾 Saved: {raw_path}")
    
    analysis = analyze_exp6(df)
    
    if not args.no_plot:
        plot_path = os.path.join(args.output, "game_theory.png")
        plot_exp6(analysis, args.model, save_path=plot_path)


if __name__ == "__main__":
    main()
