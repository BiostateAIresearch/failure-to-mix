#!/usr/bin/env python3
"""
Main CLI for running Failure to Mix experiments.

Usage:
    python run_experiments.py --model google/gemini-2.5-pro --exp 1 2 3
    python run_experiments.py --model openai/gpt-5 --all
    python run_experiments.py --model anthropic/claude-4.5-sonnet --exp 1 --n 1000
"""
import argparse
import asyncio
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from experiments.exp1_single_flip import run_exp1, analyze_exp1, plot_exp1
from experiments.exp2_two_flips import run_exp2, analyze_exp2, plot_exp2
from experiments.exp4_ternary import run_exp4, analyze_exp4, plot_exp4
from experiments.exp5_word_bias import run_exp5, analyze_exp5, plot_exp5
from experiments.exp6_game_theory import run_exp6, analyze_exp6, plot_exp6


def get_api_key():
    """Get API key from environment or prompt user."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    
    print("⚠️  OPENROUTER_API_KEY not found in environment.")
    key = input("Please enter your OpenRouter API key: ").strip()
    if key:
        return key
    
    print("❌ No API key provided. Exiting.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM probability calibration experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run experiment 1 with default settings
    python run_experiments.py --model google/gemini-2.5-pro --exp 1
    
    # Run all experiments with 50 trials each
    python run_experiments.py --model openai/gpt-5 --all --n 50
    
    # Run experiments 1, 2, 4 without plots
    python run_experiments.py --model anthropic/claude-4.5-sonnet --exp 1 2 4 --no-plot
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model identifier (e.g., google/gemini-2.5-pro)"
    )
    parser.add_argument(
        "--exp", "-e",
        type=int,
        nargs="+",
        choices=[1, 2, 4, 5, 6],
        help="Experiment numbers to run (1, 2, 4, 5, 6)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all experiments"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of trials per condition (default: 100)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory (default: ./results)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting"
    )
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if args.all:
        experiments = [1, 2, 4, 5, 6]
    elif args.exp:
        experiments = args.exp
    else:
        print("❌ Please specify --exp or --all")
        parser.print_help()
        sys.exit(1)
    
    # Get API key
    api_key = get_api_key()
    
    # Create config
    config = ExperimentConfig(api_keys=[api_key])
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]
    output_dir = os.path.join(args.output, f"{model_short}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🔬 Failure to Mix - Experiment Runner")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Experiments: {experiments}")
    print(f"Trials per condition: {args.n}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    results_summary = {}
    
    # Run experiments
    for exp_num in experiments:
        print(f"\n{'='*50}")
        print(f"Running Experiment {exp_num}")
        print(f"{'='*50}")
        
        exp_dir = os.path.join(output_dir, f"exp{exp_num}")
        os.makedirs(exp_dir, exist_ok=True)
        
        try:
            if exp_num == 1:
                df = asyncio.run(run_exp1(args.model, config, n_trials=args.n))
                df.to_csv(os.path.join(exp_dir, "raw_results.csv"), index=False)
                analysis = analyze_exp1(df)
                results_summary["exp1"] = {"S": analysis["S"], "score": analysis["score"]}
                if not args.no_plot:
                    plot_exp1(analysis, args.model, os.path.join(exp_dir, "single_flip.png"))
                    
            elif exp_num == 2:
                df = asyncio.run(run_exp2(args.model, config, n_trials=args.n))
                df.to_csv(os.path.join(exp_dir, "raw_results.csv"), index=False)
                analysis = analyze_exp2(df)
                results_summary["exp2"] = {
                    "S1": analysis["S1"],
                    "S2": analysis["S2"],
                    "S_mean": analysis["S_mean"]
                }
                if not args.no_plot:
                    plot_exp2(analysis, args.model, os.path.join(exp_dir, "two_flips.png"))
                    
            elif exp_num == 4:
                df = asyncio.run(run_exp4(args.model, config, n_trials=args.n))
                df.to_csv(os.path.join(exp_dir, "raw_results.csv"), index=False)
                analysis = analyze_exp4(df)
                results_summary["exp4"] = {"mean_kl": analysis["mean_kl"]}
                if not args.no_plot:
                    plot_exp4(analysis, args.model, os.path.join(exp_dir, "ternary.png"))
                    
            elif exp_num == 5:
                df = asyncio.run(run_exp5(args.model, config, n_trials=args.n))
                df.to_csv(os.path.join(exp_dir, "raw_results.csv"), index=False)
                analysis = analyze_exp5(df)
                results_summary["exp5"] = {
                    "word_bias_dev": analysis["mean_word_bias_deviation"],
                    "position_bias_dev": analysis["mean_position_bias_deviation"]
                }
                if not args.no_plot:
                    plot_exp5(analysis, args.model, os.path.join(exp_dir, "word_bias.png"))
                    
            elif exp_num == 6:
                df = asyncio.run(run_exp6(args.model, config, n_trials=args.n))
                df.to_csv(os.path.join(exp_dir, "raw_results.csv"), index=False)
                analysis = analyze_exp6(df)
                mean_errors = [v["mean_error"] for v in analysis.values()]
                results_summary["exp6"] = {"mean_error": sum(mean_errors) / len(mean_errors)}
                if not args.no_plot:
                    plot_exp6(analysis, args.model, os.path.join(exp_dir, "game_theory.png"))
                    
            print(f"✅ Experiment {exp_num} completed")
            
        except Exception as e:
            print(f"❌ Experiment {exp_num} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 Results Summary")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print()
    
    for exp, metrics in results_summary.items():
        print(f"{exp}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print(f"\n✅ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
