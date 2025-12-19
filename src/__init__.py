"""
Failure to Mix: LLM Probability Calibration Library

This library provides tools for evaluating how well large language models
can execute probabilistic tasks.
"""

from .config import ExperimentConfig, PROMPTS, get_model_color, MODEL_COLORS
from .api_caller import APICaller, APIResponse
from .metrics import (
    compute_S,
    compute_calibration_score,
    compute_kl_divergence,
    compute_chi_square,
    compute_binomial_expected,
    analyze_response_curve,
    analyze_compensation_effect
)
from .parsers import (
    extract_binary,
    extract_ternary,
    extract_heads_tails,
    extract_word_choice,
    extract_multi_digits,
    extract_game_choice,
    extract_numeric,
    normalize_response
)
from .plotting import (
    setup_plot_style,
    plot_pr_curve,
    plot_multi_model_pr,
    plot_histogram_comparison,
    plot_ternary_results,
    plot_word_bias,
    plot_two_flip_comparison
)
from .drive_uploader import DriveUploader

__version__ = "1.0.0"
__author__ = "Biostate AI"

__all__ = [
    # Config
    "ExperimentConfig",
    "PROMPTS",
    "get_model_color",
    "MODEL_COLORS",
    
    # API
    "APICaller",
    "APIResponse",
    
    # Metrics
    "compute_S",
    "compute_calibration_score",
    "compute_kl_divergence",
    "compute_chi_square",
    "compute_binomial_expected",
    "analyze_response_curve",
    "analyze_compensation_effect",
    
    # Parsers
    "extract_binary",
    "extract_ternary",
    "extract_heads_tails",
    "extract_word_choice",
    "extract_multi_digits",
    "extract_game_choice",
    "extract_numeric",
    "normalize_response",
    
    # Plotting
    "setup_plot_style",
    "plot_pr_curve",
    "plot_multi_model_pr",
    "plot_histogram_comparison",
    "plot_ternary_results",
    "plot_word_bias",
    "plot_two_flip_comparison",
    
    # Drive
    "DriveUploader",
]
