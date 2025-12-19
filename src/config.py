"""
Configuration settings for LLM probability experiments.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Model color scheme for consistent plotting (matching paper figures)
MODEL_COLORS: Dict[str, str] = {
    "google/gemini-2.5-pro": "#1f77b4",       # Blue
    "openai/gpt-5": "#2ca02c",                 # Green
    "openai/gpt-5-nano": "#d62a28",            # Red (corrected)
    "anthropic/claude-4.5-sonnet": "#8c564b",  # Brown-purple
    "anthropic/sonnet-4.5": "#8c564b",         # Brown-purple (alias)
    "moonshotai/kimi-k2-0905": "#e377c2",      # Pink
    "qwen/qwen3-vl-8b-instruct": "#662d91",    # Purple (corrected)
    "qwen/qwen3": "#662d91",                   # Purple (alias)
    "x-ai/grok-4-fast": "#c2b59b",             # Tan/beige (corrected)
    "deepseek/deepseek-v3.2": "#9467bd",       # Purple
}

# Edge/frame color for plots
EDGE_COLOR = "#999b9e"

@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""
    
    # API settings
    api_keys: List[str] = field(default_factory=list)
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    
    # Request settings
    batch_size: int = 20
    initial_wait: float = 2.0
    max_retries: int = 3
    timeout: int = 120
    max_tokens: Optional[int] = 2
    temperature: Optional[float] = None
    
    # Experiment settings
    n_trials: int = 100
    p_values: List[float] = field(default_factory=lambda: list(range(0, 101, 5)))
    
    # Output settings
    save_path: str = "./results"
    upload_to_drive: bool = False
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if not self.api_keys:
            # Try single key first
            key = os.environ.get("OPENROUTER_API_KEY")
            if key:
                self.api_keys = [key]
            else:
                # Try multiple keys
                for i in range(10):
                    key = os.environ.get(f"OPENROUTER_API_KEY_{i}")
                    if key:
                        self.api_keys.append(key)
        
        os.makedirs(self.save_path, exist_ok=True)


# Prompt templates
PROMPTS = {
    "single_flip": (
        'I\'d like you to flip a biased coin that returns "1" {p}% of the time '
        'and "0" {100-p}% of the time. Please return a 1 token response simulating a single flip.'
    ),
    "single_flip_system": (
        "Return only one token: 0 or 1. Do not include any other text."
    ),
    "ternary_draw": (
        'I\'d like to draw from a biased deck that returns "2" {q}% of the time, '
        '"1" {p}% of the time and "0" {remainder}% of the time. '
        'Please return a 1 token response simulating a single draw.'
    ),
    "multi_flip": (
        'I\'d like you to flip a biased coin that returns "1" {p}% of the time '
        'and "0" {100-p}% of the time. Please return a {D} token response simulating {D} flips. '
        'Please return exactly {D} tokens (0 or 1) separated by a single space.'
    ),
    "word_flip": (
        'I\'d like you to flip a biased coin that returns "{word1}" {p}% of the time '
        'and "{word2}" {100-p}% of the time. Please return a 1 token response simulating a single flip.'
    ),
    "two_flip_followup": "Please do it one more time.",
    
    # Game theory prompts (exact from paper)
    "bioinformatics": (
        'You are assisting with a bioinformatics task, specifically aligning NGS short reads to genes '
        'for RNA expression analysis. You are given a read R that aligns equally well to both genes A and B, '
        'because A and B are high homology. So far, {a_count} reads have been uniquely mapped to gene A, '
        'and {b_count} reads have been uniquely mapped to gene B. You must choose to map read R to either '
        'gene A or gene B as part of your task, and you wish the final results to maximally mirror the true '
        'expression levels of A and B. You MUST pick A or B for this independent task at the end of your response.'
    ),
    "matching_pennies": (
        'You are playing a one-time variant game of Matching Pennies against another participant, and you are the '
        'Matcher. You and your opponent will simultaneously pick Heads or Tails. If you match your opponent\'s '
        'choice as Heads, you win ${hh} and your opponent loses ${hh}. If you match your opponent\'s choice as '
        'Tails, you win ${tt} and your opponent loses ${tt}. If you pick Heads and your opponent chooses Tails, '
        'you lose ${ht}, and your opponent wins ${ht}. If you pick Tails and your opponent Heads, you lose ${th}, '
        'and your opponent wins ${th}. What is your choice? You MUST pick either Heads or Tails for this one-time '
        'game at the end of your response.'
    ),
    "business": (
        'You are playing a one-time game against another participant. You play the role of a startup CEO in a '
        'particular niche market, in which there is one dominant incumbent controlled by the other participant. '
        'The incumbent\'s main revenue streams are from other markets, so they plan to sell at cost to push you '
        'out of business. You must choose whether to design and market your new product as a Luxury Product or '
        'an Affordable Product. Simultaneously, the incumbent will also decide whether they will make a Luxury '
        'Product or an Affordable Product. If you choose Luxury and the incumbent chooses Affordable, then your '
        'payoff is ${luxury_win}. If you choose Affordable and the incumbent chooses Luxury, then your payoff is '
        '${affordable_win}. If the incumbent matches your product positioning choice, then your payoff is $0. '
        'Please choose your strategy; you MUST pick one strategy at the end of your response for this game.'
    ),
}


def get_model_color(model_name: str) -> str:
    """Get consistent color for a model."""
    # Check exact match first
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    
    # Check partial match
    model_lower = model_name.lower()
    for key, color in MODEL_COLORS.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return color
    
    # Check provider
    if "gemini" in model_lower:
        return MODEL_COLORS["google/gemini-2.5-pro"]
    if "gpt" in model_lower or "openai" in model_lower:
        if "nano" in model_lower:
            return MODEL_COLORS["openai/gpt-5-nano"]
        return MODEL_COLORS["openai/gpt-5"]
    if "claude" in model_lower or "anthropic" in model_lower or "sonnet" in model_lower:
        return MODEL_COLORS["anthropic/claude-4.5-sonnet"]
    if "kimi" in model_lower:
        return MODEL_COLORS["moonshotai/kimi-k2-0905"]
    if "qwen" in model_lower:
        return MODEL_COLORS["qwen/qwen3"]
    if "grok" in model_lower:
        return MODEL_COLORS["x-ai/grok-4-fast"]
    if "deepseek" in model_lower:
        return MODEL_COLORS["deepseek/deepseek-v3.2"]
    
    # Default gray
    return "#7f7f7f"
