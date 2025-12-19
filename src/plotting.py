"""
Plotting utilities for visualizing LLM probability experiments.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .config import get_model_color, EDGE_COLOR


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.figsize': (8, 6),
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.alpha': 0.5,
    })


def plot_pr_curve(
    p_values: np.ndarray,
    r_values: np.ndarray,
    model_name: str,
    S_value: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    show_ideal: bool = True,
    x_range: Tuple[float, float] = (0, 100),
    **kwargs
) -> plt.Axes:
    """
    Plot p-r response curve for a single model.
    
    Args:
        p_values: Target probabilities (%)
        r_values: Observed response rates (%)
        model_name: Model name for label
        S_value: Optional S metric to show in legend
        ax: Matplotlib axes (created if None)
        color: Line color (auto-selected if None)
        show_ideal: Whether to show ideal y=x line
        x_range: X-axis range
        **kwargs: Additional plot arguments
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if color is None:
        color = get_model_color(model_name)
    
    # Create label
    label = model_name
    if S_value is not None:
        label = f"{model_name} (S={S_value:.3f})"
    
    # Sort by p
    order = np.argsort(p_values)
    p_sorted = np.array(p_values)[order]
    r_sorted = np.array(r_values)[order]
    
    # Plot line
    ax.plot(p_sorted, r_sorted, color=color, linewidth=1.5, zorder=2, **kwargs)
    
    # Plot points
    ax.scatter(
        p_sorted, r_sorted,
        s=45, facecolors=color, edgecolors='black',
        linewidths=0.6, zorder=3, label=label
    )
    
    # Ideal line
    if show_ideal:
        ax.plot([0, 100], [0, 100], '--', color='gray', linewidth=0.8, alpha=0.5, zorder=1)
    
    # Styling
    ax.set_xlim(x_range)
    ax.set_ylim(0, 100)
    ax.set_xlabel("p (%)", fontweight='bold')
    ax.set_ylabel("r (%)", fontweight='bold')
    ax.axvline(x=50, color=EDGE_COLOR, linewidth=0.9, alpha=0.3, zorder=0)
    
    for spine in ax.spines.values():
        spine.set_color(EDGE_COLOR)
        spine.set_linewidth(0.9)
    
    ax.tick_params(color=EDGE_COLOR, labelcolor=EDGE_COLOR)
    
    return ax


def plot_multi_model_pr(
    results: Dict[str, Dict[str, np.ndarray]],
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    x_range: Tuple[float, float] = (0, 100),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot p-r curves for multiple models.
    
    Args:
        results: Dict mapping model names to {'p': array, 'r': array, 'S': float}
        title: Plot title
        figsize: Figure size
        x_range: X-axis range
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, data in results.items():
        plot_pr_curve(
            p_values=data['p'],
            r_values=data['r'],
            model_name=model_name,
            S_value=data.get('S'),
            ax=ax,
            x_range=x_range
        )
    
    ax.legend(frameon=False, loc='lower right')
    
    if title:
        ax.set_title(title, fontweight='bold', color=EDGE_COLOR)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # Also save SVG
        svg_path = str(save_path).rsplit('.', 1)[0] + '.svg'
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"💾 Saved: {save_path} and {svg_path}")
    
    return fig


def plot_histogram_comparison(
    observed: np.ndarray,
    expected: np.ndarray,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: str = "Value",
    ylabel: str = "Count",
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram comparing observed vs expected distributions.
    
    Args:
        observed: Observed counts
        expected: Expected counts
        labels: Labels for x-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(observed))
    width = 0.35
    
    if labels is None:
        labels = [str(i) for i in x]
    
    # Plot bars
    ax.bar(x - width/2, observed, width, label='Observed', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, expected, width, label='Expected', color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    if title:
        ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        svg_path = str(save_path).rsplit('.', 1)[0] + '.svg'
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"💾 Saved: {save_path} and {svg_path}")
    
    return fig


def plot_ternary_results(
    q_values: np.ndarray,
    r0_values: np.ndarray,
    r1_values: np.ndarray,
    r2_values: np.ndarray,
    p1_fixed: float = 40,
    model_name: str = "",
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ternary distribution results (Figure 4).
    
    Args:
        q_values: p(2) probabilities (%)
        r0_values: Observed r(0) (%)
        r1_values: Observed r(1) (%)
        r2_values: Observed r(2) (%)
        p1_fixed: Fixed p(1) value (%)
        model_name: Model name for title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Expected values
    expected_r0 = 100 - p1_fixed - q_values
    expected_r1 = np.full_like(q_values, p1_fixed)
    expected_r2 = q_values
    
    color = get_model_color(model_name) if model_name else '#1f77b4'
    
    # Plot r0
    axes[0].plot(q_values, r0_values, 'o-', color=color, label='Observed')
    axes[0].plot(q_values, expected_r0, '--', color='gray', label='Expected')
    axes[0].set_xlabel("q = p(2) (%)")
    axes[0].set_ylabel("r(0) (%)")
    axes[0].set_title("Output rate for '0'")
    axes[0].legend()
    
    # Plot r1
    axes[1].plot(q_values, r1_values, 'o-', color=color, label='Observed')
    axes[1].axhline(y=p1_fixed, color='gray', linestyle='--', label='Expected')
    axes[1].set_xlabel("q = p(2) (%)")
    axes[1].set_ylabel("r(1) (%)")
    axes[1].set_title(f"Output rate for '1' (p={p1_fixed}%)")
    axes[1].legend()
    
    # Plot r2
    axes[2].plot(q_values, r2_values, 'o-', color=color, label='Observed')
    axes[2].plot(q_values, expected_r2, '--', color='gray', label='Expected')
    axes[2].set_xlabel("q = p(2) (%)")
    axes[2].set_ylabel("r(2) (%)")
    axes[2].set_title("Output rate for '2'")
    axes[2].legend()
    
    if model_name:
        fig.suptitle(f"Ternary Distribution Results - {model_name}", fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        svg_path = str(save_path).rsplit('.', 1)[0] + '.svg'
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"💾 Saved: {save_path} and {svg_path}")
    
    return fig


def plot_word_bias(
    word_pairs: List[Tuple[str, str]],
    word_bias: List[float],
    position_bias: List[float],
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot word choice and position bias results (Figure 5).
    
    Args:
        word_pairs: List of (word1, word2) tuples
        word_bias: Bias toward word1 for each pair
        position_bias: Bias toward first position for each pair
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Word bias bar chart
    x = np.arange(len(word_pairs))
    labels = [f"{w1}/{w2}" for w1, w2 in word_pairs]
    
    colors = ['#2ca02c' if b > 0.5 else '#d62728' for b in word_bias]
    axes[0].bar(x, [b - 0.5 for b in word_bias], color=colors, alpha=0.7)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_xlabel("Word Pair")
    axes[0].set_ylabel("Word Bias (relative to 0.5)")
    axes[0].set_title("Word Choice Bias")
    
    # Position bias scatter
    axes[1].scatter(x, position_bias, s=100, c='#1f77b4', alpha=0.7)
    axes[1].axhline(y=0.5, color='gray', linestyle='--')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_xlabel("Word Pair")
    axes[1].set_ylabel("Position Bias (toward first)")
    axes[1].set_title("Position Bias")
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        svg_path = str(save_path).rsplit('.', 1)[0] + '.svg'
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"💾 Saved: {save_path} and {svg_path}")
    
    return fig


def plot_two_flip_comparison(
    p_values: np.ndarray,
    s1_values: np.ndarray,
    s2_values: np.ndarray,
    s_mean_values: np.ndarray,
    model_name: str = "",
    figsize: Tuple[float, float] = (15, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot two-flip experiment results (Figure 2).
    
    Args:
        p_values: Target probabilities (%)
        s1_values: First flip response rates (%)
        s2_values: Second flip response rates (%)
        s_mean_values: Mean of both flips (%)
        model_name: Model name for title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    color = get_model_color(model_name) if model_name else '#1f77b4'
    
    # Plot each panel
    for i, (values, title) in enumerate([
        (s1_values, "Turn 1"),
        (s2_values, "Turn 2"),
        (s_mean_values, "Mean")
    ]):
        axes[i].plot(p_values, values, 'o-', color=color)
        axes[i].plot([0, 100], [0, 100], '--', color='gray', alpha=0.5)
        axes[i].set_xlim(0, 100)
        axes[i].set_ylim(0, 100)
        axes[i].set_xlabel("p (%)")
        axes[i].set_ylabel("r (%)")
        axes[i].set_title(title)
        axes[i].axvline(x=50, color=EDGE_COLOR, linewidth=0.9, alpha=0.3)
    
    if model_name:
        fig.suptitle(f"Two-Flip Results - {model_name}", fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        svg_path = str(save_path).rsplit('.', 1)[0] + '.svg'
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"💾 Saved: {save_path} and {svg_path}")
    
    return fig
