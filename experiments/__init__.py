"""
Experiment modules for Failure to Mix paper.

Each experiment corresponds to a figure in the paper:
- exp1_single_flip: Figure 1 - Basic p-r response curves
- exp2_two_flips: Figure 2 - Sequential flip compensation (D=2, D=3)
- exp3_multi_flip: Figure 3 - D=10 flip ensemble analysis  
- exp4_ternary: Figure 4 - Three-outcome distribution
- exp5_word_bias: Figure 5 - Semantic word and position bias
- exp6_game_theory: Figure 6 - Mixed strategy problems
"""

from .exp1_single_flip import run_exp1, analyze_exp1, plot_exp1
from .exp2_two_flips import run_exp2, analyze_exp2, plot_exp2
from .exp3_multi_flip import run_exp3, analyze_exp3_by_j, analyze_exp3_mean, plot_fig3a, plot_fig3b, plot_fig3c
from .exp4_ternary import run_exp4, analyze_exp4, plot_exp4
from .exp5_word_bias import run_exp5, analyze_exp5, plot_exp5
from .exp6_game_theory import run_exp6, analyze_exp6, plot_exp6

__all__ = [
    "run_exp1", "analyze_exp1", "plot_exp1",
    "run_exp2", "analyze_exp2", "plot_exp2",
    "run_exp3", "analyze_exp3_by_j", "analyze_exp3_mean", "plot_fig3a", "plot_fig3b", "plot_fig3c",
    "run_exp4", "analyze_exp4", "plot_exp4",
    "run_exp5", "analyze_exp5", "plot_exp5",
    "run_exp6", "analyze_exp6", "plot_exp6",
]
