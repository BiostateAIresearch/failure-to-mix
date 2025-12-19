"""
Metrics for evaluating LLM probability calibration.

Key metric: Step-likeness (S) - measures deviation from ideal linear response.
"""
import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy import stats


def compute_S(
    p_values: np.ndarray,
    r_values: np.ndarray,
    extend_to_bounds: bool = True
) -> float:
    """
    Compute Step-likeness metric S.
    
    S measures how closely the response curve resembles a step function
    versus the ideal linear relationship r = p.
    
    Formula: S = 4 × ∫₀¹ |r(p) - p| dp
    
    Args:
        p_values: Target probabilities (0-100 as percentages or 0-1 as fractions)
        r_values: Observed response rates (same scale as p_values)
        extend_to_bounds: Whether to extend curve to [0, 100] boundaries
        
    Returns:
        S value where:
        - S = 0: Perfect calibration (r = p)
        - S = 1: Perfect step function
        - S > 1: Anti-correlated responses
    """
    p = np.asarray(p_values, dtype=float)
    r = np.asarray(r_values, dtype=float)
    
    # Remove NaN values
    mask = np.isfinite(p) & np.isfinite(r)
    p, r = p[mask], r[mask]
    
    if len(p) < 2:
        return np.nan
    
    # Sort by p
    order = np.argsort(p)
    p, r = p[order], r[order]
    
    # Detect if values are percentages (0-100) or fractions (0-1)
    if p.max() > 1:
        # Convert to fractions
        p = p / 100.0
        r = r / 100.0
    
    # Extend to [0, 1] boundaries
    if extend_to_bounds:
        if p[0] > 0:
            p = np.insert(p, 0, 0)
            r = np.insert(r, 0, r[0])
        if p[-1] < 1:
            p = np.append(p, 1)
            r = np.append(r, r[-1])
    
    # Compute using trapezoidal rule
    dp = np.diff(p)
    left = np.abs(r[:-1] - p[:-1])
    right = np.abs(r[1:] - p[1:])
    trap = np.sum(dp * (left + right) / 2.0)
    
    S = 4.0 * trap
    return S


def compute_calibration_score(S: float) -> float:
    """
    Convert S to a 0-10 calibration score.
    
    Score = 10 × (1 - S)
    
    Args:
        S: Step-likeness metric
        
    Returns:
        Score from 0 (worst) to 10 (perfect calibration)
    """
    if np.isnan(S):
        return np.nan
    return 10.0 * (1.0 - min(S, 1.0))


def compute_kl_divergence(
    observed: np.ndarray,
    expected: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute KL divergence D_KL(observed || expected).
    
    Args:
        observed: Observed probability distribution
        expected: Expected probability distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence value
    """
    p = np.asarray(observed, dtype=float)
    q = np.asarray(expected, dtype=float)
    
    # Normalize
    p = p / (p.sum() + epsilon)
    q = q / (q.sum() + epsilon)
    
    # Add epsilon to avoid log(0)
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    return np.sum(p * np.log(p / q))


def compute_chi_square(
    observed: np.ndarray,
    expected: np.ndarray
) -> Tuple[float, float]:
    """
    Compute chi-square statistic and p-value.
    
    Args:
        observed: Observed counts
        expected: Expected counts
        
    Returns:
        Tuple of (chi_square_statistic, p_value)
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    
    # Remove zeros
    mask = (expected > 0) & (observed >= 0)
    observed = observed[mask]
    expected = expected[mask]
    
    if len(observed) < 2:
        return np.nan, np.nan
    
    chi2, p_value = stats.chisquare(observed, expected)
    return chi2, p_value


def compute_binomial_expected(
    n_flips: int,
    p: float,
    n_trials: int
) -> np.ndarray:
    """
    Compute expected binomial distribution.
    
    Args:
        n_flips: Number of flips (D)
        p: Probability of success per flip
        n_trials: Total number of trials
        
    Returns:
        Expected counts for each possible sum (0 to n_flips)
    """
    from scipy.stats import binom
    
    k = np.arange(n_flips + 1)
    probs = binom.pmf(k, n_flips, p)
    return probs * n_trials


def analyze_response_curve(
    p_values: List[float],
    r_values: List[float]
) -> Dict[str, float]:
    """
    Comprehensive analysis of a response curve.
    
    Args:
        p_values: Target probabilities (percentages)
        r_values: Observed response rates (percentages)
        
    Returns:
        Dictionary with various metrics
    """
    p = np.array(p_values)
    r = np.array(r_values)
    
    # Basic statistics
    S = compute_S(p, r)
    score = compute_calibration_score(S)
    
    # Mean absolute error
    mae = np.mean(np.abs(r - p))
    
    # Root mean square error
    rmse = np.sqrt(np.mean((r - p) ** 2))
    
    # Correlation
    correlation = np.corrcoef(p, r)[0, 1] if len(p) > 1 else np.nan
    
    # Bias (mean of r - p)
    bias = np.mean(r - p)
    
    # Slope from linear regression
    if len(p) > 1:
        slope, intercept = np.polyfit(p, r, 1)
    else:
        slope, intercept = np.nan, np.nan
    
    return {
        "S": S,
        "score": score,
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "bias": bias,
        "slope": slope,
        "intercept": intercept
    }


def analyze_compensation_effect(
    s1_values: List[float],
    s2_values: List[float],
    p_values: List[float]
) -> Dict[str, float]:
    """
    Analyze compensation effect in two-flip experiments.
    
    Checks if second flip tends to compensate for deviation in first flip.
    
    Args:
        s1_values: Response rates for first flip
        s2_values: Response rates for second flip
        p_values: Target probabilities
        
    Returns:
        Dictionary with compensation metrics
    """
    s1 = np.array(s1_values)
    s2 = np.array(s2_values)
    p = np.array(p_values) / 100.0  # Convert to fractions
    
    # Compute mean of both flips
    s_mean = (s1 + s2) / 2
    
    # Deviation from target for each flip
    dev1 = s1 - p
    dev2 = s2 - p
    
    # Compensation: negative correlation between deviations
    compensation_corr = np.corrcoef(dev1, dev2)[0, 1] if len(dev1) > 1 else np.nan
    
    # S values for each
    S1 = compute_S(p * 100, s1 * 100)
    S2 = compute_S(p * 100, s2 * 100)
    S_mean = compute_S(p * 100, s_mean * 100)
    
    return {
        "S1": S1,
        "S2": S2,
        "S_mean": S_mean,
        "compensation_correlation": compensation_corr,
        "mean_deviation_flip1": np.mean(np.abs(dev1)),
        "mean_deviation_flip2": np.mean(np.abs(dev2)),
        "mean_deviation_average": np.mean(np.abs(s_mean - p))
    }
