#!/usr/bin/env python3
"""
Generate all figures for the Failure to Mix paper.

This script reads processed data from Google Sheets or local CSVs
and generates publication-quality figures.

Usage:
    python generate_figures.py --data-dir ./data --output-dir ./figures
    python generate_figures.py --sheet-keyword figure1 --output-dir ./figures
"""
import argparse
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_model_color, EDGE_COLOR, MODEL_COLORS
from src.metrics import compute_S


def setup_figure_style():
    """Set up publication-quality figure style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.transparent': True,
    })


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a dataframe from Google Sheets format.
    Expects: model column + percentage columns (0%, 5%, ..., 100%)
    """
    df = df.copy().dropna(how="all")
    
    # Find model column
    model_col = next(
        (c for c in df.columns if str(c).strip().lower() == "model"),
        df.columns[0]
    )
    df = df.rename(columns={model_col: "model"})
    
    # Keep only percentage columns
    def is_pct_col(s):
        return bool(re.fullmatch(r"\d+(\.\d+)?%?", str(s).strip()))
    
    keep = ["model"] + [c for c in df.columns if c != "model" and is_pct_col(c)]
    df = df[keep].copy()
    
    # Convert column names to numeric
    new_cols = ["model"] + [
        float(str(c).strip().rstrip("%")) for c in df.columns if c != "model"
    ]
    df.columns = new_cols
    
    # Convert cell values to float
    for c in df.columns:
        if c == "model":
            continue
        df[c] = (
            df[c].astype(str)
            .str.replace("%", "", regex=False)
            .replace("", np.nan)
            .astype(float)
        )
    
    # Scale if values are 0-1
    for c in df.columns:
        if c == "model":
            continue
        col_max = df[c].max(skipna=True)
        if pd.notna(col_max) and 0 < col_max < 1:
            df[c] = df[c] * 100
    
    return df.reset_index(drop=True)


def plot_figure_1b(df: pd.DataFrame, output_path: str):
    """
    Figure 1B: Single Gemini p-r curve (paper-style, minimal).
    """
    tbl = normalize_dataframe(df)
    
    # Get Gemini row
    gemini_row = tbl[tbl["model"].str.contains("gemini", case=False)].iloc[0]
    
    p_cols = sorted([c for c in tbl.columns if c != "model"])
    p_vals = np.array(p_cols)
    r_vals = gemini_row[p_cols].values
    
    # Extend to 0 and 100
    if p_vals[0] > 0:
        p_vals = np.insert(p_vals, 0, 0)
        r_vals = np.insert(r_vals, 0, r_vals[0])
    if p_vals[-1] < 100:
        p_vals = np.append(p_vals, 100)
        r_vals = np.append(r_vals, r_vals[-1])
    
    S = compute_S(p_vals, r_vals)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    color = MODEL_COLORS["google/gemini-2.5-pro"]
    
    ax.plot(p_vals, r_vals, color=color, linewidth=1.8, zorder=2)
    ax.scatter(p_vals, r_vals, s=50, facecolors=color, edgecolors='black',
               linewidths=0.6, zorder=3)
    
    ax.plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, linewidth=0.8)
    ax.axvline(x=50, color=EDGE_COLOR, linewidth=0.9, alpha=0.3)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("p (%)", fontweight='bold')
    ax.set_ylabel("r (%)", fontweight='bold')
    
    for spine in ax.spines.values():
        spine.set_color(EDGE_COLOR)
        spine.set_linewidth(0.9)
    ax.tick_params(color=EDGE_COLOR, labelcolor=EDGE_COLOR)
    
    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"   S = {S:.4f}")


def plot_figure_1c(df: pd.DataFrame, output_path: str, x_range=(45, 55)):
    """
    Figure 1C: Zoomed view around p=50.
    """
    tbl = normalize_dataframe(df)
    
    p_cols = sorted([c for c in tbl.columns if c != "model"])
    p_all = np.array(p_cols)
    mask = (p_all >= x_range[0]) & (p_all <= x_range[1])
    p_sel = p_all[mask]
    
    fig, ax = plt.subplots(figsize=(9, 3))
    
    for _, row in tbl.iterrows():
        model = row["model"]
        r_vals = row[p_cols].values[mask]
        color = get_model_color(model)
        
        ax.plot(p_sel, r_vals, marker='o', markersize=7, linewidth=1.5,
                markeredgecolor='black', markeredgewidth=0.8,
                markerfacecolor=color, color=color)
    
    ax.axvline(x=50, color=EDGE_COLOR, linewidth=0.9, alpha=0.4)
    ax.set_xlim(x_range)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([0, 50, 100])
    
    for spine in ax.spines.values():
        spine.set_color(EDGE_COLOR)
    ax.tick_params(color=EDGE_COLOR, labelcolor=EDGE_COLOR)
    
    plt.tight_layout()
    save_figure(fig, output_path)


def plot_figure_2(df: pd.DataFrame, output_path: str):
    """
    Figure 2: Multi-model comparison.
    """
    tbl = normalize_dataframe(df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    p_cols = sorted([c for c in tbl.columns if c != "model"])
    
    for _, row in tbl.iterrows():
        model = row["model"].strip()
        p_vals = np.array(p_cols)
        r_vals = row[p_cols].values
        
        # Remove NaN
        mask = np.isfinite(r_vals)
        p_vals = p_vals[mask]
        r_vals = r_vals[mask]
        
        if len(p_vals) < 2:
            continue
        
        # Extend to boundaries
        if p_vals[0] > 0:
            p_vals = np.insert(p_vals, 0, 0)
            r_vals = np.insert(r_vals, 0, r_vals[0])
        if p_vals[-1] < 100:
            p_vals = np.append(p_vals, 100)
            r_vals = np.append(r_vals, r_vals[-1])
        
        S = compute_S(p_vals, r_vals)
        color = get_model_color(model)
        
        ax.plot(p_vals, r_vals, color=color, linewidth=1.8, zorder=2)
        ax.scatter(p_vals, r_vals, s=45, facecolors=color, edgecolors='black',
                   linewidths=0.6, zorder=3, label=f"{model} (S={S:.3f})")
    
    ax.plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, linewidth=0.8)
    ax.axvline(x=50, color=EDGE_COLOR, linewidth=0.9, alpha=0.3)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("p (%)", fontweight='bold')
    ax.set_ylabel("r (%)", fontweight='bold')
    ax.legend(frameon=False, loc='lower right')
    
    for spine in ax.spines.values():
        spine.set_color(EDGE_COLOR)
    ax.tick_params(color=EDGE_COLOR, labelcolor=EDGE_COLOR)
    
    plt.tight_layout()
    save_figure(fig, output_path)


def plot_figure_5(df: pd.DataFrame, output_path: str):
    """
    Figure 5: Word pair bias (horizontal bar chart).
    """
    # Expects: Pairs column + model columns with percentages
    pairs = df.iloc[:, 0].astype(str).str.strip()
    
    # Find first numeric column
    for col in df.columns[1:]:
        try:
            values = pd.to_numeric(df[col].astype(str).str.replace("%", ""), errors='coerce')
            if values.notna().any():
                break
        except:
            continue
    
    LEFT_BLUE = "#5dade2"
    RIGHT_RED = "#e74c3c"
    
    vals_right = values.clip(0, 100).values
    vals_left = 100 - vals_right
    
    # Parse pair labels
    left_labels = [p.split("/")[0].strip() if "/" in p else p for p in pairs]
    right_labels = [p.split("/")[-1].strip() if "/" in p else "" for p in pairs]
    
    y = np.arange(len(pairs))
    h = max(4, 0.6 * len(pairs))
    
    fig, ax = plt.subplots(figsize=(5, h))
    
    ax.barh(y, vals_left, color=LEFT_BLUE, label="First word")
    ax.barh(y, vals_right, left=100 - vals_right, color=RIGHT_RED, label="Second word")
    
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l} / {r}" for l, r in zip(left_labels, right_labels)])
    ax.set_xlabel("Percentage")
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis='x', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_path)


def save_figure(fig, path):
    """Save figure in both PNG and SVG formats."""
    fig.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    svg_path = path.rsplit('.', 1)[0] + '.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
    print(f"💾 Saved: {path} and {svg_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--data-dir", type=str, help="Directory with CSV data files")
    parser.add_argument("--sheet-keyword", type=str, help="Google Sheet keyword to find")
    parser.add_argument("--output-dir", type=str, default="./figures")
    args = parser.parse_args()
    
    setup_figure_style()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎨 Generating Failure to Mix Figures")
    print(f"   Output: {args.output_dir}")
    
    # Example with local CSV files
    if args.data_dir:
        # Load data from CSVs
        for csv_file in os.listdir(args.data_dir):
            if not csv_file.endswith('.csv'):
                continue
            
            df = pd.read_csv(os.path.join(args.data_dir, csv_file))
            name = csv_file.replace('.csv', '')
            
            if '1b' in name.lower():
                print("\n📊 Generating Figure 1B...")
                plot_figure_1b(df, os.path.join(args.output_dir, "figure_1b.png"))
            elif '1c' in name.lower():
                print("\n📊 Generating Figure 1C...")
                plot_figure_1c(df, os.path.join(args.output_dir, "figure_1c.png"))
            elif '2' in name.lower() and 'model' in df.columns[0].lower():
                print("\n📊 Generating Figure 2...")
                plot_figure_2(df, os.path.join(args.output_dir, "figure_2.png"))
            elif '5' in name.lower():
                print("\n📊 Generating Figure 5...")
                plot_figure_5(df, os.path.join(args.output_dir, "figure_5.png"))
    
    print("\n✅ Figure generation complete!")


if __name__ == "__main__":
    main()
