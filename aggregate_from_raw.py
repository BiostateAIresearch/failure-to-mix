"""
aggregate_from_raw.py (v2.4)
============================
Aggregate raw experiment results (results_raw.csv)
into structured CSVs for each figure (1C, 1E, 2C1, 3C, 4B1, 5BE, 5F, 6B).

F2 UPDATED:
- For digits with D=2, compute per model & p:
    r1 = P(digit_1 == 1)
    r2 = P(digit_2 == 1)
    r_mean = (r1 + r2)/2  (all in 0–100 scale for plotting)
"""

import pandas as pd
import numpy as np
import os
import re

RAW_FILE = "results_raw.csv"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

print("=== Aggregating raw results (v2.4) ===")
df = pd.read_csv(RAW_FILE)

def prop_one(series):
    return (series.astype(str) == "1").mean()

# ---------- F1: Coin -> 1C,1E ----------
coin = df[df["task_type"].str.lower() == "coin"]
f1 = (coin.groupby(["model","p"])["answer"]
      .apply(prop_one)
      .reset_index(name="prop_output_1"))
f1.to_csv(os.path.join(OUT_DIR,"1C.csv"),index=False)
f1.to_csv(os.path.join(OUT_DIR,"1E.csv"),index=False)

# ---------- F2: Digits D=2 -> 2C1 (UPDATED) ----------
digits = df[df["task_type"].str.lower() == "digits"].copy()

# Try to use parsed digit_1, digit_2 if present; otherwise parse from raw_text.
def get_digit(col_series, fallback_series, idx):
    # prefer explicit digit_i column if exists, else parse from raw_text
    if col_series is not None:
        return pd.to_numeric(col_series, errors="coerce")
    # parse fallback raw_text
    vals = []
    for s in fallback_series.fillna("").astype(str):
        nums = re.findall(r"-?\d+", s)
        if len(nums) >= idx:
            try:
                vals.append(int(nums[idx-1]))
            except Exception:
                vals.append(np.nan)
        else:
            vals.append(np.nan)
    return pd.Series(vals)

d1 = digits["digit_1"] if "digit_1" in digits.columns else None
d2 = digits["digit_2"] if "digit_2" in digits.columns else None
digits["d1"] = get_digit(d1, digits.get("raw_text"), 1)
digits["d2"] = get_digit(d2, digits.get("raw_text"), 2)

# Heuristic: keep rows that look like D=2 (no reliable d3)
if "digit_3" in digits.columns:
    mask_two = digits["digit_3"].isna()
else:
    # if third number can be parsed in raw_text, exclude
    third = get_digit(None, digits.get("raw_text"), 3)
    mask_two = third.isna()
digits2 = digits[mask_two].copy()

def pct(x): return 100.0 * x

f2 = (digits2.groupby(["model","p"])
      .apply(lambda g: pd.Series({
          "r1": pct((g["d1"] == 1).mean() if len(g) else np.nan),
          "r2": pct((g["d2"] == 1).mean() if len(g) else np.nan),
      }))
      .reset_index())
f2["r_mean"] = (f2["r1"] + f2["r2"]) / 2.0
f2 = f2.sort_values(["model","p"]).reset_index(drop=True)
f2.to_csv(os.path.join(OUT_DIR,"2C1.csv"), index=False)

# ---------- F3: Digits D=10 -> 3C ----------
# build per-model digit histogram if possible; fall back to global
def first_10_ints(s):
    nums = re.findall(r"-?\d+", str(s))
    return [int(x) for x in nums[:10]]

rows = []
for _, r in df[df["task_type"].str.lower()=="digits"].iterrows():
    digs = [d for d in first_10_ints(r.get("raw_text","")) if 0 <= d <= 9]
    if len(digs) == 10:  # only treat true D=10 as input to F3
        rows.append({"model": r["model"], "digits": digs})

if rows:
    rec = pd.DataFrame(rows)
    out_rows = []
    for model, sub in rec.groupby("model"):
        counts = pd.Series(np.concatenate(sub["digits"].to_numpy())).value_counts().reindex(range(10), fill_value=0).sort_index()
        for k, v in counts.items():
            out_rows.append({"model": model, "digit": k, "count": int(v)})
    f3 = pd.DataFrame(out_rows)
else:
    # fallback: global count from any digits rows
    all_digits = []
    for _, r in df[df["task_type"].str.lower()=="digits"].iterrows():
        all_digits += [d for d in first_10_ints(r.get("raw_text","")) if 0 <= d <= 9]
    counts = pd.Series(all_digits).value_counts().reindex(range(10), fill_value=0).sort_index()
    f3 = counts.reset_index().rename(columns={"index":"digit",0:"count"})

f3.to_csv(os.path.join(OUT_DIR,"3C.csv"), index=False)

# ---------- F4: three_options -> 4B1 ----------
three = df[df["task_type"].str.lower()=="three_options"]
f4 = three.groupby(["model","p","answer"]).size().reset_index(name="count")
f4["prop"] = f4.groupby(["model","p"])["count"].transform(lambda x: x/x.sum())
f4.to_csv(os.path.join(OUT_DIR,"4B1.csv"), index=False)

# ---------- F5: words -> 5BE, 5F ----------
words = df[df["task_type"].str.lower()=="words"].copy()
if "matched" not in words.columns:
    # try to infer matched from answer/v/v2 if needed
    words["matched"] = "other"
f5 = words.groupby(["model","p","matched"]).size().reset_index(name="count")
f5["prop"] = f5.groupby(["model","p"])["count"].transform(lambda x: x/x.sum())
f5.to_csv(os.path.join(OUT_DIR,"5BE.csv"), index=False)
f5.to_csv(os.path.join(OUT_DIR,"5F.csv"), index=False)

# ---------- F6: asymmetric -> 6B ----------
asym = df[df["task_type"].str.lower()=="asymmetric"]
f6 = (asym.groupby(["model","p"])["answer"]
      .apply(prop_one)
      .reset_index(name="prop_output_1"))
f6.to_csv(os.path.join(OUT_DIR,"6B.csv"), index=False)

print(f"✅ Aggregation complete. Files saved to {OUT_DIR}/")
