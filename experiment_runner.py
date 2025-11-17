"""
Unified Experiment Runner for Multiple Bias Evaluation Tasks
============================================================

This script performs automated large-scale evaluation of LLM biases
across multiple task types (binary coin, multi-class, word-choice,
digit-sequence, and asymmetric tasks).

The script:
    - Reads experiment definitions from a CSV or Google Sheet
    - Constructs prompts with {p}, {100-p}, {v}, {v2}
    - Calls OpenRouter API asynchronously
    - Parses responses according to declared `task_type`
    - Saves *results_raw.csv* only (no plots, no summary)

Allowed task_types:
    coin
    three_options
    digits
    words
    asymmetric
    generic

This script is suitable for academic publication and reproducibility.
"""

import asyncio
import aiohttp
import pandas as pd
import os
import datetime
import re
import traceback
from tqdm import tqdm

# ============================================================
# 1. LOAD API KEYS
# ============================================================

api_keys_raw = os.getenv("OPENROUTER_API_KEYS", "")
API_KEYS = [k.strip() for k in api_keys_raw.split(",") if k.strip()]
if not API_KEYS:
    raise RuntimeError("OPENROUTER_API_KEYS environment variable not set.")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ============================================================
# 2. CONFIGURATION
# ============================================================

INPUT_FILE = "input.csv"
OUTPUT_FILE = "results_raw.csv"

BATCH_SIZE = 20
INITIAL_WAIT = 2.0
MAX_RETRIES = 3
MAX_TOKENS_LIMIT = None
TEMPERATURE = None

RUN_CODES = []  # Optional filtering; empty = run all

os.makedirs("results", exist_ok=True)
SAVE_ROOT = "results"
ts_run = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
RUN_LOCAL_DIR = os.path.join(SAVE_ROOT, f"run_{ts_run}")
os.makedirs(RUN_LOCAL_DIR, exist_ok=True)

# ============================================================
# 3. READ INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"{INPUT_FILE} not found.")

df_input = pd.read_csv(INPUT_FILE).dropna(how="all")

required = {"code", "model", "task_type", "p", "prompt", "trials"}
missing = required - set(df_input.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

df_input["p"] = pd.to_numeric(df_input["p"], errors="coerce").fillna(0.0)
df_input["trials"] = pd.to_numeric(df_input["trials"], errors="coerce").astype(int)
df_input["v"] = df_input.get("v", "")
df_input["v2"] = df_input.get("v2", "")

if RUN_CODES:
    df_input = df_input[df_input["code"].isin(RUN_CODES)]

# ============================================================
# 4. PROMPT FILLING
# ============================================================

def fill_prompt(template: str, p: float, v: str, v2: str) -> str:
    t = template or ""
    return (
        t.replace("{p}", str(p))
         .replace("{100-p}", str(100 - p))
         .replace("{v}", str(v))
         .replace("{v2}", str(v2))
    )

df_input["prompt_filled"] = [
    fill_prompt(r["prompt"], r["p"], r["v"], r["v2"])
    for _, r in df_input.iterrows()
]

# ============================================================
# 5. RESPONSE PARSING UTILITIES
# ============================================================

def parse_response(d: dict):
    if not isinstance(d, dict):
        return "", "", ""
    choices = d.get("choices") or [{}]
    msg = choices[0].get("message", {}) or {}
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning", "") or ""

    if not reasoning:
        rd = msg.get("reasoning_details")
        if isinstance(rd, list) and rd:
            reasoning = rd[0].get("summary") or rd[0].get("data") or ""

    merged = (content or reasoning).strip()
    return content.strip(), reasoning.strip(), merged

# ------- task-specific extractors -------

def ex_coin(text):
    if not text:
        return "Unknown", {}
    m = re.findall(r"\b(Heads?|Tails?)\b", text, flags=re.I)
    if m:
        return m[-1].capitalize(), {}
    m2 = re.findall(r"\b[01]\b", text)
    if m2:
        return m2[-1], {}
    return "Unknown", {}

def ex_three(text):
    if not text:
        return "other", {}
    m = re.findall(r"\b[0-2]\b", text)
    if m:
        return m[-1], {}
    return "other", {}

def ex_digits(text, max_len=10):
    nums = [int(x) for x in re.findall(r"-?\d+", text or "")]
    nums = nums[:max_len]
    extra = {f"digit_{i+1}": (nums[i] if i < len(nums) else None)
             for i in range(max_len)}
    extra["sum_digits"] = sum(nums) if nums else None
    ans = " ".join(str(x) for x in nums) if nums else "Unknown"
    return ans, extra

def ex_words(text, v, v2):
    t = (text or "").strip()
    if not t:
        return "Unknown", {"matched": "other"}
    last = re.sub(r"[^\w]+$", "", t.split()[-1]).lower()
    v_clean, v2_clean = v.lower(), v2.lower()
    if last == v_clean:
        return last, {"matched": "v"}
    if last == v2_clean:
        return last, {"matched": "v2"}
    return last, {"matched": "other"}

def ex_asym(text):
    return ex_coin(text)

def ex_generic(text):
    t = (text or "").split()
    return (t[-1] if t else "Unknown"), {}

TASK_MAP = {
    "coin": ex_coin,
    "three_options": ex_three,
    "digits": ex_digits,
    "words": ex_words,
    "asymmetric": ex_asym,
    "generic": ex_generic,
}

def normalize_type(t):
    t = (t or "").strip().lower()
    return t if t in TASK_MAP else "generic"

# ============================================================
# 6. API CALL
# ============================================================

async def safe_post(session, payload, api_key, model, attempt=1):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with session.post(API_URL, json=payload, headers=headers, timeout=240) as r:
            return await r.json()
    except Exception:
        if attempt < MAX_RETRIES:
            await asyncio.sleep(2 * attempt)
            return await safe_post(session, payload, api_key, model, attempt+1)
        return None

# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

async def run_experiment():
    recs = [dict(row) for _, row in df_input.iterrows() for _ in range(row["trials"])]
    results = []

    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(recs), BATCH_SIZE)):
            batch = recs[i:i+BATCH_SIZE]
            tasks = []

            for idx, b in enumerate(batch):
                api_key = API_KEYS[idx % len(API_KEYS)]
                sys = "You are a helpful model. Provide a clear final answer."

                payload = {
                    "model": b["model"],
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": b["prompt_filled"]}
                    ]
                }

                if MAX_TOKENS_LIMIT is not None:
                    payload["max_tokens"] = int(MAX_TOKENS_LIMIT)
                if TEMPERATURE is not None:
                    payload["temperature"] = float(TEMPERATURE)

                tasks.append(safe_post(session, payload, api_key, b["model"]))

            outs = await asyncio.gather(*tasks)

            for b, d in zip(batch, outs):
                raw_c, raw_r, merged = "", "", ""
                ans, extra = "Unknown", {}
                ttype = normalize_type(b["task_type"])

                if d:
                    raw_c, raw_r, merged = parse_response(d)
                    extractor = TASK_MAP[ttype]

                    if ttype == "digits":
                        ans, extra = extractor(merged)
                    elif ttype == "words":
                        ans, extra = extractor(merged, b.get("v", ""), b.get("v2", ""))
                    else:
                        ans, extra = extractor(merged)

                record = {
                    "code": b["code"],
                    "model": b["model"],
                    "task_type": ttype,
                    "p": b["p"],
                    "v": b.get("v", ""),
                    "v2": b.get("v2", ""),
                    "prompt": b["prompt_filled"],
                    "raw_content": raw_c,
                    "raw_reasoning": raw_r,
                    "raw_text": merged,
                    "answer": ans,
                    "status": "success" if d else "error",
                }
                record.update(extra)
                results.append(record)

            await asyncio.sleep(INITIAL_WAIT)

    df = pd.DataFrame(results)
    out = os.path.join(RUN_LOCAL_DIR, OUTPUT_FILE)
    df.to_csv(out, index=False)
    print("Saved:", out)
    return df

# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(run_experiment())
        print("Completed. RAW results only.")
    except Exception as e:
        err_file = os.path.join(RUN_LOCAL_DIR, "error_log.txt")
        with open(err_file, "w") as f:
            f.write(traceback.format_exc())
        print("Error occurred. Saved:", err_file)
