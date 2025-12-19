"""
Response parsers for extracting answers from LLM outputs.
"""
import re
from typing import Optional, List, Tuple


def extract_binary(text: str) -> str:
    """
    Extract binary (0/1) answer from text.
    
    Priority:
    1. Exact match "0" or "1"
    2. Contains only one digit type
    3. Last standalone digit found
    4. Semantic parsing ("one", "zero")
    
    Args:
        text: Raw LLM response
        
    Returns:
        "0", "1", or "error"
    """
    if not text:
        return "error"
    
    t = text.strip()
    
    # Exact match
    if t == "0":
        return "0"
    if t == "1":
        return "1"
    
    # Find all standalone digits
    matches = re.findall(r"\b([01])\b", t)
    if matches:
        return matches[-1]  # Return last match
    
    # Contains only one type
    has_0 = "0" in t
    has_1 = "1" in t
    if has_1 and not has_0:
        return "1"
    if has_0 and not has_1:
        return "0"
    
    # Semantic parsing
    low = t.lower()
    if "one" in low and "zero" not in low:
        return "1"
    if "zero" in low and "one" not in low:
        return "0"
    
    return "error"


def extract_ternary(text: str) -> str:
    """
    Extract ternary (0/1/2) answer from text.
    
    Args:
        text: Raw LLM response
        
    Returns:
        "0", "1", "2", or "error"
    """
    if not text:
        return "error"
    
    t = text.strip()
    
    # Exact match
    if t in ["0", "1", "2"]:
        return t
    
    # Find standalone digits
    matches = re.findall(r"\b([012])\b", t)
    if matches:
        return matches[-1]
    
    # Check for single digit type
    has_0 = "0" in t
    has_1 = "1" in t
    has_2 = "2" in t
    
    count = sum([has_0, has_1, has_2])
    if count == 1:
        if has_2:
            return "2"
        if has_1:
            return "1"
        if has_0:
            return "0"
    
    # Semantic parsing
    low = t.lower()
    if "two" in low and "one" not in low and "zero" not in low:
        return "2"
    if "one" in low and "two" not in low and "zero" not in low:
        return "1"
    if "zero" in low and "one" not in low and "two" not in low:
        return "0"
    
    return "error"


def extract_heads_tails(text: str) -> str:
    """
    Extract Heads/Tails answer from text.
    
    Args:
        text: Raw LLM response
        
    Returns:
        "Heads", "Tails", or "error"
    """
    if not text:
        return "error"
    
    # Find all matches
    matches = re.findall(r"(Heads|Tails)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()
    
    return "error"


def extract_word_choice(text: str, word1: str, word2: str) -> str:
    """
    Extract choice between two words.
    
    Args:
        text: Raw LLM response
        word1: First option
        word2: Second option
        
    Returns:
        word1, word2, or "error"
    """
    if not text:
        return "error"
    
    t = text.strip().lower()
    w1_lower = word1.lower()
    w2_lower = word2.lower()
    
    # Exact match
    if t == w1_lower:
        return word1
    if t == w2_lower:
        return word2
    
    # Find matches
    pattern = f"({re.escape(w1_lower)}|{re.escape(w2_lower)})"
    matches = re.findall(pattern, t, flags=re.IGNORECASE)
    
    if matches:
        last = matches[-1].lower()
        return word1 if last == w1_lower else word2
    
    return "error"


def extract_multi_digits(text: str, n_digits: int = 10) -> Tuple[List[int], int]:
    """
    Extract multiple digits from text.
    
    Args:
        text: Raw LLM response
        n_digits: Expected number of digits
        
    Returns:
        Tuple of (list of digits, sum of digits)
    """
    if not text:
        return [None] * n_digits, 0
    
    t = text.strip()
    
    # Try space-separated
    if " " in t:
        nums = [int(x) for x in t.split() if x.isdigit() and len(x) == 1]
    else:
        # Try consecutive digits
        nums = [int(x) for x in t if x.isdigit()]
    
    # Pad or truncate to n_digits
    if len(nums) < n_digits:
        nums = nums + [None] * (n_digits - len(nums))
    nums = nums[:n_digits]
    
    # Compute sum (ignoring None values)
    total = sum(x for x in nums if x is not None)
    
    return nums, total


def extract_game_choice(text: str, options: List[str]) -> str:
    """
    Extract game theory choice from text.
    
    Args:
        text: Raw LLM response
        options: List of valid options (e.g., ["A", "B"] or ["Heads", "Tails"])
        
    Returns:
        Chosen option or "error"
    """
    if not text:
        return "error"
    
    t = text.strip()
    
    # Exact match first
    for opt in options:
        if t.lower() == opt.lower():
            return opt
    
    # Find matches
    pattern = "|".join(re.escape(opt) for opt in options)
    matches = re.findall(f"({pattern})", t, flags=re.IGNORECASE)
    
    if matches:
        # Return matching option in original case
        last = matches[-1].lower()
        for opt in options:
            if opt.lower() == last:
                return opt
    
    return "error"


def extract_numeric(text: str) -> Optional[float]:
    """
    Extract numeric value from text.
    
    Args:
        text: Raw LLM response
        
    Returns:
        Extracted number or None
    """
    if not text:
        return None
    
    # Find all numbers (including decimals)
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    
    return None


def normalize_response(text: str) -> str:
    """
    Normalize response text for comparison.
    
    Args:
        text: Raw response
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Strip whitespace
    t = text.strip()
    
    # Remove common prefixes
    prefixes = [
        "the answer is",
        "my answer is",
        "i choose",
        "i pick",
        "i select",
        "result:",
        "answer:",
    ]
    
    t_lower = t.lower()
    for prefix in prefixes:
        if t_lower.startswith(prefix):
            t = t[len(prefix):].strip()
            break
    
    return t
