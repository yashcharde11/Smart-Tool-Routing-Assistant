"""Calculator tool — SAFE math evaluation only.

Security: we never call Python's built-in ``eval()`` on user input. All
evaluation goes through ``sympy.sympify`` with a restricted symbol table, so a
user cannot smuggle in ``__import__`` or attribute access. This is the single
most important security property of the project (see the plan's security note).
"""

from __future__ import annotations

import re

from sympy import sympify
from sympy.core.sympify import SympifyError

# A small, explicit allow-list of math functions/constants the parser may use.
# Anything not listed here (e.g. attribute access, names) is rejected by sympify.
_ALLOWED = {
    name: getattr(__import__("sympy"), name)
    for name in (
        "sqrt", "log", "exp", "sin", "cos", "tan", "asin", "acos", "atan",
        "pi", "E", "Abs", "factorial", "floor", "ceiling", "Pow", "Mod",
    )
}

# Word operators applied before parsing (longer phrases first so "divided by"
# wins over a bare "by"). Note: "mod" and "%-of" are handled separately, below,
# because they conflict with the bare-percentage rule.
_WORD_OPS = [
    (r"\bmultiplied by\b", "*"),
    (r"\bdivided by\b", "/"),
    (r"\bto the power of\b", "**"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\btimes\b", "*"),
    (r"\bover\b", "/"),
    (r"\bpower\b", "**"),
    (r"\bsquared\b", "**2"),
    (r"\bcubed\b", "**3"),
]

# Standalone English words that are safe to drop (question scaffolding). Any
# *other* alphabetic token survives and is rejected by the allow-list check, so
# injection terms like 'os', 'import', 'exec' are never silently swallowed.
_FILLER = {
    "what", "whats", "is", "are", "the", "a", "an", "of", "how", "much", "many",
    "please", "calculate", "compute", "evaluate", "result", "value", "me",
    "give", "tell", "find", "equals", "equal", "to", "be", "would", "do", "does",
    "you", "can", "and", "then", "left", "from", "annual", "allowance",
}


class CalculatorError(ValueError):
    """Raised when the input cannot be safely interpreted as arithmetic."""


def normalize_expression(text: str) -> str:
    """Turn a natural-language math question into a clean arithmetic string.

    Handles the common phrasings the router will route here: ``"What is 18% of
    4,500?"``, ``"12 times 7"``, ``"sqrt(144)"``, ``"15% off 80"``. The result
    is a string of digits, operators and allow-listed function names only.
    """
    s = text.strip().lower()

    # Drop polite/question scaffolding and trailing punctuation.
    s = re.sub(r"^(what\s+is|whats|calculate|compute|evaluate|how much is|solve)\b", " ", s)
    s = s.replace("?", " ").replace("=", " ")

    # Currency and thousands separators: "$4,500" -> "4500".
    s = s.replace("$", " ").replace("€", " ").replace("£", " ")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    # Roots and factorial (phrasings that need parentheses inserted).
    s = re.sub(r"square root of\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", s)
    s = re.sub(r"cube root of\s*(\d+(?:\.\d+)?)", r"(\1)**(1/3)", s)
    s = re.sub(r"(\d+)\s*factorial", r"factorial(\1)", s)

    # "X mod Y" -> Mod(X, Y). Done before % rules so it doesn't become a percent.
    s = re.sub(r"(\d+(?:\.\d+)?)\s*mod\s*(\d+(?:\.\d+)?)", r"Mod(\1, \2)", s)

    # Percent: "X% off Y" -> Y*(1-X/100); "X% of Y" -> (X/100)*Y; bare "X%".
    s = re.sub(r"percent of", "% of", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*off\s*(\d+(?:\.\d+)?)", r"\2*(1-\1/100)", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", r"(\1/100)*\2", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)

    # Word operators.
    for pattern, repl in _WORD_OPS:
        s = re.sub(pattern, repl, s)

    # 'x' used as a multiplication sign between numbers: "3 x 4" -> "3 * 4".
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", s)

    # Drop standalone filler words (keeps unknown/injection terms for rejection).
    s = " ".join(w for w in s.split() if not (w.isalpha() and w in _FILLER))

    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def calculate(text: str) -> str:
    """Evaluate a math question and return a human-readable answer string.

    Raises ``CalculatorError`` for anything that is not safe, well-formed
    arithmetic, so callers can show a friendly message instead of crashing.
    """
    expr = normalize_expression(text)
    if not expr:
        raise CalculatorError("No arithmetic expression found in the question.")

    # Reject anything containing letters that are not allow-listed function
    # names — this blocks attribute access and arbitrary identifiers early.
    leftover = re.sub(r"[0-9\.\+\-\*/%()\s,]", "", expr)
    tokens = re.findall(r"[a-zA-Z_]+", leftover)
    for tok in tokens:
        if tok not in _ALLOWED:
            raise CalculatorError(f"Unsupported term in expression: '{tok}'.")

    try:
        result = sympify(expr, locals=_ALLOWED, evaluate=True)
        value = result.evalf()
    except (SympifyError, SyntaxError, TypeError, ValueError) as exc:
        raise CalculatorError(f"Could not evaluate '{expr}'.") from exc

    # Present clean integers without a trailing ".0", others rounded sensibly.
    try:
        f = float(value)
        if f == int(f):
            pretty = str(int(f))
        else:
            pretty = f"{round(f, 6):g}"
    except (TypeError, ValueError):
        pretty = str(value)

    return f"{expr} = {pretty}"
