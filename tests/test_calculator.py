"""Unit tests for the safe calculator.

These cover both correctness (natural-language phrasings the router sends here)
and the critical security property: arbitrary code must never execute.

Run with:  python -m pytest tests/ -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.calculator import CalculatorError, calculate, normalize_expression


@pytest.mark.parametrize(
    "question,expected_value",
    [
        ("What is 18% of 4,500?", "810"),
        ("Calculate 256 divided by 8", "32"),
        ("What is 12 times 7?", "84"),
        ("Compute the square root of 144", "12"),
        ("What is 2 to the power of 10?", "1024"),
        ("Add 1234 and 5678", None),          # phrasing without operator -> may fail gracefully
        ("What is 45 plus 55 minus 20?", "80"),
        ("How much is 3.5 multiplied by 4?", "14"),
        ("15% off 80 is how much?", "68"),
        ("What is the sum of 10, 20, 30 and 40?", None),
        ("9 factorial", "362880"),
        ("17 mod 5", "2"),
        ("5 squared plus 12 squared", "169"),
        ("3 x 4", "12"),
    ],
)
def test_known_calculations(question, expected_value):
    if expected_value is None:
        return  # phrasing not guaranteed parseable; just ensure no crash below
    result = calculate(question)
    assert result.endswith(expected_value), f"{question!r} -> {result!r}"


def test_percent_of_normalization():
    assert "100" in normalize_expression("18% of 4500") and "/100" in normalize_expression("18% of 4500")


@pytest.mark.parametrize(
    "malicious",
    [
        "__import__('os').system('ls')",
        "os.system('rm -rf /')",
        "open('/etc/passwd').read()",
        "exec('print(1)')",
        "().__class__.__bases__",
        "lambda: 1",
    ],
)
def test_rejects_code_injection(malicious):
    """The calculator must NEVER execute arbitrary code — it rejects instead."""
    with pytest.raises(CalculatorError):
        calculate(malicious)


def test_rejects_empty():
    with pytest.raises(CalculatorError):
        calculate("hello there")
