"""Routing-accuracy evaluation.

Feeds every labeled question to the router, records the predicted tool, and
compares to the gold label. Reports overall accuracy, a per-class
precision/recall/F1 table, and saves a confusion-matrix image.

Usage (from the project root, with Azure credentials set as env vars):

    python -m eval.evaluate
    python -m eval.evaluate --docs        # tell the router docs are available

The ``--docs`` flag matters because DOC_SEARCH questions assume an uploaded
document exists; set it when evaluating the realistic helpdesk scenario.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Allow running as a script or module from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TOOLS  # noqa: E402
from src.router import route  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def run_evaluation(questions_csv: str, has_documents: bool) -> pd.DataFrame:
    df = pd.read_csv(questions_csv)
    preds, reasons = [], []
    for i, row in df.iterrows():
        tool, reason = route(str(row["question"]), has_documents=has_documents)
        preds.append(tool)
        reasons.append(reason)
        print(f"[{i + 1:>2}/{len(df)}] {row['label']:<10} -> {tool:<10} | {row['question'][:55]}")
    df["predicted"] = preds
    df["reason"] = reasons
    df["correct"] = df["label"] == df["predicted"]
    return df


def save_confusion_matrix(df: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(TOOLS)
    cm = confusion_matrix(df["label"], df["predicted"], labels=labels)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Routing Confusion Matrix")
    for r in range(len(labels)):
        for c in range(len(labels)):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nSaved confusion matrix -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate router routing accuracy.")
    parser.add_argument(
        "--questions",
        default=os.path.join(HERE, "questions.csv"),
        help="Path to labeled questions CSV (columns: question,label).",
    )
    parser.add_argument(
        "--docs",
        action="store_true",
        help="Signal to the router that documents are available for DOC_SEARCH.",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = run_evaluation(args.questions, has_documents=args.docs)

    acc = accuracy_score(df["label"], df["predicted"])
    print("\n" + "=" * 60)
    print(f"ROUTING ACCURACY: {acc:.3f}  ({df['correct'].sum()}/{len(df)})")
    print("=" * 60)
    print("\nPer-class report:\n")
    print(classification_report(df["label"], df["predicted"], labels=list(TOOLS), zero_division=0))

    print("Misrouted questions:")
    wrong = df[~df["correct"]]
    if wrong.empty:
        print("  (none)")
    else:
        for _, row in wrong.iterrows():
            print(f"  {row['label']} -> {row['predicted']}: {row['question']}")

    out_csv = os.path.join(RESULTS_DIR, "eval_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed results -> {out_csv}")

    save_confusion_matrix(df, os.path.join(RESULTS_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
