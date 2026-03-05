#!/usr/bin/env python
"""Update benchmark performance tables in python/docs/index.rst and README.md.

Reads exported JSON results from benchmark scripts and patches the
summary tables in both the Sphinx documentation and the repo root README.

Usage:
    python benchmarks/update_docs.py --from-json benchmarks/.results/
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_RST_PATH = REPO_ROOT / "python" / "docs" / "index.rst"
README_MD_PATH = REPO_ROOT / "README.md"

# Maps JSON filenames to their benchmark category
JSON_FILES = {
    "lm.json": "lm",
    "wordseg.json": "wordseg",
    "tagger.json": "tagger",
    "hmm.json": "hmm",
    "chat.json": "chat",
}


def compute_speedup(rustling_time: float, python_time: float) -> float:
    """Compute speedup ratio (python_time / rustling_time)."""
    if rustling_time <= 0:
        return float("inf")
    return python_time / rustling_time


def extract_speedups(json_dir: Path) -> dict[str, float | dict[str, float]]:
    """Extract speedup values from exported JSON files.

    Returns a flat dict mapping task keys to speedup values.
    For LM Generate, returns a dict with "min" and "max" keys.
    """
    speedups: dict[str, float | dict[str, float]] = {}

    # --- Language Models ---
    lm_path = json_dir / "lm.json"
    if lm_path.exists():
        with open(lm_path) as f:
            lm_data = json.load(f)

        fit_speedups = []
        score_speedups = []
        generate_speedups = []

        for model_name in ["MLE", "Lidstone", "Laplace"]:
            models = lm_data["benchmarks"][0]["models"]
            if model_name not in models:
                continue

            model = models[model_name]
            for op, collector in [
                ("fit", fit_speedups),
                ("score", score_speedups),
                ("generate", generate_speedups),
            ]:
                if model[op]["rustling"] and model[op]["nltk"]:
                    s = compute_speedup(
                        model[op]["rustling"]["time_seconds"],
                        model[op]["nltk"]["time_seconds"],
                    )
                    collector.append(s)

        if fit_speedups:
            speedups["lm:Fit"] = statistics.mean(fit_speedups)
        if score_speedups:
            speedups["lm:Score"] = statistics.mean(score_speedups)
        if generate_speedups:
            speedups["lm:Generate"] = {
                "min": min(generate_speedups),
                "max": max(generate_speedups),
            }

    # --- Word Segmentation ---
    ws_path = json_dir / "wordseg.json"
    if ws_path.exists():
        with open(ws_path) as f:
            ws_data = json.load(f)

        for algo in ["LongestStringMatching"]:
            bench = ws_data["benchmarks"].get(algo, {})
            if bench.get("rustling") and bench.get("wordseg"):
                s = compute_speedup(
                    bench["rustling"]["time_seconds"],
                    bench["wordseg"]["time_seconds"],
                )
                speedups[f"wordseg:{algo}"] = s

    # --- POS Tagger ---
    tagger_path = json_dir / "tagger.json"
    if tagger_path.exists():
        with open(tagger_path) as f:
            tagger_data = json.load(f)

        benchmarks = tagger_data.get("benchmarks", {})
        if "training" in benchmarks and "speedup" in benchmarks["training"]:
            speedups["tagger:Training"] = benchmarks["training"]["speedup"]
        if "tagging" in benchmarks and "speedup" in benchmarks["tagging"]:
            speedups["tagger:Tagging"] = benchmarks["tagging"]["speedup"]

    # --- HMM ---
    hmm_path = json_dir / "hmm.json"
    if hmm_path.exists():
        with open(hmm_path) as f:
            hmm_data = json.load(f)

        benchmarks = hmm_data.get("benchmarks", {})
        for op in ["fit", "predict", "score"]:
            if op in benchmarks and "speedup" in benchmarks[op]:
                speedups[f"hmm:{op.capitalize()}"] = benchmarks[op]["speedup"]

    # --- CHAT Parsing ---
    chat_path = json_dir / "chat.json"
    if chat_path.exists():
        with open(chat_path) as f:
            chat_data = json.load(f)

        chat_tasks = [
            "from_zip",
            "from_strs",
            "utterances",
            "tokens",
        ]
        for task in chat_tasks:
            bench = chat_data["benchmarks"].get(task, {})
            if bench.get("rustling") and bench.get("pylangacq"):
                s = compute_speedup(
                    bench["rustling"]["time_seconds"],
                    bench["pylangacq"]["time_seconds"],
                )
                speedups[f"chat:{task}"] = s

    return speedups


def format_speedup(value: float | dict[str, float]) -> str:
    """Format a speedup value for summary tables.

    - Range dicts: "N--Mx" (e.g., "85--97x")
    - Values >= 2: rounded integer "Nx" (e.g., "9x")
    - Values < 2: one decimal "N.Nx" (e.g., "1.5x")
    """
    if isinstance(value, dict):
        lo = round(value["min"])
        hi = round(value["max"])
        return f"{lo}--{hi}x"
    if value >= 2:
        return f"{round(value)}x"
    return f"{value:.1f}x"


# Table row definitions: (component, task, speedup_key, vs_library)
TABLE_ROWS = [
    ("**Language Models**", "Fit", "lm:Fit", "NLTK"),
    ("", "Score", "lm:Score", "NLTK"),
    ("", "Generate", "lm:Generate", "NLTK"),
    (
        "**Word Segmentation**",
        "LongestStringMatching",
        "wordseg:LongestStringMatching",
        "wordseg",
    ),
    ("**POS Tagging**", "Training", "tagger:Training", "NLTK"),
    ("", "Tagging", "tagger:Tagging", "NLTK"),
    ("**HMM**", "Fit", "hmm:Fit", "hmmlearn"),
    ("", "Predict", "hmm:Predict", "hmmlearn"),
    ("", "Score", "hmm:Score", "hmmlearn"),
    ("**CHAT Parsing**", "Reading from a ZIP archive", "chat:from_zip", "pylangacq"),
    ("", "Reading from strings", "chat:from_strs", "pylangacq"),
    ("", "Parsing utterances", "chat:utterances", "pylangacq"),
    ("", "Parsing tokens", "chat:tokens", "pylangacq"),
]


def generate_rst_table(speedups: dict[str, float | dict[str, float]]) -> str:
    """Generate an RST list-table for python/docs/index.rst."""
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 25 25 15 35",
        "",
        "   * - Component",
        "     - Task",
        "     - Speedup",
        "     - vs.",
    ]
    for component, task, key, vs in TABLE_ROWS:
        if key not in speedups:
            continue
        formatted = format_speedup(speedups[key])
        lines.append(f"   * - {component}")
        lines.append(f"     - {task}")
        lines.append(f"     - **{formatted}**")
        lines.append(f"     - {vs}")
    return "\n".join(lines)


def generate_md_table(speedups: dict[str, float | dict[str, float]]) -> str:
    """Generate a GitHub-Flavored Markdown table for README.md."""
    lines = [
        "| Component | Task | Speedup | vs. |",
        "|---|---|---|---|",
    ]
    for component, task, key, vs in TABLE_ROWS:
        if key not in speedups:
            continue
        formatted = format_speedup(speedups[key])
        lines.append(f"| {component} | {task} | **{formatted}** | {vs} |")
    return "\n".join(lines)


def update_readme_md(speedups: dict[str, float | dict[str, float]]) -> bool:
    """Update the performance table in README.md.
    Returns True if changed.
    """
    content = README_MD_PATH.read_text()
    new_table = generate_md_table(speedups)

    # Replace existing markdown table between ## Performance and See [
    pattern = r"(\n\| Component \|.*?\n(?:\|.*\n)*)"
    match = re.search(pattern, content)
    if not match:
        print(
            "ERROR: Could not find performance table or marker in README.md",
            file=sys.stderr,
        )
        return False
    new_content = (
        content[: match.start()] + "\n" + new_table + "\n" + content[match.end() :]
    )

    if new_content == content:
        print("README.md: no changes needed")
        return False

    README_MD_PATH.write_text(new_content)
    print("README.md: updated performance table")
    return True


def update_index_rst(speedups: dict[str, float | dict[str, float]]) -> bool:
    """Update the performance table in python/docs/index.rst.
    Returns True if changed.
    """
    content = INDEX_RST_PATH.read_text()
    new_table = generate_rst_table(speedups)

    # Match the list-table block
    pattern = r"\.\. list-table::\n(?:[ \t]+[^\n]*\n|\n)*(?:   \* -[^\n]*\n(?:     -[^\n]*\n)*)+"  # noqa: E501
    match = re.search(pattern, content)
    if not match:
        print(
            "ERROR: Could not find performance table in python/docs/index.rst",
            file=sys.stderr,
        )
        return False

    new_content = content[: match.start()] + new_table + "\n" + content[match.end() :]
    if new_content == content:
        print("python/docs/index.rst: no changes needed")
        return False

    INDEX_RST_PATH.write_text(new_content)
    print("python/docs/index.rst: updated performance table")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update benchmark tables in README.md and python/docs/index.rst"
    )
    parser.add_argument(
        "--from-json",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory containing exported JSON files (lm.json, wordseg.json, etc.)",
    )
    args = parser.parse_args()

    json_dir = Path(args.from_json)
    if not json_dir.is_dir():
        print(f"ERROR: {json_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Check which JSON files are available
    available = []
    for filename in JSON_FILES:
        path = json_dir / filename
        if path.exists():
            available.append(filename)
        else:
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)

    if not available:
        print("ERROR: No JSON files found", file=sys.stderr)
        sys.exit(1)

    print(f"Reading results from: {json_dir}")
    print(f"Found: {', '.join(available)}")

    speedups = extract_speedups(json_dir)

    print(f"\nExtracted {len(speedups)} speedup values:")
    for key, value in speedups.items():
        print(f"  {key}: {format_speedup(value)}")

    print()
    update_readme_md(speedups)
    update_index_rst(speedups)


if __name__ == "__main__":
    main()
