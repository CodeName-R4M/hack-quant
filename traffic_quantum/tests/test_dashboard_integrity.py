import json
import os
import re
import pandas as pd
import pytest

BANNED_PHRASES = [
    "zero signal flicker",
    "resilience",
]

# Known historical hardcoded numbers that must never re-appear
HISTORICAL_BANNED_NUMBERS = [
    "58.93",
    "54.90",
    "60.45",
    "55.41",
    "11.75",
    "13.4",
    "9.7",
    "0.37s",
]


def test_dashboard_contains_no_hardcoded_result_literals():
    """Scans dashboard.py to ensure numbers from result files are not typed-in literals."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dashboard_path = os.path.join(repo_root, "dashboard.py")
    assert os.path.exists(dashboard_path), f"dashboard.py not found at {dashboard_path}"

    with open(dashboard_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Collect all numeric literals from current result files
    dynamic_numbers = set()
    results_dir = os.path.join(repo_root, "results")

    sc_csv = os.path.join(results_dir, "scenario_benchmark.csv")
    if os.path.exists(sc_csv):
        df = pd.read_csv(sc_csv)
        for col in ["Wait Mean (s)", "Wait Std (s)"]:
            if col in df.columns:
                for val in df[col].dropna():
                    if isinstance(val, (int, float)) and val > 1.0:
                        dynamic_numbers.add(f"{val:.2f}")

    tradeoff_json = os.path.join(results_dir, "preemption_tradeoff.json")
    if os.path.exists(tradeoff_json):
        try:
            with open(tradeoff_json, "r", encoding="utf-8") as f:
                po_data = json.load(f)
            for entry in po_data.get("soft_preemption", []):
                for k in ["amb_time", "amb_time_saved", "extra_delay", "normal_wait"]:
                    if k in entry and isinstance(entry[k], (int, float)) and entry[k] > 1.0:
                        dynamic_numbers.add(f"{entry[k]:.2f}")
        except Exception:
            pass

    all_banned_numbers = set(HISTORICAL_BANNED_NUMBERS) | dynamic_numbers

    violations = []
    for i, line in enumerate(lines, 1):
        line_clean = line.strip()
        if line_clean.startswith("#"):
            continue
        
        # Check banned phrases
        for phrase in BANNED_PHRASES:
            if phrase in line_clean.lower():
                violations.append((i, phrase, line_clean))

        # Check banned result numbers in string literals
        pattern = rf'["\'](.*?)["\']'
        matches = re.findall(pattern, line_clean)
        for m in matches:
            for banned in all_banned_numbers:
                if banned in m:
                    violations.append((i, banned, line_clean))

    assert not violations, f"Found hardcoded result literals or banned phrases in dashboard.py: {violations}"

