import re
import argparse
import pandas as pd
from rascal.utils.logger import logger, _rel


def read_df(file_path):
    try:
        df = pd.read_excel(str(file_path))
        logger.info(f"Successfully read file: {_rel(file_path)}")
        return df
    except Exception as e:
        logger.error(f"Failed to read file {_rel(file_path)}: {e}")
        return None

def parse_stratify_fields(values: list[str] | None) -> list[str]:
    """
    Accepts:
      --stratify site test
      --stratify site,test
      --stratify "site, test"
      --stratify site --stratify test
    """
    if not values:
        return []
    items: list[str] = []
    for v in values:
        parts = re.split(r"[,\s]+", v.strip())
        parts = [x for x in parts if x]
        items.extend(parts)
    # preserve order but dedupe
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


# ---------------------------------------------------------------------
# Unified parser builder
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the TAALCR argument parser (shared by CLI and direct run)."""
    parser = argparse.ArgumentParser(description="TAALCR CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- turns ---
    subparsers.add_parser("turns", help="Analyze digital conversation turns")

    # --- powers ---
    powers_parser = subparsers.add_parser("powers", help="POWERS coding workflow")
    powers_parser.add_argument(
        "action",
        choices=["make", "analyze", "evaluate", "reselect", "select", "validate"],
        help="POWERS step"
    )

    # --- powers: select ---
    powers_parser.add_argument(
        "--stratify",
        action="append",
        help="Fields to stratify by. Accepts repeated flags or comma/space-delimited list "
             "(e.g., --stratify site,test)."
    )
    powers_parser.add_argument(
        "--strata",
        type=int,
        default=5,
        help="Number of samples to draw per stratum (default: 5)."
    )
    powers_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for selection (default: 42)."
    )

    # --- powers: validate ---
    powers_parser.add_argument(
        "--selection",
        type=str,
        default=None,
        help="Optional path to a selection .xlsx to restrict validation (output of 'powers select')."
    )
    powers_parser.add_argument(
        "--numbers",
        type=str,
        default=None,
        help="Optional selection of stratum numbers to include in validation."
    )

    # --- global options ---
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)."
    )
    return parser
