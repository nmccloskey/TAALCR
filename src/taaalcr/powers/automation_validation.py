import pandas as pd
from pathlib import Path
from datetime import datetime
from taaalcr.utils.auxiliary import read_df
from rascal.utils.logger import logger, _rel
from rascal.utils.auxiliary import find_files, extract_transcript_data


def _filter_df(df, cols):
    try:
        return df[cols]
    except:
        return None

def select_validation_samples(input_dir: str | Path,
                              output_dir: str | Path,
                              stratify: list[str],
                              strata: int,
                              seed: int):
    """
    Generate a stratified random selection of samples for manual validation.

    Steps:
      - Reads utterance-level tables in `input_dir`.
      - Restricts to columns ["sample_id", "file"] + `stratify`.
      - Groups by the `stratify` fields and randomly selects up to `strata`
        samples per group (with reproducible seeding).
      - Assigns stratum numbers 1..k within each group.
      - Writes a consolidated selection table to Excel.
      - Optionally propagates stratum labels back into POWERS coding files
        (if present in `input_dir`).

    Parameters
    ----------
    input_dir : str | Path
        Directory containing utterance-level tables.
    output_dir : str | Path
        Directory to save the selection file and labeled coding files.
    stratify : list of str
        Column names to group by (e.g., ["site", "test"]).
    strata : int
        Number of samples per stratum group.
    seed : int
        Random seed for reproducibility.

    Raises
    ------
    RuntimeError
        If no valid tables or groups are found.
    ValueError
        If required columns are missing.
    """

    # Collect utterance tables
    transcript_tables = find_files(directories=[input_dir, output_dir],
                                   search_base="transcript_tables")
    utt_cols = ["sample_id", "file"] + stratify
    utt_dfs = [extract_transcript_data(tt) for tt in transcript_tables]

    filtered_udfs = [df for udf in utt_dfs if (df := _filter_df(udf, utt_cols)) is not None]
    if not filtered_udfs:
        raise RuntimeError("No transcript table files with required columns found.")
    sample_info = pd.concat(filtered_udfs, ignore_index=True)
    sample_info.drop_duplicates(inplace=True, ignore_index=True)

    # Stratified random selection
    selections = []
    for keys, g in sample_info.groupby(stratify, dropna=False, sort=False):
        g = g.sample(n=min(strata, len(g)), replace=False, random_state=seed)
        # If fewer than strata available, we still label from 1..len(g)
        k = min(strata, len(g))
        order = list(range(1, k + 1))
        g = g.assign(stratum_no=order[:len(g)])
        selections.append(g)

        if len(g) < strata:
            logger.warning(f"Group {keys} had only {len(g)} < {strata} samples; selecting all available.")

    if not selections:
        raise RuntimeError("No groups found to stratify. Check your input tables and --stratify fields.")

    sel = pd.concat(selections, ignore_index=True)
    # Order for readability
    sel = sel.loc[:, stratify + ["sample_id", "file", "stratum_no"]].sort_values(by=stratify + ["stratum_no", "sample_id"])

    ts = datetime.now().strftime("%y%m%d_%H%M")
    out_file = output_dir / f"powers_validation_selection_{ts}.xlsx"
    sel.to_excel(out_file, index=False)
    logger.info(f"Wrote selection table: {_rel(out_file)}")

    # Optionally collect empty POWERS coding tables
    # User would have run powers make with automate_POWERS=False
    pc_files = find_files(directories=[input_dir, output_dir],
                                       search_base="powers_coding")
    if pc_files:
        for pcf in pc_files:
            if (pcdf := read_df(pcf)) is not None and not pcdf.empty:
                labeled_pcdf = sel[["sample_id", "stratum_no"]].merge(pcdf, on=["sample_id"], how="right")
                out_pc_file = output_dir / pcf.name.replace("powers_coding", "powers_coding_labeled")
                labeled_pcdf.to_excel(out_pc_file, index=False)
                logger.info(f"Wrote labeled POWERS coding table: {_rel(out_pc_file)}")
    else:
        logger.warning("No POWERS coding files available for direct stratum number assignment.")


def validate_automation(input_dir: str | Path,
                        output_dir: str | Path,
                        selection_table: str | Path | None = None,
                        stratum_numbers: list[int] = []):
    """
    Merge automatic and manual POWERS coding files for validation.

    Steps:
      - Reads all coding files in `input_dir` from 'auto/' and 'manual/' subdirectories.
      - Drops duplicate columns and aligns auto/manual codes on `sample_id` (and optionally utterance_id).
      - If `stratum_no` is missing from manual files, merges in stratum assignments
        from a `selection_table`.
      - Optionally filters the merged data to only specified `stratum_numbers`.
      - Writes an Excel file with aligned auto vs manual codes to `output_dir`.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing POWERS coding files.
    output_dir : str | Path
        Directory where the merged validation file is written.
    selection_table : str | Path | None, optional
        Path to a selection table containing 'sample_id' and 'stratum_no'.
        Required if stratum labels are not already in the manual coding files.
    stratum_numbers : list of int, optional
        Restrict output to specific stratum numbers.

    Raises
    ------
    FileNotFoundError
        If no auto/manual folders are found.
    ValueError
        If selection table is missing required columns.
    """

    # Collect POWERS coding files
    auto_pc_files = find_files(directories=[input_dir / "auto", output_dir],
                               search_base="powers_coding")
    manual_pc_files = find_files(directories=[input_dir / "manual", output_dir],
                                 search_base="powers_coding")

    auto = [df for p in auto_pc_files if (df := read_df(p)) is not None]
    manual = [df for p in manual_pc_files if (df := read_df(p)) is not None]
    
    # Pair automatic and manual codes and prep for analyze POWERS
    if not auto or not manual:
        raise FileNotFoundError("Both /auto and /manual POWERS coding folders are required.")
    auto_df = pd.concat(auto, ignore_index=True)
    manual_df = pd.concat(manual, ignore_index=True)

    stratum_col = ["stratum_no"] if "stratum_no" in manual_df.columns else []
    manual_cols = [c for c in manual_df.columns if c.startswith("c2_")]
    auto_df.drop(columns=manual_cols, inplace=True, errors="ignore")

    merge_keys = ["utterance_id", "sample_id"]
    merged = auto_df.merge(manual_df[merge_keys + manual_cols + stratum_col], on=merge_keys, how="inner")

    # Select based on random assignment
    if not stratum_col:
        sel_df = pd.read_excel(selection_table)
        if "sample_id" not in sel_df.columns or "stratum_no" not in sel_df.columns:
            raise ValueError(f"selection table {selection_table} must contain a 'sample_id' and a 'stratum_no' column.")
        merged = merged.merge(sel_df[["sample_id", "stratum_no"]], on=["sample_id"], how="inner")
    
    if stratum_numbers:
        merged = merged[merged["stratum_no"].isin([int(n) for n in stratum_numbers])]

    output_dir = output_dir / "automation_validation"
    out_file = output_dir / "powers_coding_auto_vs_manual.xlsx"
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_excel(out_file, index=False)
