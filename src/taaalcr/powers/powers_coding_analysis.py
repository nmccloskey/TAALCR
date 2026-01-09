import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pingouin import intraclass_corr
from sklearn.metrics import cohen_kappa_score

from rascal.utils.logger import logger, _rel
from rascal.utils.auxiliary import find_files

TURN_AGG_COLS = ["speech_units", "content_words", "num_nouns", "filled_pauses"]

def match_reliability_files(input_dir, output_dir):
    """
    Match and merge POWERS coding files with reliability coding files.

    This function searches `input_dir` for baseline POWERS coding Excel files
    (*powers_coding*.xlsx) and their corresponding reliability re-coding files
    (*powers_reliability_coding*.xlsx). For each matched pair, it merges data on
    utterance_id and sample_id, drops coder-1 columns, and writes a new merged
    file into `{output_dir}/powers_reliability_analysis`.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing both coding and reliability coding files.
    output_dir : str or Path
        Root directory where merged files will be saved.

    Returns
    -------
    None
        Writes merged Excel files named
        *powers_reliability_coding_merged*.xlsx into the reliability directory.
    """

    # Make POWERS Reliability Analysis folder.
    output_dir = Path(output_dir)
    powers_reliability_dir = output_dir / "powers_reliability_analysis"
    try:
        powers_reliability_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {_rel(powers_reliability_dir)}")
    except Exception as e:
        logger.error(f"Failed to create directory {_rel(powers_reliability_dir)}: {e}")
        return

    pc_files = find_files(directories=[input_dir, output_dir],
                          search_base="powers_coding")
    rel_files = find_files(directories=[input_dir, output_dir],
                           search_base="powers_reliability_coding")

    # Match original coding and reliability files.
    for rel in tqdm(rel_files, desc="Analyzing POWERS reliability coding..."):
        for cod in pc_files:
            if cod.name.replace("powers_coding", "powers_reliability_coding") == rel.name:
                try:
                    powers_cod = pd.read_excel(cod)
                    pc_rel = pd.read_excel(rel)
                    logger.info(f"Processing coding file: {_rel(cod)} and reliability file: {_rel(rel)}")
                except Exception as e:
                    logger.error(f"Failed to read files {_rel(cod)} or {_rel(rel)}: {e}")
                    continue
    
                pc_rel_merge_cols = ["utterance_id", "sample_id"] + [col for col in pc_rel.columns if col.startswith("c3")]
                merged = powers_cod.merge(pc_rel[pc_rel_merge_cols], on=["utterance_id", "sample_id"], how="inner")
                merged.drop(columns=[col for col in merged.columns if col.startswith("c1")], inplace=True)

                merged_filename = Path(powers_reliability_dir, rel.name.replace("powers_reliability_coding", "powers_reliability_coding_merged"))

                try:
                    merged_filename.parent.mkdir(parents=True, exist_ok=True)
                    merged.to_excel(merged_filename, index=False)
                    logger.info(f"Wrote merged POWERS coding & reliability file: {_rel(merged_filename)}")
                except Exception as e:
                    logger.error(f"Failed to write merged POWERS coding & reliability file {_rel(merged_filename)}: {e}")

# ----------------------------- Helpers ----------------------------- #

def number_turns(turn_type_col):
    """Assign sequential numeric labels to turns of type T, MT, ST, or NV."""
    new_col = []
    turn_counts = {"T": 0, "MT": 0, "ST": 0, "NV": 0}
    for t in turn_type_col:
        if t not in turn_counts:
            try:
                new_t = new_col[-1]
            except Exception as e:
                print(
                    f"Blank turn cell cannot inherit previous cell's value: {e}. Marking as error (X)"
                )
                new_t = "X"
        else:
            turn_counts[t] += 1
            new_t = f"{t}{turn_counts.get(t, 'X')}"
        new_col.append(new_t)
    return new_col

def count_value(val):
    def inner(series):
        return np.sum(series == val)
    return inner

def add_turn_labels(utt_df: pd.DataFrame, coders: list[str]) -> pd.DataFrame:
    """
    Insert sequential turn labels (e.g., T1/MT2) for each coder.

    Parameters
    ----------
    utt_df : pandas.DataFrame
        Must contain f"{coder}_turn_type" columns for each coder in `coders`.
    coders : list of str
        Coder prefixes like ["c1","c2"].

    Returns
    -------
    pandas.DataFrame
        A copy of `utt_df` with new f"{coder}_turn_label" columns inserted
        immediately after each corresponding f"{coder}_turn_type" column.
    """
    df = utt_df.copy()
    for coder in coders:
        col_type = f"{coder}_turn_type"
        if col_type not in df.columns:
            raise KeyError(f"Missing column: {col_type}")
        insert_at = df.columns.to_list().index(col_type) + 1
        df.insert(insert_at, f"{coder}_turn_label", number_turns(df[col_type]))
    return df

def compute_level_summaries(utt_df: pd.DataFrame, coders: list[str]) -> dict[str, pd.DataFrame]:
    """
    Build turn-, speaker-, and dialog-level summaries for POWERS coding.

    Parameters
    ----------
    utt_df : pandas.DataFrame
        Utterance-level table containing columns needed by TURN_AGG_COLS
        for each coder (e.g., f"{coder}_{metric}").
    coders : list of str
        Coder prefixes like ["c1","c2"].

    Returns
    -------
    dict
        {
          "Turns":    DataFrame (per sample_id x speaker x (c1_label,c2_label)),
          "Speakers": DataFrame (per sample_id x speaker),
          "Dialogs":  DataFrame (per sample_id)
        }

    Notes
    -----
    Expects globals: TURN_AGG_COLS, count_value.
    """
    c1, c2 = coders

    # 1) Turn-level
    auto_summed = {
        f"{coder}_{col}_sum": (f"{coder}_{col}", "sum")
        for coder in coders
        for col in TURN_AGG_COLS
    }
    turn_df = (
        utt_df.groupby(by=["sample_id", "speaker", f"{c1}_turn_label", f"{c2}_turn_label"])
              .agg(**auto_summed)
              .reset_index()
    )

    # 2) Speaker-level
    speaker_aggs = {
        **auto_summed,
        **{f"{coder}_total_turns": (f"{coder}_turn_label", "nunique") for coder in coders},
        **{
            f"{coder}_num_{ttype}": (f"{coder}_turn_type", count_value(ttype))
            for coder in coders for ttype in ["T", "MT", "ST", "NV"]
        },
    }
    speaker_df = (
        utt_df.groupby(["sample_id", "speaker"])
              .agg(**speaker_aggs)
              .reset_index()
    )

    # Derived ratios/means (use *_sum columns produced above)
    for coder in coders:
        # mean turn length
        speaker_df[f"{coder}_mean_turn_length"] = (
            speaker_df[f"{coder}_speech_units_sum"] / speaker_df[f"{coder}_total_turns"].replace(0, np.nan)
        )

        # ratios: (num_nouns|content_words)_sum / (speech_units_sum | total_turns | num_ST)
        numerator_cols = {"num_nouns": f"{coder}_num_nouns_sum",
                          "content_words": f"{coder}_content_words_sum"}
        denom_map = {
            "speech_units": f"{coder}_speech_units_sum",
            "total_turns":  f"{coder}_total_turns",
            "num_ST":       f"{coder}_num_ST",
        }
        for num_key, num_col in numerator_cols.items():
            for denom_key, denom_col in denom_map.items():
                out = f"{coder}_ratio_{num_key}_to_{denom_key}"
                speaker_df[out] = speaker_df[num_col] / speaker_df[denom_col].replace(0, np.nan)

        # ratios of turn-type counts to total turns
        for ttype in ["ST", "MT"]:
            speaker_df[f"{coder}_ratio_{ttype}s_to_turns"] = (
                speaker_df[f"{coder}_num_{ttype}"] / speaker_df[f"{coder}_total_turns"].replace(0, np.nan)
            )

    # 3) Dialog-level (per sample)
    sample_aggs = {**auto_summed}
    for coder in coders:
        sample_aggs.update({
            f"{coder}_num_repairs": (f"{coder}_collab_repair", "nunique"),
            f"{coder}_prop_repairs": (f"{coder}_collab_repair", lambda x: x.notna().mean()),
        })
    sample_df = utt_df.groupby("sample_id").agg(**sample_aggs).reset_index()

    return {"Utterances": utt_df, "Turns": turn_df, "Speakers": speaker_df, "Dialogs": sample_df}

def format_just_c2_powers(df_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Strip c1_* columns and de-prefix c2_* -> bare names (for final resolved data).

    Parameters
    ----------
    df_dict : dict[str, pandas.DataFrame]
        Output of `compute_level_summaries`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        New dict with c1_* dropped and c2_* columns renamed by removing "c2_".
    """
    out = {}
    for name, df in df_dict.items():
        dd = df.copy()
        dd = dd.drop(columns=[c for c in dd.columns if c.startswith("c1_")], errors="ignore")
        dd = dd.rename(columns={c: c.replace("c2_", "") for c in dd.columns})
        out[name] = dd
    return out

def compute_reliability(utt_df: pd.DataFrame, c1: str, c2: str, investigators: list) -> dict[str, pd.DataFrame]:
    """
    Compute utterance-level reliability between two coders.

    Parameters
    ----------
    utt_df : pandas.DataFrame
        Utterance-level table with coder-specific columns f"{c}_<metric>".
    c1, c2 : str
        Coder prefixes (e.g., "c1", "c2").
    investigators : list
        Speakers to exclude from reliability metrics on content_words, num_nouns, & filled_pauses

    Returns
    -------
    dict
        {
          "continuous_reliability": DataFrame (ICC2 per metric in TURN_AGG_COLS),
          "categorical_reliability": DataFrame (kappa & agreement for turn_type and collab_repair)
        }

    Notes
    -----
    Expects globals: TURN_AGG_COLS, intraclass_corr, cohen_kappa_score.
    """
    # Continuous metrics: ICC(2,1)
    icc_rows = []
    for col in TURN_AGG_COLS:
        try:
            # Calculate reliability only on metrics which were recorded for all speakers
            if col != "speech_units":
                filtered_utt_df = utt_df[~utt_df["speaker"].isin(investigators)].copy()
            else:
                filtered_utt_df = utt_df.copy()

            tmp = filtered_utt_df[[f"{c1}_{col}", f"{c2}_{col}"]]
            tmp = tmp.dropna(how="any")
            if tmp.shape[0] < 2 or tmp.nunique().min() <= 1:
                logger.warning(f"Insufficient variability for ICC on {col}")
                continue
            tmp_long = tmp.melt(var_name="coder", value_name="score", ignore_index=False).reset_index()
            tmp_long.rename(columns={"index": "target"}, inplace=True)
            res = intraclass_corr(
                data=tmp_long, targets="target", raters="coder", ratings="score", nan_policy="omit"
            )
            icc2_row = res.query('Type == "ICC2"')
            if not icc2_row.empty:
                icc_rows.append({"metric": col, "ICC2": icc2_row.iloc[0]["ICC"]})
        except Exception as e:
            logger.error(f"ICC failure for {col}: {e}")
    icc_df = pd.DataFrame(icc_rows)

    # Categorical reliability
    y1 = utt_df.get(f"{c1}_turn_type", pd.Series(dtype=str)).fillna("MISSING").astype(str)
    y2 = utt_df.get(f"{c2}_turn_type", pd.Series(dtype=str)).fillna("MISSING").astype(str)
    try:
        kappa_turn = cohen_kappa_score(y1, y2)
        agree_turn = (y1 == y2).mean()
    except Exception:
        kappa_turn, agree_turn = np.nan, np.nan

    # Clean collab_repair columns: treat 0 as np.nan
    for c in [c1, c2]:
        col = f"{c}_collab_repair"
        if col in utt_df.columns:
            utt_df[col] = np.where(utt_df[col] == 0, np.nan, utt_df[col])

    # Binary presence/absence
    c1_bin = (~utt_df.get(f"{c1}_collab_repair", pd.Series(index=utt_df.index)).isna()).astype(int)
    c2_bin = (~utt_df.get(f"{c2}_collab_repair", pd.Series(index=utt_df.index)).isna()).astype(int)
    try:
        if len(np.unique(c1_bin)) > 1 or len(np.unique(c2_bin)) > 1:
            kappa_repair = cohen_kappa_score(c1_bin, c2_bin)
            agree_repair = (c1_bin == c2_bin).mean()
        else:
            kappa_repair, agree_repair = np.nan, np.nan
    except Exception:
        kappa_repair, agree_repair = np.nan, np.nan

    cat_rel = pd.DataFrame([
        {"metric": "turn_type", "kappa": kappa_turn, "agreement": agree_turn},
        {"metric": "collab_repair", "kappa": kappa_repair, "agreement": agree_repair},
    ])

    return {
        "continuous_reliability": icc_df,
        "categorical_reliability": cat_rel,
    }

def write_analysis_workbook(out_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """
    Write multiple sheets to a single Excel workbook.

    Parameters
    ----------
    out_path : pathlib.Path
        Output .xlsx path (parent directories created if needed).
    sheets : dict[str, pandas.DataFrame]
        Mapping of sheet name -> DataFrame to write.

    Returns
    -------
    None
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


# ----------------------------- Orchestrator ----------------------------- #

def analyze_powers_coding(input_dir, output_dir, reliability=False, just_c2_powers=False, exclude_participants=[]):
    """
    Analyze POWERS coding to produce turn/speaker/dialog summaries and (optionally) reliability.

    Parameters
    ----------
    input_dir : str or Path
        Directory to search recursively for input files.
    output_dir : str or Path
        Root directory to write analysis workbooks.
    reliability : bool, optional
        If True, expects merged reliability files and computes reliability for c2 vs c3.
        If False, expects standard coding files (c1 vs c2).
    just_c2_powers : bool, optional
        If True, drop c1_* columns and rename c2_* -> bare names (no reliability sheets).
    exclude_participants : list, optional
        Exclude from POWERS reliability measures on content_words, num_nouns, & filled_pauses (mostly for validating automation)

    Returns
    -------
    None
        Writes one workbook per input file under:
        - {output_dir}/powers_coding_analysis or
        - {output_dir}/powers_reliability_analysis
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    out_folder = "powers_coding_analysis" if not reliability else "powers_reliability_analysis"
    pc_analysis_dir = output_dir / out_folder
    try:
        pc_analysis_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {_rel(pc_analysis_dir)}")
    except Exception as e:
        logger.error(f"Failed to create POWERS analysis directory {_rel(pc_analysis_dir)}: {e}")
        return

    if not reliability:
        pc_files = find_files(directories=[input_dir, output_dir],
                              search_base="powers_coding")
        coders = ["c1", "c2"]
    else:
        # If your merged reliability files are stored elsewhere, change to input_dir accordingly.
        pc_files = list(pc_analysis_dir.rglob("*powers_reliability_coding_merged*.xlsx"))
        coders = ["c2", "c3"]
    c1, c2 = coders

    for pc_file in tqdm(pc_files, desc="Analyzing POWERS coding..."):
        try:
            utt_df = pd.read_excel(pc_file)
            logger.info(f"Processing file: {_rel(pc_file)}")
        except Exception as e:
            logger.error(f"Failed to read file {_rel(pc_file)}: {e}")
            continue

        # 1) turn labels
        try:
            utt_df = add_turn_labels(utt_df, coders)
        except Exception as e:
            logger.error(f"Failed to add turn labels for {_rel(pc_file)}: {e}")
            continue

        # 2) summaries
        try:
            df_dict = compute_level_summaries(utt_df, coders)
        except Exception as e:
            logger.error(f"Failed to summarize levels for {_rel(pc_file)}: {e}")
            continue

        # 3) either strip to final c2-* only, or compute reliability
        if just_c2_powers:
            df_dict = format_just_c2_powers(df_dict)
        else:
            try:
                rel = compute_reliability(utt_df, c1, c2, exclude_participants)
                df_dict.update(rel)
            except Exception as e:
                logger.error(f"Reliability computation failed for {_rel(pc_file)}: {e}")

        # 4) write
        to_replace = "coding" if not reliability else "coding_merged"
        out_file = pc_analysis_dir / (pc_file.stem.replace(to_replace, "_analysis") + ".xlsx")
        try:
            write_analysis_workbook(out_file, df_dict)
            logger.info(f"Wrote analysis workbook: {_rel(out_file)}")
        except Exception as e:
            logger.error(f"Failed to write {_rel(out_file)}: {e}")
