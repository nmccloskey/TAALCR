import math
import io
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from taaalcr.powers import powers_coding_analysis as apc


# -------------------- Fixtures --------------------

@pytest.fixture
def utt_df_minimal():
    # Two speakers over one sample, with c1/c2 metrics present
    # Include a mix of T/MT/ST and some collab_repair markers
    return pd.DataFrame(
        {
            "sample_id": ["S1"] * 6,
            "speaker":   ["A", "A", "B", "B", "B", "A"],
            "utterance_id": [1, 2, 3, 4, 5, 6],
            "c1_turn_type": ["T", "T", "ST", "MT", "T", "ST"],
            "c2_turn_type": ["T", "MT", "ST", "MT", "T", "ST"],
            "c1_speech_units": [4, 3, 2, 5, 1, 2],
            "c2_speech_units": [4, 4, 2, 4, 1, 3],
            "c1_content_words": [3, 2, 1, 4, 1, 1],
            "c2_content_words": [3, 3, 1, 3, 1, 2],
            "c1_num_nouns": [1, 1, 0, 2, 0, 1],
            "c2_num_nouns": [1, 2, 0, 1, 0, 1],
            "c1_filled_pauses": [0, 1, 0, 0, 0, 0],
            "c2_filled_pauses": [0, 1, 0, 0, 0, 1],
            # Use integers for repairs (nunique should count non-null distinct IDs)
            "c1_collab_repair": [np.nan, 1, np.nan, np.nan, np.nan, 2],
            "c2_collab_repair": [np.nan, 1, np.nan, 3, np.nan, np.nan],
        }
    )


# -------------------- Unit tests: small helpers --------------------

def test_number_turns_happy_and_inherit():
    seq = ["T", "T", "MT", None, "ST", None, None]
    out = apc.number_turns(seq)
    # Expected numbering per type; missing inherits previous label except first -> 'X'
    assert out[0] == "T1"
    assert out[1] == "T2"
    assert out[2] == "MT1"
    assert out[3] == "MT1"  # inherited
    assert out[4] == "ST1"
    assert out[5] == "ST1"  # inherited
    assert out[6] == "ST1"  # inherited


def test_add_turn_labels_insertion_and_values(utt_df_minimal):
    df = apc.add_turn_labels(utt_df_minimal, ["c1", "c2"])
    # Columns inserted immediately after *_turn_type
    c1_idx = df.columns.get_loc("c1_turn_type")
    c2_idx = df.columns.get_loc("c2_turn_type")
    assert df.columns[c1_idx + 1] == "c1_turn_label"
    assert df.columns[c2_idx + 1] == "c2_turn_label"
    # Monotonic labels per type
    assert df.loc[df["c1_turn_type"] == "T", "c1_turn_label"].tolist()[:2] == ["T1", "T2"]


# -------------------- Summaries --------------------

def test_compute_level_summaries_shapes_and_ratios(utt_df_minimal):
    df = apc.add_turn_labels(utt_df_minimal, ["c1", "c2"])
    out = apc.compute_level_summaries(df, ["c1", "c2"])

    # Keys and non-empty
    assert set(out.keys()) == {"Turns", "Speakers", "Dialogs"}
    assert not out["Turns"].empty and not out["Speakers"].empty and not out["Dialogs"].empty

    sp = out["Speakers"]
    # Derived columns should exist
    for coder in ["c1", "c2"]:
        assert f"{coder}_mean_turn_length" in sp.columns
        assert f"{coder}_ratio_content_words_to_speech_units" in sp.columns
        assert f"{coder}_ratio_num_nouns_to_total_turns" in sp.columns
        assert f"{coder}_ratio_STs_to_turns" in sp.columns
        assert f"{coder}_ratio_MTs_to_turns" in sp.columns

    # Spot-check a stable ratio: for speaker A, coder c2:
    # speaker A rows: idx 0,1,5 with c2_content_words [3,3,2], c2_speech_units [4,4,3]
    # sums: cw=8, su=11 => ratio ≈ 0.7273
    a_c2 = sp[(sp["speaker"] == "A")]
    got = a_c2["c2_ratio_content_words_to_speech_units"].iloc[0]
    assert got == pytest.approx(8/11)

    # Dialog-level repairs: nunique (non-null) and proportion notna
    dl = out["Dialogs"]
    assert "c2_num_repairs" in dl.columns and "c2_prop_repairs" in dl.columns
    # For c2: repairs at utterances 2 and 4, two distinct IDs {1,3}, 2/6 non-null = 0.333...
    row = dl.iloc[0]
    assert row["c2_num_repairs"] == 2
    assert row["c2_prop_repairs"] == pytest.approx(2/6)


def test_format_just_c2_POWERS_strips_and_renames(utt_df_minimal):
    df = apc.add_turn_labels(utt_df_minimal, ["c1", "c2"])
    out = apc.compute_level_summaries(df, ["c1", "c2"])
    just = apc.format_just_c2_POWERS(out)

    # No c1_ columns and c2_ prefix removed
    for name, d in just.items():
        assert not any(c.startswith("c1_") for c in d.columns)
        assert not any(c.startswith("c2_") for c in d.columns)
    # Core columns present after de-prefix
    sp = just["Speakers"]
    assert {"mean_turn_length", "num_ST", "speech_units_sum"} <= set(sp.columns)


# -------------------- Reliability --------------------

def test_compute_reliability_with_stubbed_icc(monkeypatch, utt_df_minimal):
    # Stub pingouin.intraclass_corr to return a frame containing an ICC2 row
    def fake_icc(data, targets, raters, ratings, nan_policy="omit"):
        return pd.DataFrame(
            {"Type": ["ICC1", "ICC2", "ICC3"], "ICC": [0.1, 0.85, 0.9]}
        )
    monkeypatch.setattr(apc, "intraclass_corr", fake_icc)

    res = apc.compute_reliability(utt_df_minimal, "c1", "c2")

    icc = res["ContinuousReliability"]
    cat = res["CategoricalReliability"]
    # ICC contains one row per metric in TURN_AGG_COLS (if enough variability)
    assert set(icc.columns) == {"metric", "ICC2"}
    assert set(cat["metric"]) == {"turn_type", "collab_repair"}
    # Kappa is finite for turn_type because labels exist for both coders
    kappa_turn = float(cat.loc[cat["metric"] == "turn_type", "kappa"].iloc[0])
    assert not math.isnan(kappa_turn)


# -------------------- Excel helpers --------------------

def test_write_analysis_workbook_creates_sheets(tmp_path, utt_df_minimal):
    out_path = tmp_path / "POWERS_CodingAnalysis" / "test.xlsx"
    sheets = {
        "Turns": utt_df_minimal.head(2),
        "Speakers": pd.DataFrame({"x": [1, 2]}),
        # Check 31-char truncation (Excel sheet name limit)
        "ExtremelyLongSheetNameThatShouldBeTrimmed": pd.DataFrame({"y": [3]}),
    }
    apc.write_analysis_workbook(out_path, sheets)
    assert out_path.exists()

    # Verify sheet names present (with truncation)
    with pd.ExcelFile(out_path) as xf:
        names = set(xf.sheet_names)
    assert "Turns" in names and "Speakers" in names
    assert "ExtremelyLongSheetNameThatShoul" in names  # 31 chars


def test_match_reliability_files_merges_and_drops_c1(tmp_path):
    # Create matching coding and reliability files
    inp = tmp_path / "input"
    out = tmp_path / "out"
    inp.mkdir()

    coding = pd.DataFrame(
        {
            "utterance_id": [1, 2, 3],
            "sample_id": ["S1", "S1", "S1"],
            "c1_turn_type": ["T", "ST", "MT"],
            "c2_turn_type": ["T", "ST", "MT"],
            "c1_speech_units": [4, 2, 5],
            "c2_speech_units": [4, 2, 5],
        }
    )
    reliability = pd.DataFrame(
        {
            "utterance_id": [2, 3, 4],  # note: 4 will be dropped by inner merge
            "sample_id": ["S1", "S1", "S1"],
            "c2_turn_type": ["MT", "ST", "T"],
            "c3_turn_type": ["MT", "ST", "T"],
            "c1_extra_should_drop": [9, 9, 9],
        }
    )

    cod_path = inp / "myfile_PO W ERS_Coding_v1.xlsx".replace(" ", "")
    rel_path = inp / "myfile_POWERS_ReliabilityCoding_v1.xlsx"
    coding.to_excel(cod_path, index=False)
    reliability.to_excel(rel_path, index=False)

    apc.match_reliability_files(inp, out)
    merged_dir = out / "POWERS_ReliabilityAnalysis"
    # Should have written a merged file
    files = list(merged_dir.glob("*POWERS_ReliabilityCoding_Merged*.xlsx"))
    assert len(files) == 1

    merged = pd.read_excel(files[0])
    # Inner-merge rows (2,3) only; c1_* columns should be gone
    assert set(merged["utterance_id"]) == {2, 3}
    assert not any(col.startswith("c1") for col in merged.columns)


# -------------------- Orchestrator (smoke) --------------------

def test_analyze_POWERS_coding_smoke(tmp_path, utt_df_minimal, monkeypatch):
    # Prepare a tiny input workbook emulating a *POWERS_Coding*.xlsx file
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    fpath = inp / "AC12_POWERS_Coding_S1.xlsx"
    utt_df_minimal.to_excel(fpath, index=False)

    # Stub ICC to avoid pingouin dependency
    monkeypatch.setattr(
        apc, "intraclass_corr",
        lambda **kwargs: pd.DataFrame({"Type": ["ICC2"], "ICC": [0.7]})
    )

    apc.analyze_POWERS_coding(inp, out, reliability=False, just_c2_POWERS=False)

    # Workbook written under POWERS_CodingAnalysis
    out_book = next((out / "POWERS_CodingAnalysis").glob("*POWERS_Analysis*.xlsx"))
    assert out_book.exists()

    # Sheets exist (at least these)
    with pd.ExcelFile(out_book) as xf:
        names = set(xf.sheet_names)
    assert {"Turns", "Speakers", "Dialogs", "ContinuousReliability", "CategoricalReliability"} <= names
