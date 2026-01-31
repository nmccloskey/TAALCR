import math
import numpy as np
import pandas as pd
import pytest

from taalcr.convo_turns import digital_convo_turns_analysis as dct


# ---------- Fixtures ----------

@pytest.fixture
def simple_strings():
    # Two short sequences in one group/session across two bins
    # "0.1..12.0" -> digits: 0,1,1,2,0
    # "0.2.20..1" -> digits: 0,2,2,0,1
    return ["0.1..12.0", "0.2.20..1"]


@pytest.fixture
def simple_df(simple_strings):
    return pd.DataFrame(
        {
            "group": ["G1", "G1"],
            "session": ["S1", "S1"],
            "bin": [1, 2],
            "turns": simple_strings,
        }
    )


# ---------- Unit tests for small utilities ----------

def test_extract_turn_counts_and_stats():
    s = "0.1..12.0"
    # Counts of speakers ignoring dots
    counts = dct.extract_turn_counts(s)
    assert counts == {"0": 2, "1": 2, "2": 1}

    # Counts incl. single vs double dot markers
    tc, m1, m2 = dct.extract_turn_stats(s)
    assert tc == {"0": 2, "1": 2, "2": 1}
    # From "0.1..12.0": single dots after 0 and 2, a double dot after first 1
    assert m1 == {"0": 1, "2": 1}
    assert m2 == {"1": 1}


def test_mean_absolute_change():
    x = np.array([1, 4, 2], dtype=float)  # diffs: 3, 2 -> mean abs change = 2.5
    assert dct.mean_absolute_change(x) == pytest.approx(2.5)


def test_clinician_to_participant_ratio_divzero():
    df = pd.DataFrame({"speaker": ["0", "0"], "turns": [3, 2]})
    # No participant turns -> NaN
    ratio = dct.clinician_to_participant_ratio(df)
    assert math.isnan(ratio)


# ---------- Higher-level computations on prepared data ----------

def test_build_transition_matrix(simple_strings):
    sequences = [dct.extract_sequence(s) for s in simple_strings]
    M = dct.build_transition_matrix(sequences)

    # rows must sum to 1 (or be all zeros if no outgoing edges)
    row_sums = M.sum(axis=1).to_numpy()
    for rs in row_sums:
        assert rs == pytest.approx(1.0)

    # spot-check a few probabilities from hand-computed transitions:
    # From 0 -> {1,2} equally 2/3 and 1/3 across the two sequences
    assert M.loc["0", "1"] == pytest.approx(2/3)
    assert M.loc["0", "2"] == pytest.approx(1/3)
    # From 1 -> {1,2}: 0.5, 0.5 in the first sequence
    assert M.loc["1", "1"] == pytest.approx(0.5)
    assert M.loc["1", "2"] == pytest.approx(0.5)


def test_compute_transition_metrics(simple_df):
    out = dct.compute_transition_metrics(simple_df)
    assert "transition_matrices" in out and "speaker_ratios" in out
    assert set(out["transition_matrices"].keys()) == {"G1"}
    ratios = out["speaker_ratios"]
    assert list(ratios.columns) == [
        "group", "participant_to_participant", "participant_to_clinician", "clinician_to_participant"
    ]
    row = ratios.iloc[0]
    # From the two sequences:
    # ptp = 1->1 + 1->2 + 2->1 + 2->2 = 0.5 + 0.5 + 0 + 1/3 = 1.3333
    # ptc = 1->0 + 2->0 = 0 + 2/3 = 0.6667
    # cpp = 0->1 + 0->2 = 2/3 + 1/3 = 1.0
    assert row["participant_to_participant"] == pytest.approx(4/3)
    assert row["participant_to_clinician"] == pytest.approx(2/3)
    assert row["clinician_to_participant"] == pytest.approx(1.0)


def test_analyze_missing_required_column_raises():
    # _analyze_convo_turns_file requires ['group','turns']
    with pytest.raises(ValueError):
        dct._analyze_convo_turns_file(pd.DataFrame({"turns": ["0.1.2"]}))


def test_compute_levels_and_session_metrics(simple_df):
    # Run the core analyzer (in-memory use only; no Excel I/O here)
    out = dct._analyze_convo_turns_file(simple_df)

    # Speaker level expectations (totals across both bins)
    sp = out["speaker_level"]
    # Speakers 0,1,2 totals from the two strings combined:
    # totals: 0->4, 1->3, 2->3; mark1: 0->2, 1->0, 2->2; mark2: 0->1, 1->1, 2->0
    def row_for(s):
        r = sp[(sp["group"] == "G1") & (sp["speaker"] == s)].iloc[0]
        return r

    r0 = row_for("0")
    assert r0["total_turns"] == 4
    assert r0["mark1"] == 2
    assert r0["mark2"] == 1
    assert r0["prop_mark1"] == pytest.approx(0.5)
    assert r0["prop_mark2"] == pytest.approx(0.25)

    r1 = row_for("1")
    assert r1["total_turns"] == 3
    assert r1["mark1"] == 0
    assert r1["mark2"] == 1
    assert r1["prop_mark1"] == pytest.approx(0.0)
    assert r1["prop_mark2"] == pytest.approx(1/3)

    r2 = row_for("2")
    assert r2["total_turns"] == 3
    assert r2["mark1"] == 2
    assert r2["mark2"] == 0
    assert r2["prop_mark1"] == pytest.approx(2/3)
    assert r2["prop_mark2"] == pytest.approx(0.0)

    # Group level totals across all speakers
    gl = out["group_level"]
    g = gl[gl["group"] == "G1"].iloc[0]
    assert g["total_turns"] == 10
    assert g["total_mark1"] == 4
    assert g["total_mark2"] == 2
    assert g["prop_mark1"] == pytest.approx(0.4)
    assert g["prop_mark2"] == pytest.approx(0.2)

    # Session level: entropy and clinician/participant ratio checks
    sess = out["session_level"]
    s = sess[(sess["group"] == "G1") & (sess["session"] == "S1")].iloc[0]
    # totals mirror group totals in this simple case
    assert s["total_turns"] == 10
    assert s["total_mark1"] == 4
    assert s["total_mark2"] == 2
    # turn entropy of [4,3,3]
    assert s["turn_entropy"] == pytest.approx(1.0888999753452238)
    # clinician (0) / participants (1+2) = 4 / 6
    assert s["clinician_participant_ratio"] == pytest.approx(4/6)

    # Participation level: per-speaker share of session turns
    part = out["participation_level"].set_index("speaker")
    assert part.loc["0", "proportion_of_session_turns"] == pytest.approx(4/10)
    assert part.loc["1", "proportion_of_session_turns"] == pytest.approx(3/10)
    assert part.loc["2", "proportion_of_session_turns"] == pytest.approx(3/10)

    # Bin level exists and proportions within each (group, session, bin) sum to 1
    bin_df = out["bin_level"]
    grp = ["group", "session", "bin"]
    sums = bin_df.groupby(grp)["proportion_of_bin_turns"].sum().to_numpy()
    assert np.allclose(sums, 1.0)


def test_compute_bin_level_columns(simple_df):
    # Build the turn_totals (as _analyze_convo_turns_file would)
    rows = []
    for _, row in simple_df.iterrows():
        tc, m1, m2 = dct.extract_turn_stats(row["turns"])
        for spk in set(tc) | set(m1) | set(m2):
            rows.append(
                {
                    "group": row["group"],
                    "session": row["session"],
                    "speaker": spk,
                    "bin": row["bin"],
                    "turns": int(tc.get(spk, 0)),
                    "mark1": int(m1.get(spk, 0)),
                    "mark2": int(m2.get(spk, 0)),
                }
            )
    turn_totals = pd.DataFrame(rows)
    out = dct.compute_bin_level(turn_totals, grouping_cols=["group", "session", "bin"])

    for col in [
        "proportion_of_bin_turns",
        "prop_mark1",
        "prop_mark2",
        "proportion_of_bin_mark1",
        "proportion_of_bin_mark2",
    ]:
        assert col in out.columns


def test_summarize_basic():
    df = pd.DataFrame(
        {
            "a": [1.0, 3.0, 2.0],
            "b": [2.0, 2.0, 2.0],
            "non_numeric": ["x", "y", "z"],
        }
    )
    sm = dct.summarize(df, "demo")
    # Summary has one row per numeric column
    assert set(sm["metric"]) == {"a", "b"}
    assert all(sm["level"] == "demo")
    # For column b, std == 0 -> cv == 0
    brow = sm[sm["metric"] == "b"].iloc[0]
    assert brow["std"] == pytest.approx(0.0)
    assert brow["cv"] == pytest.approx(0.0)
