# tests/test_validate_automation.py
import pandas as pd
import pytest
from pathlib import Path
from taalcr.POWERS import automation_validation as va


def _mk_utt_df(site, test, ids, files=None):
    if files is None:
        files = [f"{sid}.cha" for sid in ids]
    return pd.DataFrame({
        "site": [site]*len(ids),
        "test": [test]*len(ids),
        "sample_id": ids,
        "file": files,
    })


def test_select_validation_samples_happy_path(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    # Fake utterance frames
    df_ac_pre  = _mk_utt_df("AC", "Pre",  [f"ACS{i}" for i in range(1, 6)])
    df_ac_post = _mk_utt_df("AC", "Post", [f"ACS{i}" for i in range(10, 17)])
    frames = [df_ac_pre, df_ac_post]

    utt_files = [input_dir / f"utt_{i}.parquet" for i in range(len(frames))]

    def fake_find_utt_files(in_dir, out_dir): return utt_files
    def fake_read_df(p): return frames[utt_files.index(p)]
    def fake_find_powers_coding_files(in_dir, out_dir): return []

    monkeypatch.setattr(va, "find_utt_files", fake_find_utt_files)
    monkeypatch.setattr(va, "read_df", fake_read_df)
    monkeypatch.setattr(va, "find_powers_coding_files", fake_find_powers_coding_files)

    # Run function (no return expected)
    va.select_validation_samples(
        input_dir=input_dir,
        output_dir=output_dir,
        stratify=["site", "test"],
        strata=2,
        seed=42,
    )

    # Verify an Excel file was written
    files = list(output_dir.glob("POWERS_validation_selection_*.xlsx"))
    assert files, "Selection Excel file should be written"
    sel = pd.read_excel(files[0])
    assert "stratum_no" in sel.columns


def test_select_validation_samples_no_valid_utt(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    def fake_find_utt_files(in_dir, out_dir): return [input_dir / "foo.parquet"]
    def fake_read_df(p): return pd.DataFrame({"not_sample_id": [1]})
    monkeypatch.setattr(va, "find_utt_files", fake_find_utt_files)
    monkeypatch.setattr(va, "read_df", fake_read_df)

    with pytest.raises(RuntimeError):
        va.select_validation_samples(
            input_dir=input_dir,
            output_dir=output_dir,
            stratify=["site", "test"],
            strata=2,
            seed=1,
        )


def test_validate_automation_happy_path_with_selection_filter(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    auto_dir = input_dir / "Auto"
    manual_dir = input_dir / "Manual"
    out_dir = tmp_path / "out"
    auto_dir.mkdir(parents=True)
    manual_dir.mkdir(parents=True)
    out_dir.mkdir()

    auto_files = [auto_dir / "auto_1.parquet"]
    manual_files = [manual_dir / "manual_1.parquet"]

    auto_df = pd.DataFrame({
        "sample_id": ["S1", "S1", "S2"],
        "utterance_id": [1, 2, 1],
        "auto_turn_label": ["A", "B", "A"],
    })
    manual_df = pd.DataFrame({
        "sample_id": ["S1", "S1", "S2"],
        "utterance_id": [1, 2, 1],
        "c2_turn_label": ["A", "B", "X"],
    })

    sel_path = out_dir / "selection.xlsx"
    pd.DataFrame({
        "sample_id": ["S1", "S2"],
        "stratum_no": [1, 2],
    }).to_excel(sel_path, index=False)

    def fake_find_powers_coding_files(in_dir, out_dir_unused):
        if Path(in_dir).name == "Auto":
            return auto_files
        if Path(in_dir).name == "Manual":
            return manual_files
        return []
    def fake_read_df(p):
        if Path(p) in auto_files: return auto_df
        if Path(p) in manual_files: return manual_df
        return None

    monkeypatch.setattr(va, "find_powers_coding_files", fake_find_powers_coding_files)
    monkeypatch.setattr(va, "read_df", fake_read_df)

    va.validate_automation(
        input_dir=input_dir,
        output_dir=out_dir,
        selection_table=sel_path,
        stratum_numbers=[1],
    )

    files = list((out_dir / "AutomationValidation").glob("POWERS_Coding_Auto_vs_Manual.xlsx"))
    assert files, "Merged validation Excel should be written"
    merged = pd.read_excel(files[0])
    assert "auto_turn_label" in merged.columns
    assert "c2_turn_label" in merged.columns
    assert "stratum_no" in merged.columns


def test_validate_automation_requires_auto_and_manual(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    (input_dir / "Auto").mkdir(parents=True)
    (input_dir / "Manual").mkdir(parents=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_find_powers_coding_files(in_dir, out_dir_unused):
        if Path(in_dir).name == "Auto":
            return [Path(in_dir) / "auto.parquet"]
        return []
    def fake_read_df(p): return pd.DataFrame({"sample_id": [], "utterance_id": []})

    monkeypatch.setattr(va, "find_powers_coding_files", fake_find_powers_coding_files)
    monkeypatch.setattr(va, "read_df", fake_read_df)

    with pytest.raises(FileNotFoundError):
        va.validate_automation(
            input_dir=input_dir,
            output_dir=out_dir,
            selection_table=None,
        )
