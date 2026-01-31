import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from taalcr.powers import powers_coding_files as pcf


# -------------------- Tiny fakes & stubs --------------------

class FakeToken:
    def __init__(self, text, pos_, tag_="", like_num=False):
        self.text = text
        self.pos_ = pos_
        self.tag_ = tag_
        self.like_num = like_num

class FakeDoc(list):
    """Just a list of FakeToken with optional ._raw for convenience."""
    def __init__(self, tokens, raw_text=""):
        super().__init__(tokens)
        self._raw = raw_text

class FakeNLP:
    """Produces a FakeDoc per utterance string."""
    def pipe(self, utterances, batch_size=100, n_process=2):
        docs = []
        for utt in utterances:
            # Extremely light "tagging":
            words = utt.split()
            toks = []
            for w in words:
                lw = w.lower().strip(",.!?")
                if lw in {"is", "are", "am", "was", "were", "be", "been", "being", "will", "shall", "would", "should", "can", "could", "do", "does", "did"}:
                    toks.append(FakeToken(w, pos_="AUX", tag_="MD" if lw in {"will","shall","would","should","can","could"} else ""))
                elif lw in {"dog","cat","things","stuff"}:
                    toks.append(FakeToken(w, pos_="NOUN"))
                elif lw in {"alice","bob"}:
                    toks.append(FakeToken(w, pos_="PROPN"))
                elif lw in {"run","runs","see","know"}:
                    toks.append(FakeToken(w, pos_="VERB"))
                elif lw.endswith("ly"):
                    toks.append(FakeToken(w, pos_="ADV"))
                elif lw.isdigit():
                    toks.append(FakeToken(w, pos_="NUM", tag_="CD", like_num=True))
                else:
                    toks.append(FakeToken(w, pos_="X"))
            docs.append(FakeDoc(toks, raw_text=utt))
        return docs


# -------------------- Unit tests: small helpers --------------------

def test_compute_speech_units_ignores_xx_yy_and_cleans(monkeypatch):
    # Make cleaning a no-op for predictability
    monkeypatch.setattr(pcf, "_clean_clan_for_reliability", lambda s: s)
    utt = "hello xx xxx yy yyy world"
    # speech units count excludes xx/xxx/yy/yyy
    assert pcf.compute_speech_units(utt) == 2

@pytest.mark.parametrize(
    "utt,expected",
    [
        ("um", 1),
        ("uh uh", 2),
        ("&um, erm... er eh", 4),
        ("hmm (not counted) uhm (counts as 'um')", 1),
        ("", 0),
    ],
)
def test_count_fillers_regex(utt, expected):
    assert pcf.count_fillers(utt) == expected

def test_is_predicates_on_fake_tokens():
    t_aux = FakeToken("will", "AUX", tag_="MD")
    t_noun = FakeToken("dog", "NOUN")
    t_verb = FakeToken("run", "VERB")
    t_adv = FakeToken("quickly", "ADV")
    t_num = FakeToken("3", "NUM", tag_="CD", like_num=True)
    t_gen = FakeToken("things", "NOUN")

    assert pcf.is_aux_or_modal(t_aux) is True
    assert pcf.is_noun_or_propn(t_noun) is True
    assert pcf.is_main_verb(t_verb) is True
    assert pcf.is_ly_adverb(t_adv) is True
    assert pcf.is_numeral(t_num) is True
    assert pcf.is_generic(t_gen) is True

def test_count_content_words_from_doc_and_label_turn(monkeypatch):
    # Build a FakeDoc: NOUN, VERB, ADV(-ly), NUM, AUX (excluded), generic NOUN (excluded)
    tokens = [
        FakeToken("dog", "NOUN"),
        FakeToken("runs", "VERB"),
        FakeToken("quickly", "ADV"),
        FakeToken("3", "NUM", tag_="CD", like_num=True),
        FakeToken("will", "AUX", tag_="MD"),
        FakeToken("things", "NOUN"),
    ]
    doc = FakeDoc(tokens)
    # Count "all" (excludes AUX and GENERIC terms) => dog, runs, quickly, 3 = 4
    assert pcf.count_content_words_from_doc(doc, "all") == 4
    # Count "noun" => only NOUN/PROPN that are not GENERIC => "dog" = 1
    assert pcf.count_content_words_from_doc(doc, "noun") == 1

    # Label turn rules:
    assert pcf.label_turn("I don't know", count_content_words=0) == "MT"
    assert pcf.label_turn("alright, I see", count_content_words=0) == "MT"
    assert pcf.label_turn("okAy", count_content_words=0) == "MT"
    assert pcf.label_turn("uh ... well", count_content_words=2) == "ST"
    assert pcf.label_turn("... (silence)", count_content_words=0) == "T"


# -------------------- run_automation (with NLP stub) --------------------

def test_run_automation_success(monkeypatch):
    # Stub the NLP singleton to return our FakeNLP
    class FakeNLPmodel:
        def __init__(self): pass
        def get_nlp(self, name): return FakeNLP()

    monkeypatch.setattr(pcf, "NLPmodel", lambda: FakeNLPmodel())
    monkeypatch.setattr(pcf, "_clean_clan_for_reliability", lambda s: s)

    df = pd.DataFrame({
        "utterance": [
            "Alice will run quickly 3 um",  # 3 content words (PROPN counts as content? yes), + NUM => 4; AUX excluded
            "things are things",            # generic NOUNs + AUX => 0 content words
        ],
        "speaker": ["P1", "Clinician"],
        "sample_id": ["S1", "S1"]
    })

    out = pcf.run_automation(df.copy(), "1")
    # Speech units are token counts excluding xx/yy markers (none here)
    assert "c1_speech_units" in out and list(out["c1_speech_units"]) == [6, 3]
    assert "c1_filled_pauses" in out and list(out["c1_filled_pauses"]) == [1, 0]
    # Content words & nouns (see tokenizer rules above)
    assert "c1_content_words" in out and list(out["c1_content_words"]) == [4, 0]
    assert "c1_num_nouns" in out and list(out["c1_num_nouns"]) == [1, 0]
    # Turn type labels driven by content-word counts & minimal-turn patterns
    assert "c1_turn_type" in out and list(out["c1_turn_type"]) == ["ST", "T"]

def test_run_automation_model_failure_returns_df(monkeypatch, caplog):
    # Make NLPmodel raise to trigger graceful fallback
    class Failer:
        def get_nlp(self, name): raise RuntimeError("no model")

    monkeypatch.setattr(pcf, "NLPmodel", lambda: Failer())
    df = pd.DataFrame({"utterance": ["hello"], "speaker": ["P1"], "sample_id": ["S1"]})
    with caplog.at_level("ERROR"):
        out = pcf.run_automation(df.copy(), "2")
    assert out.equals(df)  # unchanged
    assert any("Failed to load NLP model" in rec.message for rec in caplog.records)


# -------------------- File workflows: make_POWERS_coding_files --------------------

def _write_minimal_utterances_xlsx(path: Path):
    df = pd.DataFrame({
        "sample_id": ["S1", "S1", "S2", "S2"],
        "speaker": ["P1", "Clinician", "P2", "P1"],
        "utterance": ["Alice run", "uh ok", "dog 3", "things are things"],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)

def test_make_POWERS_coding_files_creates_outputs(tmp_path, monkeypatch):
    # Arrange: Utterances workbook
    inp = tmp_path / "in"
    out = tmp_path / "out"
    ufile = inp / "AC12_Utterances.xlsx"
    _write_minimal_utterances_xlsx(ufile)

    # Stub segment -> single chunk containing all sample IDs
    def fake_segment(unique_ids, n):
        return [list(unique_ids)]
    # Stub assign_coders -> give pairs (c1, c2) per segment
    def fake_assign(coders):
        return [("A", "B")] * 1

    monkeypatch.setattr(pcf, "segment", fake_segment)
    monkeypatch.setattr(pcf, "assign_coders", fake_assign)

    # Use no tiers (labels), exclude Clinician, and skip automation to avoid spaCy
    pcf.make_POWERS_coding_files(
        tiers={}, frac=0.5, coders=["1","2","3"],
        input_dir=inp, output_dir=out,
        exclude_participants=["Clinician"],
        automate_POWERS=False,
    )

    outdir = out / "POWERS_Coding"
    assert outdir.exists()
    # Expect two files
    files = sorted(outdir.glob("*.xlsx"))
    assert len(files) == 2
    pc_file = next(f for f in files if "POWERS_Coding.xlsx" in f.name)
    rel_file = next(f for f in files if "POWERS_ReliabilityCoding.xlsx" in f.name)

    pc = pd.read_excel(pc_file)
    rel = pd.read_excel(rel_file)

    # Coding file must include coder id columns and initialized coder cols
    for col in ["c1_id", "c2_id"]:
        assert col in pc.columns
        assert set(pc[col].unique()) <= {"A", "B", np.nan}

    # All standard coder columns exist
    for col in pcf.coder_cols:
        assert col in pc.columns

    # Exclusions set to "NA" for excluded speakers
    na_cols = [c for c in pcf.coder_cols if c.startswith(("c1_", "c2_")) and not c.endswith("_id")]
    assert all(pc.loc[pc["speaker"] == "Clinician", na_cols] == "NA")

    # Reliability file drops c2_* and renames c1_* -> c3_*
    assert not any(c.startswith("c2_") for c in rel.columns)
    assert any(c.startswith("c3_") for c in rel.columns)


# -------------------- Reselection workflow --------------------

def test_reselect_POWERS_reliability_writes_new_subset(tmp_path, monkeypatch):
    # Set up a small Coding + Reliability pair
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()

    coding = pd.DataFrame({
        "sample_id": ["S1", "S1", "S2", "S3"],
        "speaker": ["P1", "P2", "P1", "Clinician"],
        "utterance": ["one", "two", "three", "four"]
    })
    # Initialize coder columns to satisfy drop/rename logic
    for col in pcf.coder_cols:
        coding[col] = np.nan
    coding_path = inp / "AC12_POWERS_Coding.xlsx"
    coding.to_excel(coding_path, index=False)

    reliability = coding[coding["sample_id"].isin(["S1"])].copy()
    # make it look like reliability file with c3_ columns only
    rel_cols_drop = [c for c in pcf.coder_cols if c.startswith("c2")]
    reliability.drop(columns=rel_cols_drop, inplace=True)
    reliability.rename(columns={c: c.replace("1", "3") for c in pcf.coder_cols if c.startswith("c1")}, inplace=True)
    reliability_path = inp / "AC12_POWERS_ReliabilityCoding.xlsx"
    reliability.to_excel(reliability_path, index=False)

    # Stub run_automation to no-op
    monkeypatch.setattr(pcf, "run_automation", lambda df, num: df)

    pcf.reselect_POWERS_reliability(
        input_dir=inp, output_dir=out,
        frac=0.5, exclude_participants=["Clinician"],
        automate_POWERS=False
    )

    # A new file should exist in POWERS_ReselectedReliability
    outdir = out / "POWERS_ReselectedReliability"
    files = list(outdir.glob("*POWERS_Reselected_ReliabilityCoding*.xlsx"))
    assert len(files) >= 1

    new_rel = pd.read_excel(files[0])
    # Should not include previously covered sample S1
    assert not set(new_rel["sample_id"]).issubset({"S1"})
    # Should have c3_ columns (no c2_)
    assert not any(c.startswith("c2_") for c in new_rel.columns)
    assert any(c.startswith("c3_") for c in new_rel.columns)
