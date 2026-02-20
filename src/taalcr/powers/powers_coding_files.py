import re
import spacy
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import contractions
from pathlib import Path

from rascal.utils.logger import logger, _rel
from rascal.coding.coding_files import segment, assign_coders
from rascal.utils.auxiliary import find_files, extract_transcript_data
from rascal.transcripts.transcription_reliability_evaluation import process_utterances


POWERS_cols = [
    "id", "turn_type", "speech_units", "content_words", "num_nouns", "filled_pauses", "collab_repair", "POWERS_comment"
]

coder_cols = [f"c{n}_{col}" for n in ["1", "2"] for col in POWERS_cols]

client_only_cols = [col for col in coder_cols if col.endswith(
    ( "content_words", "num_nouns", "filled_pauses"))]

COMM_cols = [
    "communication", "topic", "subject", "dialogue", "conversation"
]

TT_DROP_COLS = [
    "file", "input_order", "shuffled_order", "position", "position_sub"
]

CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

GENERIC_TERMS = {"stuff", "thing", "things", "something", "anything", "everything", "nothing"}

UNINTELLIGIBLES = {"xx","xxx","yy","yyy"}

# count speech units after cleaning
def compute_speech_units(utt):
    cleaned = process_utterances(utt)
    tokens = cleaned.split()
    su = sum(tok.lower() not in UNINTELLIGIBLES for tok in tokens)
    return su

FILLER_PATTERN = re.compile(
    r"(?<!\w)(?:&-?)?(?:um+|uh+|erm+|er+|eh+)(?!\w)",
    re.IGNORECASE
)

# Count filled pauses Without cleaning
def count_fillers(utt: str) -> int:
    return len(FILLER_PATTERN.findall(utt))

# Expand contractions
def expand_contractions(utt: str) -> str:
    return contractions.fix(utt)

# Modified processing
def expand_and_process_utterances(utt: str) -> str:
    codeless_utt = " ".join([t for t in utt.split() if t and not t.startswith("&")])
    expanded_utt = expand_contractions(codeless_utt)
    modified_utt = expanded_utt.replace("-", "_")
    return process_utterances(modified_utt)

# --- NLP model singleton (your version, trimmed to essentials here) ---
class NLPmodel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._nlp_models = {}
            cls._instance.load_nlp()
        return cls._instance

    def load_nlp(self, model_name="en_core_web_trf"):
        if model_name not in self._nlp_models:
            self._nlp_models[model_name] = spacy.load(model_name)

    def get_nlp(self, model_name="en_core_web_trf"):
        if model_name not in self._nlp_models:
            self.load_nlp(model_name)
        return self._nlp_models[model_name]

# ---------- Rule helpers ----------
def is_generic(token) -> bool:
    return token.text.lower() in GENERIC_TERMS

def is_aux_or_modal(token) -> bool:
    """
    True for auxiliaries and modals you want to EXCLUDE.
    - SpaCy marks helping verbs as AUX (be/have/do/will/shall, etc.).
    - Modals have PTB tag 'MD'.
    """
    if token.pos_ != "AUX":
        return False
    # If it's AUX, always exclude for your rule set
    return True  # (covers modals + non-modal auxiliaries)

def is_unintelligble(token) -> bool:
    """Exclude unintelligible speech"""
    return token.text.lower() in UNINTELLIGIBLES

def is_ly_adverb(token) -> bool:
    # Only count adverbs that end with -ly
    return token.pos_ == "ADV" and token.text.lower().endswith("ly")

def is_numeral(token) -> bool:
    # Count numerals; SpaCy may set pos_==NUM, tag_==CD, and/or like_num==True
    return token.pos_ == "NUM" or token.tag_ == "CD" or token.like_num

def is_main_verb(token) -> bool:
    # Count ONLY main verbs (VERB); exclude AUX (handled separately)
    return token.pos_ == "VERB"

def is_noun_or_propn(token) -> bool:
    return token.pos_ in {"NOUN", "PROPN"}

def is_adjective(token) -> bool:
    return token.pos_ == "ADJ"

def is_chat_code(token) -> bool:
    return token.text.startswith("&")

def is_content_token(token) -> bool:
    """
    Master predicate implementing your rules:
    - Include: NOUN, PROPN, VERB (main only), ADJ, ADV(-ly only), NUM
    - Exclude: AUX (including modals), generic terms
    """
    if is_generic(token):
        return False
    if is_aux_or_modal(token):
        return False
    if is_unintelligble(token):
        return False
    if is_chat_code(token):
        return False

    if is_noun_or_propn(token):
        return True
    if is_main_verb(token):
        return True
    if is_adjective(token):
        return True
    if is_ly_adverb(token):
        return True
    if is_numeral(token):
        return True

    return False

def check_main_verb(tagged_utt: str, total_cw: int) -> tuple[str, int]:
    """
    Treat 'be' form as a main verb in the absence of any other VERB tag.

    Parameters
    ----------
    tagged_utt : str
        Utterance tagged with POS markers (e.g., "_VERB_CW").
    total_cw : int
        Current count of content words.

    Returns
    -------
    tuple[str, int]
        Possibly updated tagged utterance and content word count.
    """
    # Only apply if spaCy found no main verbs
    if "_VERB" not in tagged_utt:
        # Match standalone forms of 'be' (case-insensitive)
        m = re.search(r"\b(?:be|am|are|is|was|were|been|being)\b", tagged_utt, flags=re.IGNORECASE)
        if m:
            tagged_utt += "_BE_FORM_MAIN"
            total_cw += 1
    return tagged_utt, total_cw

# ---------- Core counting function ----------
def count_content_words_from_doc(doc):
    """
    Tally content words & nouns from a spaCy Doc object.
    Also tag tokens for manual review.
    """
    total_cw = total_nouns = 0
    tagged_utt = ""
    for tok in doc:
        tagged_utt += f"{tok}"
        if is_content_token(tok):
            total_cw += 1
            tagged_utt += f"_{tok.pos_}_CW"
            if tok.pos_ in ("NOUN", "PROPN"):
                total_nouns += 1
                tagged_utt += "_N"
        tagged_utt += " "
    tagged_utt, total_cw = check_main_verb(tagged_utt, total_cw)
    return total_cw, total_nouns, tagged_utt

def run_automation(df, coder_num):
    """
    Apply automated linguistic measures to a POWERS coding dataframe.

    Loads a spaCy transformer pipeline (en_core_web_trf) and applies:
      - Speech unit counts
      - Filled pause counts
      - Content word counts (all and nouns)

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least an "utterance" column.
    coder_num : str or int
        Coder identifier (e.g., "1", "2", "3"). Used to prefix new columns.

    Returns
    -------
    pandas.DataFrame
        Input dataframe with added columns:
        - c{coder_num}_speech_units
        - c{coder_num}_filled_pauses
        - c{coder_num}_content_words
        - c{coder_num}_num_nouns
    """

    try:
        NLP = NLPmodel()
        nlp = NLP.get_nlp("en_core_web_trf")
    except Exception as e:
        logger.error(f"Failed to load NLP model - automation not available: {e}")
        return df
    
    try:
        df[f"c{coder_num}_speech_units"] = df["utterance"].apply(compute_speech_units)
        df[f"c{coder_num}_filled_pauses"] = df["utterance"].apply(count_fillers)

        content_counts, noun_counts, tagged_utts = [], [], []
        utterances = df["utterance"].fillna("").map(expand_and_process_utterances)

        total_its = len(utterances)
        for doc in tqdm(nlp.pipe(utterances, batch_size=100, n_process=2),
                total=total_its, desc="Applying automation to utterances"):

            num_content_words, num_nouns, tagged_utt = count_content_words_from_doc(doc)
            content_counts.append(num_content_words)
            noun_counts.append(num_nouns)
            tagged_utts.append(tagged_utt)
        
        df[f"c{coder_num}_content_words"] = content_counts
        df[f"c{coder_num}_num_nouns"] = noun_counts
        
        utt_idx = df.columns.tolist().index("utterance")
        df.insert(utt_idx + 1, "tagged_utterance", tagged_utts)
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to apply automation: {e}")
        return df

def make_powers_coding_files(tiers, frac, coders, input_dir, output_dir, exclude_participants, automate_powers=True):
    """
    Generate POWERS coding and reliability files from utterance-level transcripts.

    Steps:
      1. Load transcript table files from input_dir/output_dir.
      2. Assign two coders per sample and shuffle sample order.
      3. Write a POWERS coding file with initialized coder columns.
      4. Select a fraction of samples for a reliability coder, producing a
         POWERS reliability coding file.
      5. Optionally run automated speech measures via run_automation.

    Parameters
    ----------
    tiers : dict
        Mapping of tier patterns to regex objects for file labeling.
    frac : float
        Proportion of samples to assign for reliability coding (0-1).
    coders : list of str
        List of coder IDs. If fewer than 3 provided, defaults to ['1','2','3'].
    input_dir : str or Path
        Directory containing input Utterances.xlsx files.
    output_dir : str or Path
        Base directory for powers_coding output.
    exclude_participants : list
        Speakers to exclude (filled with "NA").
    automate_powers : bool, optional
        If True, apply run_automation() to coder 1 columns.

    Returns
    -------
    None
        Writes Excel files to output_dir/powers_coding.
    """

    if len(coders) < 3:
        logger.warning(f"Coders entered: {coders} do not meet minimum of 3. Using default 1, 2, 3.")
        coders = ['1', '2', '3']

    output_dir = Path(output_dir)
    powers_coding_dir = output_dir / "powers_coding"
    logger.info(f"Writing POWERS coding files to {_rel(powers_coding_dir)}")

    # Collect utterance tables
    transcript_tables = find_files(directories=[input_dir, output_dir],
                                   search_base="transcript_tables")
    utt_dfs = [extract_transcript_data(tt) for tt in transcript_tables]

    for file, uttdf in tqdm(zip(transcript_tables, utt_dfs), desc="Generating POWERS coding files"):
        logger.info(f"Processing file: {_rel(file)}")
        labels = [t.match(file.name, return_None=True) for t in tiers.values()]
        labels = [l for l in labels if l is not None]

        assignments = assign_coders(coders)

        # Shuffle samples
        subdfs = []
        for _, subdf in uttdf.groupby(by="sample_id"):
            subdfs.append(subdf)
        random.shuffle(subdfs)
        shuffled_utt_df = pd.concat(subdfs, ignore_index=True)

        pc_df = shuffled_utt_df.drop(columns=[
            col for col in TT_DROP_COLS + \
                [t for t in tiers if t.lower() not in COMM_cols] if col in shuffled_utt_df.columns
            ]).copy()
        
        pc_df["c1_id"] = pd.Series(dtype="object")
        pc_df["c2_id"] = pd.Series(dtype="object")

        for col in coder_cols:
            if col in client_only_cols:
                pc_df[col] = np.where(pc_df["speaker"].isin(exclude_participants), "NA", "")
            else:
                pc_df[col] = ""
        
        if automate_powers:
            pc_df = run_automation(pc_df, "1")

        unique_sample_ids = list(pc_df['sample_id'].drop_duplicates(keep='first'))
        segments = segment(unique_sample_ids, n=len(coders))
        rel_subsets = []

        for seg, ass in zip(segments, assignments):
            pc_df.loc[pc_df['sample_id'].isin(seg), 'c1_id'] = ass[0]
            pc_df.loc[pc_df['sample_id'].isin(seg), 'c2_id'] = ass[1]

            rel_samples = random.sample(seg, k=max(1, round(len(seg) * frac)))
            relsegdf = pc_df[pc_df['sample_id'].isin(rel_samples)].copy()

            rel_subsets.append(relsegdf)

        reldf = pd.concat(rel_subsets)

        rel_drop_cols = [col for col in coder_cols if col.startswith("c2")]
        reldf.drop(columns=rel_drop_cols, inplace=True, errors='ignore')
        
        rename_map = {col:col.replace("1", "3") for col in coder_cols if col.startswith("c1")}
        reldf.rename(columns=rename_map, inplace=True)
        
        logger.info(f"Selected {len(set(reldf['sample_id']))} samples for reliability from {len(set(pc_df['sample_id']))} total samples.")

        lab_str = '_'.join(labels) + '_' if labels else ''

        pc_filename = Path(powers_coding_dir, *labels, f"{lab_str}powers_coding.xlsx")
        rel_filename = Path(powers_coding_dir, *labels, f"{lab_str}powers_reliability_coding.xlsx")

        try:
            pc_filename.parent.mkdir(parents=True, exist_ok=True)
            pc_df.to_excel(pc_filename, index=False, na_rep="")
            logger.info(f"Successfully wrote POWERS coding file: {_rel(pc_filename)}")
        except Exception as e:
            logger.error(f"Failed to write POWERS coding file {_rel(pc_filename)}: {e}")

        try:
            rel_filename.parent.mkdir(parents=True, exist_ok=True)
            reldf.to_excel(rel_filename, index=False, na_rep="")
            logger.info(f"Successfully wrote POWERS reliability coding file: {_rel(rel_filename)}")
        except Exception as e:
            logger.error(f"Failed to write POWERS reliability coding file {_rel(rel_filename)}: {e}")


def reselect_powers_reliability(input_dir, output_dir, frac, exclude_participants, automate_powers):
    """
    Reselect new reliability subsets from existing POWERS coding files.

    Finds powers_coding and powers_reliability_coding files, determines
    which samples are already covered by reliability coders, and selects
    new samples from the remaining pool. Optionally applies automation.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing original powers_coding and powers_reliability_coding files.
    output_dir : str or Path
        Directory for saving powers_reselected_reliability outputs.
    frac : float
        Fraction of samples per file to assign to reliability (0-1).
    exclude_participants : list
        Speakers to exclude (filled with "NA").
    automate_powers : bool
        If True, apply run_automation() to coder 3 columns.

    Returns
    -------
    None
        Writes new Excel reliability files to output_dir/powers_reselected_reliability.
    """

    output_dir = Path(output_dir)
    
    powers_reselected_reliability_dir = output_dir / "powers_reselected_reliability"
    try:
        powers_reselected_reliability_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {_rel(powers_reselected_reliability_dir)}")
    except Exception as e:
        logger.error(f"Failed to create directory {_rel(powers_reselected_reliability_dir)}: {e}")
        return

    coding_files = [f for f in Path(input_dir).rglob('*powers_coding*.xlsx')]
    rel_files = [f for f in Path(input_dir).rglob('*powers_reliability_coding*.xlsx')]

    # Match original coding and reliability files.
    for cod in tqdm(coding_files, desc="Reselecting POWERS reliability coding..."):
        try:
            covered_sample_ids = set()
            powers_coding_df = pd.read_excel(cod)
            logger.info(f"Processing coding file: {_rel(cod)}")
        except Exception as e:
            logger.error(f"Failed to read file {_rel(cod)}: {e}")
            continue
        for rel in rel_files:
            if cod.name.replace("powers_coding", "powers_reliability_coding") == rel.name:
                try:
                    powers_rel_df = pd.read_excel(rel)
                    logger.info(f"Processing reliability file: {_rel(rel)}")
                except Exception as e:
                    logger.error(f"Failed to read file {_rel(rel)}: {e}")
                    continue
                
            covered_sample_ids.update(set(powers_rel_df["sample_id"].dropna()))
        
        if covered_sample_ids:
            all_samples = set(powers_coding_df["sample_id"].dropna())
            available_samples = list(all_samples - covered_sample_ids)

            if len(available_samples) == 0:
                logger.warning(f"No available samples to reselect for {cod.name}. Skipping.")
                continue
            
            num_to_select = max(1, round(len(all_samples) * float(frac)))
            if len(available_samples) < num_to_select:
                logger.warning(
                    f"Not enough unused samples in {cod.name}. "
                    f"Selecting {len(available_samples)} instead of target {num_to_select}."
                )
                num_to_select = len(available_samples)
            
            reselected_rel_samples = set(random.sample(available_samples, k=num_to_select))
            new_rel_df = powers_coding_df[powers_coding_df['sample_id'].isin(reselected_rel_samples)].copy()

            for col in coder_cols:
                new_rel_df[col] = np.where(new_rel_df["speaker"].isin(exclude_participants), "NA", "")

            rel_drop_cols = [col for col in coder_cols if col.startswith("c2")]
            new_rel_df.drop(columns=rel_drop_cols, inplace=True, errors='ignore')
            
            rename_map = {col:col.replace("1", "3") for col in coder_cols if col.startswith("c1")}
            new_rel_df.rename(columns=rename_map, inplace=True)
            
            logger.info(f"Reselected {len(set(new_rel_df['sample_id']))} samples for reliability from {len(set(powers_coding_df['sample_id']))} total samples.")

            if automate_powers:
                new_rel_df = run_automation(new_rel_df, "3")

            try:
                new_rel_filename = cod.name.replace("powers_coding", "reselected_powers_reliability_coding")
                new_rel_filepath = powers_reselected_reliability_dir / new_rel_filename
                new_rel_df.to_excel(new_rel_filepath, index=False)
                logger.info(f"Successfully wrote reselected POWERS reliability coding file: {_rel(new_rel_filepath)}")
            except Exception as e:
                logger.error(f"Failed to write reselected POWERS reliability coding file {_rel(new_rel_filepath)}: {e}")
