# TAALCR — Toolkit for Aggregate Analysis of Language in Conversation, for Research

TAALCR is a research toolkit for batched dialog analysis that includes workflows for analyzing digital conversation turns and [POWERS](https://doi.org/10.3233/ACS-2013-20107) coding. It complements (and imports) the monologic discourse analysis system [RASCAL](https://github.com/nmccloskey/RASCAL). A third functionality for characterizing clinical language elicitation is underway.

---

## Overview (more details below)

- **Digital Conversation Turns Analysis**
   - Tracking turn-taking in dialogs can reveal meaningful linguistic and psychosocial patterns ([Tuomenoksa, et al., 2020](https://doi.org/10.1080/02687038.2020.1852518)).
   - Recording turns with a sequence of digits enables analysis of both tallies and transition probabilities (see below).
- **POWERS Coding**
   - Profile of Word Errors and Retrieval in Speech (POWERS) is an aphasiological coding system for analyzing dialogic speech (Herbet, et al., 2013).
   - TAALCR POWERS pipeline:
      - generates coder workbooks, automating most fields
      - summarizes coding and reports ICC2 values between coders
      - evaluates and optionally reselects reliability coding
   - Automation validation (CLI only)
      - select (stratified) random subset for manual coding
      - evaluate reliability between automatic & manual codes
---

## Web App

You can use TAALCR in your browser — no installation required:

👉 [Launch the TAALCR Web App](https://taalcr.streamlit.app/)

---

## Installation

A dedicated virtual environment using Anaconda is recommended:

### 1. Create and activate your environment:

```bash
conda create --name taalcr python=3.12
conda activate taalcr
```

### 2. Download TAALCR:
```bash
# directly from PyPI
pip install taalcr

# or from GitHub
pip install git+https://github.com/nmccloskey/taalcr.git@main
```

### 3. Install the `en_core_web_trf` model (for POWERS coding automation):
```bash
python -m spacy download en_core_web_trf
```

---

## Setup

To prepare for running TAALCR, complete the following steps:

### 1. Create your working directory:

Example structure:
```
your_project/
├── config.yaml           # Configuration file (see below)
└── taalcr_data/
    └── input/            # Place your .cha or .xlsx files here
                          # (TAALCR will make an output directory)
```

### 2. Provide a `config.yaml` file

This file specifies the directories, coders, settings, and tier structure.

You can download the example config file from the repo or create your own like this:

```yaml
input_dir: taalcr_data/input
output_dir: taalcr_data/output
reliability_fraction: 0.2
automate_POWERS: true
just_c2_POWERS: false
exclude_participants:
coders:
- '1'
- '2'
- '3'
tiers:
  time:
    values:
    - PreTx
    - PostTx
  client_id:
    values: \d+
  setting:
    values:
    - LargeGroup
    - SmallGroup
```
### Explanation:

- **General**
  - `reliability_fraction` – proportion of data to subset for reliability (default 20%).
  - `coders` – alphanumeric coder identifiers (3 required for function `powers make`).
  - `exclude_participants` – speakers appearing in .cha files to exclude from POWERS coding files.
  - `automate_POWERS` – toggle automated preparation of POWERS coding spreadsheets (coder 1 fields).
  - `just_c2_POWERS` – whether to use only coder 2 columns in analysis outputs.

- **Tiers**
  - Define metadata fields extracted from filenames (`time`, `client_id`, `setting`).
  - Each tier has attributes:
    - `values` – acceptable set of identifiers or regex patterns.
    - `partition` – (True/False) creates separate coding and reliability files for that tier.

   > See [RASCAL](https://github.com/nmccloskey/RASCAL) for more information about the **tier** system for organizing data based on .cha file names.

---

## Quickstart — Command Line

TAALCR exposes a concise CLI with subcommands:

```bash
# Analyze digital conversation turns
taalcr turns

# POWERS workflow
taalcr powers make       # prepare POWERS coding files
taalcr powers analyze    # analyze completed POWERS coding
taalcr powers evaluate   # evaluate completed POWERS reliability coding
taalcr powers reselect   # randomly reselect reliability subset

# Automation validation
taalcr powers select     # randomly select subset for validating automation
taalcr powers validate   # compute reliability metrics on automated vs manual codes 
```
---
# Digital Conversation Turns (DCT) Protocol

TAALCR includes a lightweight system for analyzing **digital conversational turns** in group treatment sessions for people with aphasia.  
Instead of simple tallies, the DCT protocol records the **sequence of turns** compactly, enabling analysis of turn-taking dynamics and engagement, with optional markers for capturing turn qualities (e.g., length/substantiveness).

---

## Coding Procedure

### 1. Speaker Assignment
- `0` = Clinician(s) (all individuals not receiving treatment collapsed under this code)
- `1` = Participant 1
- `2` = Participant 2
- Continue incrementing (`3`, `4`, …) as needed.

### 2. Turn Entry with Markers
For each conversational turn, enter the assigned digit for the speaker (e.g., `0`, `1`, `2`).

Marking system:
- Digits are followed by one dot `.` (mark1), two dots `..` (mark2) or no dots
- Example usage:
   - Add `.` if the turn is *substantial* (contains an independent clause).
   - Add `..` if the turn is *monologic* (contains at least two independent clauses)
   - Add no dots otherwise, or the turn is *minimal* (brief/no full idea)

### 3. Input Coding Table Format
- Turns are entered sequentially as a continuous string of digits and dots. 
- Bins are recommended for some temporal granularity (e.g., six 10-minute bins for a 1-hour conversation/treatment session).
- Case-insensitive file name regex `r'.*(Convo|Conversation)_?Turns.*\.xlsx$'` looks for files like `*TU_ConvoTurns.xlsx` or `*converstation_turns_2025.xlsx`

### Example: Digital Conversation Turns Coding Input

| site | session | group   | coder | bin | turns |
|------|---------|---------|-------|-----|-------|
| TU   | 12      | Dyad1   | NM    | 1   | `212012.02121210.10101.210.12.021212121210.210.2.1.010121.010.110.2102.12.` |
| TU   | 12      | Dyad1   | NM    | 2   | `0202.121212101.011101.2.12.120201.212101020202.10.21212.02.12010212.` |
| TU   | 12      | Dyad1   | NM    | 3   | `12..121.212.1212.0202.12120.201.210101..2012121.2121.2..1212.12.020.2.0` |
| TU   | 12      | Dyad1   | NM    | 4   | `010202.02121021020212101.01012101210010102.1210101010101010101010121020.1.` |
| TU   | 12      | Dyad1   | NM    | 5   | `0.121210.1010102120.102.02120212.0.2.020212121202121212.120.21010101212121` |
| TU   | 12      | Dyad1   | NM    | 6   | `2120210101212121212.10121202.12.02.1212010202.02.02.0202.020201202020.22.02012102002.012102` |
| TU   | 4       | LgGroup | NM    | 1   | `4.24.242424.0640.4.206.434343430606.060436.3706.0406.76760.602.502.326207.07.67.06767.3737.17.0701270606.06.54321007` |
| TU   | 4       | LgGroup | NM    | 2   | `763670.50505620507102..02404676.70101...010.707057574767.6..76717.01.7010141.4..1014.3401.671..61016161.721.77414.0` |
| TU   | 4       | LgGroup | NM    | 3   | `2.0.2.0.3.13.23.01313535737037.0.7.137314.` |
| TU   | 4       | LgGroup | NM    | 4   | `4.0.5.35.05.0.5..7575404.53436..40575754..24242..575.4375.45705.20.6.` |
| TU   | 4       | LgGroup | NM    | 5   | `06.007070767676050.21627.17.106063434607571270101.61.01016.161.2.0.1.01` |
| TU   | 4       | LgGroup | NM    | 6   | `0.607.2707.07.06..06.06.4603403212607201202..2702760276..020.1212606016..70.701702.1.70731313510.` |

---

## Output
The `taalcr turns` command analyzes coded conversation turn files  and produces an Excel workbook with multiple sheets, capturing turn-taking behavior at **bin**, **speaker**, **session**, and **group** levels, also including **transition matrices** for a detailed view of conversational dynamics.

| Excel Sheet              | Level of Analysis | Data Included                                                                 
|---------------------------|------------------|-------------------------------------------------------------------------------|
| **Speaker_Level_Turns**   | Speaker          | Total turns, dot-mark counts (mark1/mark2), proportions                      |
| **Group_Level_Summary**   | Group            | Group totals, num participants, num sessions, marker proportions             |
| **Session_Level_Summary** | Session × Group  | Totals, entropy, clinician–participant ratio, marker proportions             |
| **Participation_Level_Turns** | Speaker × Session | Individual totals, session proportion, marker rates, bin variability stats |
| **Bin_Level_Turns**       | Speaker × Bin    | Proportion of bin turns, marker proportions within bins                      |
| *Speaker_Matrix_* *      | Group            | Conditional probabilities of turn transitions (matrix)                       |
| **Speaker_Level_Ratios**  | Group            | Participant→Participant, Participant→Clinician, Clinician→Participant ratios |
| **Summary_Statistics**    | Aggregated       | Mean, std, min, max, CV for all numeric metrics                              |

---

## Analytic Opportunities

- **Turn counts & proportions** per participant  
- **Substantial vs. monologic vs. minimal** turn ratios  
- **Transitions** (e.g., clinician → participant, participant → participant)  
- **Speaker dominance indices**  
- **Engagement rates** between participants  
- **Transition matrices & dyadic graphs**  
- **Temporal trends** (with optional bins)  
- **Reliability**: inter-coder sequence comparisons (e.g., Levenshtein distance)  
- **Correlation with treatment outcomes** (e.g., ACOM, WAB) for longitudinal studies  
- **Turn quality** (marker proportions for repairs/overlaps)  
- **Consistency over time** (bin-level variability)  
- **Interaction structure** (flow directionality between speakers)  
- **Individual engagement** (relative contributions across sessions)  
- **Balance of participation/distribution metrics** (e.g., Gini index, entropy, clinician–participant ratios)  

---

## Limitations
- **Turn Overlap**: current system assumes sequentialization - not uncommonly violated in group settings.  
- **Subjectivity**: coder judgment needed for speaker boundaries and substantiality. Calibration recommended.  
- **Binary turn length**: `mark1` vs. `mark2` is coarse; future versions may refine scale.  
- **Scalability**: currently designed for up to 9 participants, future work could accommodate codes like `C`,`P10`, `P11`.
 
---

# Profile of Word Errors and Retrieval in Speech (POWERS) coding

## Measures

The POWERS coding system addresses the need to assess language abilities (particularly lexical retrieval) in conversation for people with aphasia. TAALCR facilitates quantification of the following subset of POWERS variables for both the clinician and client (see the [POWERS](https://doi.org/10.3233/ACS-2013-20107) manual for full details): 

   - **filled pauses** - disfluencies like "um", "uh", "er", etc.
   - **speech units** - these more or less map onto non-punctuation tokens excluding filled pauses
   - **content words** - nouns (including proper nouns), non-auxiliary verbs, adjectives, *-ly*-terminal adverbs, and numerals
   - **nouns** - a subset of content words
   - **number of turns** - a verbal contribution to the conversation with three types:
      - *substantial turn* - contains at least one content word
      - *minimal turn* - hands the turn back to the other conversation partner
      - *subminimal turn (a nonce, non-canonical term)* - not classifiable as either type above
   - **collaborative repair** - sequences of turns devoted to overcoming communicative error/difficulty

## Automation (reliability details pending)

TAALCR automates as much as possible. Below are descriptions of automatability and ICC2 utterance-level reliability metrics on a stratified (by study site, mild/severe aphasia profile, and pre-/post-tx test) random selection of XX samples (XX utterances).
   - **fully automated** with regex and spaCy (`en_core_web_trf`):
      - *filled pauses:*
      - *speech units:*
      - *content words:*
      - *noun count:*
   - **semi-automated** with a computational first pass followed by manual checks:
      - *turn type:*
   - **fully manual** given the rich contextual dependencies:
      - *collaborative repair*

## Typical Workflow

1. **Tabularize utterances (if needed)**  
   If `*Utterances*.xlsx` files aren’t present, TAALCR will call RASCAL to read `.cha` files and tabularize utterances, assigning samples unique identifiers at the utterance and transcript levels.

2. **Prepare POWERS coding files**  
   `taalcr powers make` creates full dataset plus reliability coding workbooks, with most coding automated.

3. **Human coding**  
   Coders complete POWERS annotations in the generated spreadsheets.

4. **Analyze**  
   `taalcr powers analyze` aggregates and reports POWERS metrics at the turn, speaker, and dialog levels.

5. **Reliability evaluation**  
   `taalcr powers evaulate` matches reliability files and runs ICC2 evaluation.

6. **Reliability subset (optional)**  
   `taalcr powers reselect` eselects reliability coding subset if ICC2 measures fail to meet threshold (0.7 a typical minimum).

### Pipeline Commands

| Command | Function (name)                          | Input                                 | Output                                              |
|---------|------------------------------------------|---------------------------------------|-----------------------------------------------------|
| powers make     | Prepare POWERS coding files (*make_POWERS_coding_files*) | Either `.cha` files or utterance tables generated with RASCAL    | POWERS coding spreadsheets for coders               |
| powers analyze  | Analyze POWERS coding (*analyze_POWERS_coding*) | Completed POWERS spreadsheets         | Turn-, speaker-, and dialog-level aggregates        |
| powers evaluate | Evaluate POWERS reliability (*match_reliability_files*, *analyze_POWERS_coding*) | Coder 2 + Coder 3 spreadsheets        | Reliability metrics (ICC2, kappa, etc.)        |
| powers reselect | Reselect POWERS reliability (*reselect_POWERS_reliability*) | Original + reliability spreadsheets   | New reliability subset(s) for reassignment          |

---

## Automation Validation

TAALCR includes CLI utilities to validate automatic POWERS coding against manual coding.

This workflow has two main steps:

### 1. Select Validation Samples
Use (stratified) random sampling to create a balanced subset of samples for manual validation.

**Arguments:**

- `--stratify`: Optional fields to group by (comma, space, or repeated flags) in random sample selection.
   
   Example: `--stratify site,test` or `--stratify site --stratify test`.

- `--strata`: Number of samples to draw per stratum (default: 5).

- `--seed`: Random number generator seed for reproducibility (default: 42).

**Output:**

- An Excel file `POWERS_validation_selection_<timestamp>.xlsx` containing the selected samples.

- The `stratum_no` column facilitates "chunking" the reliability subset. For example:

   - Code through stratum numbers 1 & 2
   - Evaluate reliability
   - Work through further strata if agreement is poor

- If POWERS coding tables exist in the input folder, labeled versions with `stratum_no` will also be written.


```bash
# Example
taalcr powers select \
  --stratify site,test \
  --strata 5 \
  --seed 42
```

### 2. Validate Automation

Merge the automatic and manual coding files for side-by-side comparison and reliability checks.

**Requirements:**

- Place your coding files in two subdirectories under the input folder:

   - `Auto/` containing automatically generated coding files

   - `Manual/` containing manually coded files

**Arguments:**

- `--selection`: Path to the selection Excel file from the previous step. Required if `stratum_no` is not already in the Manual coding files.

- `--numbers`: Optional comma- or space-separated list of stratum numbers to include (e.g., `--numbers 1,2`).

**Output:**

- An Excel file POWERS_Coding_Auto_vs_Manual.xlsx inside a new AutomationValidation/ folder.
This file contains paired automatic and manual codes, restricted to the requested strata if specified.

```bash
# Example
taalcr powers validate \
  --selection taalcr_powers_select_output_250930/POWERS_validation_selection_250930_1530.xlsx \
  --numbers 1,2
```

**Typical Workflow**

1. Run `powers select` to generate a stratified subset of samples.

2. Manually code samples marked with `stratum_no`.

3. After manual coding, run `powers validate` to merge auto vs manual annotations.

4. Use the merged file to compute inter-coder reliability or other evaluation metrics.

---

## 🧪 Testing

This project uses [pytest](https://docs.pytest.org/) for its testing suite.  
All tests are located under the `tests/` directory, organized by module/function.

### Running Tests
To run the full suite:

```bash
pytest
```
Run "quietly":
```bash
pytest -q
```
Run a specific test file:
```bash
pytest tests/test_samples/test_digital_convo_turns_analyzer.py
```
---

## Status and Contact

I warmly welcome feedback, feature suggestions, or bug reports. Feel free to reach out by:

- Submitting an issue through the GitHub Issues tab

- Emailing me directly at: nsm [at] temple.edu

Thanks for your interest and collaboration!

*(This project was previously developed under the working name “DIAAD”; the rename reflects a clarified scope and focus.)*

---

## Citation & Acknowledgments

Full details of the POWERS coding system can be found in the manual:

> Herbert, R., Best, W., Hickin, J., Howard, D., & Osborne, F. (2013). Powers: Profile of word errors and retrieval in speech: An assessment tool for use with people with communication impairment. CQUniversity.

If TAALCR supports your work, please cite the repo:

> McCloskey N. (2025). TAALCR: Toolkit for Aggregate Analysis of Language in Conversation, for Research. GitHub repository. https://github.com/nmccloskey/taalcr

---
