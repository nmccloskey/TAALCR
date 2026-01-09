def run_analyze_digital_convo_turns(input_dir, output_dir):
    from taaalcr.convo_turns.digital_convo_turns_analysis import analyze_digital_convo_turns
    analyze_digital_convo_turns(input_dir=input_dir, output_dir=output_dir)

def run_make_powers_coding_files(tiers, frac, coders, input_dir, output_dir, exclude_participants, automate_powers=True):
    from taaalcr.powers.powers_coding_files import make_powers_coding_files
    make_powers_coding_files(
        tiers=tiers,
        frac=frac,
        coders=coders,
        input_dir=input_dir,
        output_dir=output_dir,
        exclude_participants=exclude_participants,
        automate_powers=automate_powers
    )

def run_analyze_powers_coding(input_dir, output_dir, reliability=False, just_c2_powers=False, exclude_participants=[]):
    from taaalcr.powers.powers_coding_analysis import analyze_powers_coding
    analyze_powers_coding(
        input_dir=input_dir,
        output_dir=output_dir,
        reliability=reliability,
        just_c2_powers=just_c2_powers, exclude_participants=exclude_participants)

def run_evaluate_powers_reliability(input_dir, output_dir):
    from taaalcr.powers.powers_coding_analysis import match_reliability_files, analyze_powers_coding
    match_reliability_files(input_dir=input_dir, output_dir=output_dir)
    analyze_powers_coding(input_dir=input_dir, output_dir=output_dir, reliability=True, just_c2_powers=False)

def run_reselect_powers_reliability_coding(input_dir, output_dir, frac, exclude_participants, automate_powers):
    from taaalcr.powers.powers_coding_files import reselect_powers_reliability
    reselect_powers_reliability(
        input_dir=input_dir,
        output_dir=output_dir,
        frac=frac,
        exclude_participants=exclude_participants,
        automate_powers=automate_powers)
