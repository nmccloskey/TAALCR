#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
from taalcr.utils.auxiliary import parse_stratify_fields, build_arg_parser
from taalcr.run_wrappers import (
    run_analyze_digital_convo_turns,
    run_make_powers_coding_files,
    run_analyze_powers_coding,
    run_evaluate_powers_reliability,
    run_reselect_powers_reliability_coding
)
from taalcr.powers.automation_validation import select_validation_samples, validate_automation
from rascal.utils.auxiliary import load_config, project_path, find_files
from rascal.run_wrappers import run_read_tiers, run_read_cha_files, run_make_transcript_tables
from rascal.utils.logger import (
    get_root,
    set_root,
    logger,
    initialize_logger,
    terminate_logger,
)


def main(args):
    """Process input arguments and execute appropriate TAAALCR operations."""
    try:
        start_time = datetime.now()
        set_root(Path.cwd())

        # -----------------------------------------------------------------
        # Configuration and directories
        # -----------------------------------------------------------------
        config_path = project_path(args.config or "config.yaml")
        config = load_config(config_path)
    
        input_dir = project_path(config.get("input_dir", "taaalcr_data/input"))
        if not input_dir.is_relative_to(get_root()):
            logger.warning(f"Input directory {input_dir} is outside the project root.")
        output_dir = project_path(config.get("output_dir", "taaalcr_data/output"))

        timestamp = start_time.strftime("%y%m%d_%H%M")
        out_dir = (output_dir / f"taaalcr_output_{timestamp}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------------------
        # Initialize logger once output folder is ready
        # -----------------------------------------------------------------
        initialize_logger(start_time, out_dir, "TAAALCR")
        logger.info("Logger initialized and early logs flushed.")

        frac = config.get('reliability_fraction', 0.2)
        coders = config.get('coders', []) or []
        exclude_participants = config.get('exclude_participants', []) or []
        automate_powers = config.get('automate_powers', True)
        just_c2_powers = config.get('just_c2_powers', False)

        tiers = run_read_tiers(config.get('tiers', {})) or {}

        # ---------------------------------------------------------
        # Dispatch
        # ---------------------------------------------------------
        if args.command == "turns":
            run_analyze_digital_convo_turns(input_dir, out_dir)

        elif args.command == "powers":
            if args.action == "make":
                transcript_tables = find_files(directories=[input_dir, out_dir],
                                               search_base="transcript_tables")
                if not transcript_tables:
                    chats = run_read_cha_files(input_dir)
                    run_make_transcript_tables(tiers, chats, out_dir)
                run_make_powers_coding_files(
                    tiers, frac, coders, input_dir, out_dir, exclude_participants, automate_powers
                )

            elif args.action == "analyze":
                run_analyze_powers_coding(input_dir, out_dir, just_c2_powers)

            elif args.action == "evaluate":
                run_evaluate_powers_reliability(input_dir, out_dir)

            elif args.action == "reselect":
                run_reselect_powers_reliability_coding(
                    input_dir, out_dir, frac, exclude_participants, automate_powers
                )

            elif args.action == "select":
                stratify_fields = parse_stratify_fields(args.stratify)
                select_validation_samples(
                    input_dir=input_dir,
                    output_dir=out_dir,
                    stratify=stratify_fields,
                    strata=args.strata,
                    seed=args.seed
                )

            elif args.action == "validate":
                selection_table = args.selection if args.selection else None
                stratum_numbers = parse_stratify_fields(args.numbers)
                validate_automation(
                    input_dir=input_dir,
                    output_dir=out_dir,
                    selection_table=selection_table,
                    stratum_numbers=stratum_numbers
                )
                run_analyze_powers_coding(input_dir, out_dir, exclude_participants=exclude_participants)

            else:
                logger.error(f"Unknown powers action: {args.action}")
        else:
            logger.error(f"Unknown command: {args.command}")

    except Exception as e:
        logger.error(f"TAAALCR execution failed: {e}", exc_info=True)
        raise
    
    finally:
        # Always finalize logging and metadata
        terminate_logger(input_dir, out_dir, config_path, config, start_time, "TAAALCR")

# -------------------------------------------------------------
# Direct execution
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
