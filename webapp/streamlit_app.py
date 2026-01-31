"""
TAALCR Streamlit App
-------------------
Web interface for the Toolkit for Aggregate Analysis of Language in Conversation, for Research (TAALCR).
Allows users to upload configuration and conversation data, select analysis modules,
and download structured outputs.
"""

from pathlib import Path
import streamlit as st
import yaml
import tempfile
import zipfile
from io import BytesIO
from datetime import datetime

# --- Ensure src path is importable for local dev installs ---
def add_src_to_sys_path():
    import sys
    src_path = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_path))

add_src_to_sys_path()

# --- TAALCR & RASCAL imports ---
from taalcr.main import (
    run_analyze_powers_coding,
    run_make_powers_coding_files,
    run_analyze_digital_convo_turns,
)
from taalcr.run_wrappers import (
    run_reselect_powers_reliability_coding,
    run_evaluate_powers_reliability,
)
from rascal.run_wrappers import run_read_tiers, run_read_cha_files, run_make_transcript_tables


# --- Streamlit UI ---
st.title("TAALCR Web App")
st.subheader("Toolkit for Aggregate Analysis of Language in Conversation, for Research")

if "confirmed_config" not in st.session_state:
    st.session_state.confirmed_config = False


# --- Utility: ZIP entire folder ---
def zip_folder(folder_path: Path) -> BytesIO:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(folder_path))
    buffer.seek(0)
    return buffer


# --- Step 1: Upload or build config ---
st.header("1️⃣ Configuration")

config_file = st.file_uploader("Upload your config.yaml", type=["yaml", "yml"])
config = None

if config_file:
    config = yaml.safe_load(config_file)
    st.success("✅ Config file loaded.")
else:
    st.info("No config uploaded yet. Please upload a YAML configuration file.")

# --- Step 2: Upload input files ---
st.header("2️⃣ Upload Input Files")
uploaded_files = st.file_uploader("Upload CHA or XLSX input files", type=["cha", "xlsx"], accept_multiple_files=True)

if config and uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output"
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Save uploads
        for f in uploaded_files:
            (input_dir / f.name).write_bytes(f.read())

        # Load configuration
        tiers = run_read_tiers(config.get("tiers", {})) or {}
        frac = config.get("reliability_fraction", 0.2)
        coders = config.get("coders", []) or []
        exclude_participants = config.get("exclude_participants", []) or []
        automate_powers = config.get("automate_powers", True)
        just_c2_powers = config.get("just_c2_powers", False)

        # --- Step 3: Select analysis type ---
        st.header("3️⃣ Select Function")
        options = [
            "Analyze Digital Conversation Turns",
            "Make POWERS Coding Files",
            "Analyze POWERS Coding",
            "Evaluate POWERS Reliability",
            "Reselect POWERS Reliability Coding",
        ]
        choice = st.selectbox("Select a process:", options)

        if st.button("🚀 Run Analysis"):
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            out_dir = output_dir / f"taalcr_output_{timestamp}"
            out_dir.mkdir(parents=True, exist_ok=True)

            if "Turns" in choice:
                run_analyze_digital_convo_turns(input_dir, out_dir)

            elif "Make POWERS" in choice:
                chats = run_read_cha_files(input_dir)
                run_make_transcript_tables(tiers, chats, output_dir)
                run_make_powers_coding_files(
                    tiers, frac, coders, input_dir, output_dir, exclude_participants, automate_powers
                )

            elif "Analyze POWERS" in choice:
                run_analyze_powers_coding(input_dir, out_dir, just_c2_powers)

            elif "Evaluate POWERS" in choice:
                run_evaluate_powers_reliability(input_dir, out_dir)

            elif "Reselect POWERS" in choice:
                run_reselect_powers_reliability_coding(input_dir, out_dir, frac, exclude_participants, automate_powers)

            st.success("✅ Analysis complete!")
            zip_data = zip_folder(out_dir)
            st.download_button(
                label="Download Results ZIP",
                data=zip_data,
                file_name=f"{choice.replace(' ', '_').lower()}_{timestamp}.zip",
                mime="application/zip",
            )


# --- Optional CLI launcher ---
def main():
    """Allow launching this Streamlit app from CLI."""
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])

if __name__ == "__main__":
    main()
