import re
import streamlit as st
import yaml
from io import BytesIO

def _split_values(text: str):
    """
    Split user input into values. Supports comma or newline separators.
    Trims whitespace and ignores empty entries.
    """
    if not text:
        return []
    parts = re.split(r'[,\n]', text)
    return [p.strip() for p in parts if p.strip()]

def build_config_ui():
    st.subheader("🔧 Create TAAALCR Config")

    # ---- Top-level defaults for public release ----
    input_dir = st.text_input("Input directory", value="data/input")
    output_dir = st.text_input("Output directory", value="data/output")
    reliability_fraction = st.number_input(
        "Reliability fraction", min_value=0.0, max_value=1.0, value=0.2
    )

    coders = st.text_input("Coders (comma-separated)", value="1, 2, 3")
    coder_list = [c.strip() for c in coders.split(",") if c.strip()]

    exclude_participants_str = st.text_input(
        "Exclude participants (comma-separated)", value="INV"
    )
    exclude_participants_list = [
        ep.strip() for ep in exclude_participants_str.split(",") if ep.strip()
    ]

    # --- Boolean toggles (public defaults) ---
    st.subheader("⚙️ Processing Options")
    automate_POWERS = st.checkbox("Automate POWERS coding", value=True)

    # --- Tier Instructions ---
    with st.expander("📘 Tier entry instructions", expanded=False):
        st.markdown(
            """
**How to enter tier values**

- **Multiple values**: enter as a comma- or newline-separated list of **literal** options.
  - Example – *setting*:
    ```
    LargeGroup, SmallGroup
    ```
  These are treated as **literal choices** and combined into a regex internally.

- **Single value**: treated as a **regular expression** and validated immediately.
  - Examples:
    - Digits only: `\\d+`
    - Lab site + digits: `(AC|BU|TU)\\d+`
    - Three uppercase letters + three digits: `[A-Z]{3}\\d{3}`

- **Tier attribute**
  - **Partition**: creates separate coding files and **separate reliability** calculations by that tier (e.g., by test).
            """
        )
        st.caption("Tip: If your regex contains commas, paste it as a single value (no commas or newlines).")

    # --- Tier Builder (public defaults) ---
    st.subheader("📐 Tier Definitions")
    if "tiers" not in st.session_state:
        # Minimal public-facing defaults — users can add/modify as needed
        st.session_state.tiers = [
            {"label": "participant_id", "values": r"\d+", "is_partition": False},
            {"label": "conversation", "values": "LargeGroup, SmallGroup", "is_partition": False},
        ]

    # Interactive tier editor
    regex_errors = []
    for i, tier in enumerate(st.session_state.tiers):
        cols = st.columns([2, 4, 1, 1])
        tier["label"] = cols[0].text_input(f"Name of Tier {i+1}", value=tier["label"], key=f"tier_label_{i}")
        tier["values"] = cols[1].text_area(
            f"Values (comma/newline-separated literals OR single regex) – {tier['label']}",
            value=tier["values"], key=f"tier_values_{i}",
            help="If exactly one entry is provided, it will be validated as a regex."
        )
        tier["is_partition"] = cols[2].checkbox("Partition", value=tier["is_partition"], key=f"tier_partition_{i}")

        # Live validation: if single value, treat as regex and compile
        vals = _split_values(tier["values"])
        if len(vals) == 1 and vals[0]:
            try:
                re.compile(vals[0])
            except re.error as e:
                msg = f"Tier '{tier['label']}': invalid regex: {e}"
                regex_errors.append(msg)
                st.error(msg)

    col1, col2 = st.columns([1, 1])
    if col1.button("➕ Add Tier"):
        st.session_state.tiers.append({"label": "", "values": "", "is_partition": False})
    if col2.button("➖ Remove Last Tier"):
        if st.session_state.tiers:
            st.session_state.tiers.pop()

    # --- YAML Assembly ---
    tiers_dict = {}
    for tier in st.session_state.tiers:
        name = (tier.get("label") or "").strip()
        raw_values = tier.get("values", "")

        if not name:
            continue

        values = _split_values(raw_values)
        # Store one string for single value (regex), list for multi-value literals
        tier_entry = {"values": values if len(values) > 1 else (values[0] if values else "")}

        if tier.get("is_partition"):
            tier_entry["partition"] = True

        tiers_dict[name] = tier_entry

    # Top-level config
    config = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "reliability_fraction": reliability_fraction,
        "coders": coder_list,
        "exclude_participants": exclude_participants_list,
        "automate_POWERS": automate_POWERS,
        "tiers": tiers_dict,
    }

    yaml_config = yaml.dump(config, sort_keys=False, allow_unicode=True)
    yaml_bytes = BytesIO(yaml_config.encode("utf-8"))

    st.subheader("📄 YAML Config Preview")
    st.code(yaml_config, language="yaml")

    # Disable download if regex errors present
    disabled = len(regex_errors) > 0
    if disabled:
        st.warning("Fix the regex errors above to enable download.")

    st.download_button(
        "📥 Download config.yaml",
        data=yaml_bytes,
        file_name="config.yaml",
        mime="application/x-yaml",
        disabled=disabled
    )

    return config
