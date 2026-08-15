"""
CLIN-RAG · Step 5 – Streamlit User Interface
=============================================
Clinical Evidence Retrieval and Grounded Report Generation system.

This module provides the front-end user interface for the RAG pipeline.
It handles user uploads (chest X-rays), triggers the retrieval of historical
cases via the FAISS index, and displays the structured generative report 
produced by MedGemma, alongside the supporting clinical evidence.

The XAI heatmap is derived from MedGemma's own self-attention
(generated text tokens → prepended image tokens), reflecting the
generator's diagnostic reasoning rather than the retrieval encoder's
feature maps.

Author:  CLIN-RAG Team
Created: 2026-08-03
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import streamlit as st
from PIL import Image

from src.rag_pipeline import ClinicalRAGSystem
from src.xai_utils import process_vision_attention
from src import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

config.TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Resource Management
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading Clinical RAG Models & Vector DB into VRAM...", max_entries=1)
def load_system() -> ClinicalRAGSystem:
    """Instantiate and cache the ClinicalRAGSystem globally.
    
    Using @st.cache_resource ensures that the 4B parameter MedGemma model, 
    the MedSigLIP encoder, and the FAISS index are loaded exactly once 
    at startup, preventing memory leaks and severe reload delays.
    """
    return ClinicalRAGSystem()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


def main() -> None:
    if "is_analyzing" not in st.session_state:
        st.session_state["is_analyzing"] = False

    # 1. Page Configuration
    st.set_page_config(
        page_title="CLIN-RAG: Clinical Evidence Retrieval",
        page_icon="🩺",
        layout="wide",
    )

    # 2. Sidebar Settings
    st.sidebar.markdown(
        """
        ## ⚙️ Settings
        """
    )
    st.sidebar.markdown("---")
    
    k_cases: int = st.sidebar.slider(
        "Number of Reference Cases (k)",
        min_value=1,
        max_value=5,
        value=config.RETRIEVAL_K_CASES,
        help="The number of structurally similar historical cases to retrieve as context for the generator.",
        disabled=st.session_state["is_analyzing"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Explainability (XAI)")
    heatmap_threshold: float = st.sidebar.slider(
        "XAI Heatmap Focus",
        min_value=0.0,
        max_value=1.0,
        value=config.DEFAULT_HEATMAP_THRESHOLD,
        step=0.05,
        help="0.0 shows all activations. 1.0 hides the heatmap entirely. Use this to isolate high-attention (red) areas.",
        disabled=st.session_state["is_analyzing"]
    )

    st.sidebar.markdown("---")
    test_mode = st.sidebar.checkbox("Enable Test Mode (Ground Truth)", value=False, help="If a test sample is uploaded, this will display the original ground truth report at the bottom of the page.", disabled=st.session_state["is_analyzing"])

    # 3. Initialize Pipeline
    try:
        rag_sys: ClinicalRAGSystem = load_system()
    except Exception as e:
        st.error(f"Failed to initialize the RAG system: {e}")
        st.stop()

    # 4. Main Interface
    st.title("🏥 CLIN-RAG: Clinical Evidence Retrieval")
    st.markdown(
        "> **An AI-powered diagnostic assistant.** Upload a raw chest X-ray to retrieve structurally similar historical precedent cases and generate a grounded, structured clinical report."
    )
    st.markdown("---")

    # Layout for Upload & Target Display
    upload_col, target_col = st.columns([1, 1], gap="large")

    with upload_col:
        st.subheader("1. Upload Target Image")
        uploaded_file = st.file_uploader(
            "Upload a Chest X-ray image",
            type=["png", "jpg", "jpeg"],
            help="The image must not have any destructive preprocessing applied.",
            disabled=st.session_state["is_analyzing"]
        )

    if uploaded_file is not None:
        # Load and display the target image (preserve raw data integrity)
        try:
            target_image = Image.open(uploaded_file)
            target_image.load()
        except Exception as e:
            st.error(f"Failed to read image file: {e}")
            st.stop()



        # Clear cached results if a new file is uploaded
        current_file_name = uploaded_file.name
        if st.session_state.get("_xai_source_file") != current_file_name:
            # New file detected — purge stale cache
            for key in [
                "_xai_source_file",
                "xai_attention_map_3d",
                "generated_tokens",
                "generated_report",
                "retrieved_cases",
                "selected_token_idx",
                "gt_findings",
                "gt_impression",
                "clinical_reasoning",
            ]:
                st.session_state.pop(key, None)
            st.session_state["_xai_source_file"] = current_file_name

        st.markdown("---")
        
        # 5. Execution Logic
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            if not st.session_state["is_analyzing"]:
                if st.button("Generate Clinical Report", type="primary", disabled=uploaded_file is None):
                    st.session_state["is_analyzing"] = True
                    # Purge old results to force a fresh generation
                    for key in ["generated_report", "clinical_reasoning", "xai_vision_attention", "retrieved_cases"]:
                        st.session_state.pop(key, None)
                    st.rerun()
            else:
                if st.button("🛑 Start Over / Cancel", type="primary"):
                    st.session_state["is_analyzing"] = False
                    # Purge results to reset the UI
                    for key in ["generated_report", "clinical_reasoning", "xai_vision_attention", "retrieved_cases"]:
                        st.session_state.pop(key, None)
                    st.rerun()

        if st.session_state["is_analyzing"] and "generated_report" not in st.session_state:
            # We must write the uploaded file temporarily to disk because 
            # rag_sys.retrieve_similar_cases expects a file path.
            # We strictly avoid any image processing or filtering here.
            import uuid
            local_filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
            tmp_path = config.TEMP_UPLOAD_DIR / local_filename
            # Load and force RGB conversion to strip alpha/indexed channels
            img = Image.open(uploaded_file).convert("RGB")
            img.save(tmp_path)

            try:
                with st.spinner("Retrieving historical cases..."):
                    retrieved_cases: List[Dict[str, Any]] = rag_sys.retrieve_similar_cases(str(tmp_path), k=k_cases)

                reasoning_expander = st.expander("🧠 Live AI Clinical Chain of Thought", expanded=True)
                reasoning_placeholder = reasoning_expander.empty()

                full_text = ""
                reasoning_text = ""
                report_text = ""
                
                with st.spinner("Model is generating report..."):
                    for chunk in rag_sys.generate_report_stream(str(tmp_path), retrieved_cases):
                        if chunk["status"] == "streaming":
                            full_text += chunk["text"]
                            
                            # Parse reasoning block dynamically
                            reasoning_match = re.search(r"<clinical_reasoning>(.*?)(?:</clinical_reasoning>|$)", full_text, re.DOTALL)
                            if reasoning_match:
                                reasoning_text = reasoning_match.group(1).strip()
                                reasoning_placeholder.info(reasoning_text + " ▌")
                                
                                # If reasoning block has closed, remove the cursor
                                report_split = full_text.split("</clinical_reasoning>")
                                if len(report_split) > 1:
                                    reasoning_placeholder.info(reasoning_text)
                                
                        elif chunk["status"] == "complete":
                            # Finalize
                            if reasoning_text:
                                reasoning_placeholder.info(reasoning_text)
                            
                            report_split = full_text.split("</clinical_reasoning>")
                            if len(report_split) > 1:
                                report_text = report_split[1].strip()
                            else:
                                report_text = full_text.strip()
                                
                            # Cache data for the XAI Heatmap and UI refreshes
                            st.session_state["clinical_reasoning"] = reasoning_text
                            st.session_state["generated_report"] = report_text
                            st.session_state["xai_vision_attention"] = chunk["vision_attention"]
                            st.session_state["retrieved_cases"] = chunk["retrieved_cases"]

                if test_mode:
                    import pandas as pd
                    TEST_METADATA_PATH = config.TEST_METADATA_PATH
                    if TEST_METADATA_PATH.exists():
                        test_df = pd.read_csv(TEST_METADATA_PATH)
                        match = test_df[test_df["new_filename"] == uploaded_file.name]
                        if not match.empty:
                            st.session_state["gt_findings"] = str(match.iloc[0]["findings"])
                            st.session_state["gt_impression"] = str(match.iloc[0]["impression"])
                        else:
                            st.session_state["gt_findings"] = None
                            st.session_state["gt_impression"] = None

                st.session_state["is_analyzing"] = False
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {e}")
            finally:
                # Cleanup the temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if "clinical_reasoning" in st.session_state:
            with st.expander("🧠 View AI Clinical Chain of Thought (Reasoning Process)", expanded=False):
                st.info("This is the internal cognitive process of the AI before formulating the final diagnostic report.")
                st.markdown(st.session_state["clinical_reasoning"])

        # ---------------------------------------------------------------
        # Display Target Image & Heatmap
        # ---------------------------------------------------------------
        with target_col:
            st.subheader("Target Image & XAI Heatmap")
            
            # If we have generated the report and have a heatmap, display it here
            vision_attn = st.session_state.get("xai_vision_attention")

            if "generated_report" in st.session_state and vision_attn is not None:
                try:
                    heatmap_rgba: Image.Image = process_vision_attention(
                        vision_attention=vision_attn,
                        original_image=target_image,
                        threshold=heatmap_threshold,
                    )
                    base: Image.Image = target_image.convert("RGBA")
                    if heatmap_rgba.size != base.size:
                        heatmap_rgba = heatmap_rgba.resize(base.size, Image.BILINEAR)
                    composite: Image.Image = Image.alpha_composite(base, heatmap_rgba)
                    st.image(composite, width='stretch', caption=f"Vision Encoder Attention Saliency (threshold = {heatmap_threshold:.2f})")
                except Exception as e:
                    st.warning(f"XAI heatmap rendering failed: {e}")
                    st.image(target_image, width='stretch', caption="Uploaded Radiograph")
            else:
                st.image(target_image, width='stretch', caption="Uploaded Radiograph")


        # ---------------------------------------------------------------
        # Display cached results (persists across slider re-runs)
        # ---------------------------------------------------------------
        if "generated_report" in st.session_state:
            
            # ------ Generated Report ------
            st.subheader("2. AI-Generated Clinical Report")
            st.info("The following report was generated by MedGemma (4B) using the retrieved historical context.", icon="🤖")
            
            with st.container(border=True):
                st.markdown(st.session_state["generated_report"])

            st.markdown("---")

            # ------ Clinical Evidence ------
            retrieved_cases = st.session_state.get("retrieved_cases", [])
            st.subheader("3. Clinical Evidence (Retrieved Precedents)")
            st.markdown("These are the most structurally similar historical cases retrieved from the FAISS database to ground the generated report.")

            if retrieved_cases:
                tabs = st.tabs([f"Case {i+1} (UID: {c['uid']})" for i, c in enumerate(retrieved_cases)])
                
                for idx, case in enumerate(retrieved_cases):
                    with tabs[idx]:
                        # Fetch the actual historical image from the dataset
                        ref_image_path = config.IMAGES_DIR / case["filename"]
                        
                        ev_col_text, ev_col_img = st.columns([2, 1], gap="medium")
                        
                        with ev_col_text:
                            st.markdown(f"**Similarity Score:** `{case['score']:.4f}`")
                            st.markdown(f"**Projection:** `{case['projection']}`")
                            
                            st.markdown("#### Historical Findings")
                            st.warning(case["findings"], icon="🔬")
                            
                            st.markdown("#### Historical Impression")
                            st.success(case["impression"], icon="📝")
                            
                        with ev_col_img:
                            if ref_image_path.exists():
                                ref_img = Image.open(ref_image_path)
                                st.image(ref_img, width='stretch', caption=case["filename"])
                            else:
                                st.error("Reference image file missing from dataset archive.")
            else:
                st.warning("No similar cases were retrieved.")

            # ------ Ground Truth ------
            if test_mode and st.session_state.get("gt_findings") is not None:
                st.markdown("---")
                st.subheader("4. Ground Truth (Test Mode)")
                st.info("The actual, human-written radiological report from the dataset for this specific test image.", icon="🎯")
                
                st.markdown("#### Actual Findings")
                st.warning(st.session_state["gt_findings"], icon="🔬")
                
                st.markdown("#### Actual Impression")
                st.success(st.session_state["gt_impression"], icon="📝")

if __name__ == "__main__":
    main()
