"""
CLIN-RAG · Step 5 – Streamlit User Interface
=============================================
Clinical Evidence Retrieval and Grounded Report Generation system.

This module provides the front-end user interface for the RAG pipeline.
It handles user uploads (chest X-rays), triggers the retrieval of historical
cases via the FAISS index, and displays the structured generative report 
produced by MedGemma, alongside the supporting clinical evidence.

Author:  CLIN-RAG Team
Created: 2026-08-03
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from PIL import Image

from rag_pipeline import ClinicalRAGSystem
from xai_utils import generate_attention_heatmap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
# Pointing to the archive images to fetch reference images for display
IMAGES_DIR: Path = PROJECT_ROOT / "data" / "archive" / "images" / "images_normalized"
TEMP_UPLOAD_DIR: Path = PROJECT_ROOT / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Resource Management
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading Clinical RAG Models & Vector DB into VRAM...")
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
    # 1. Page Configuration
    st.set_page_config(
        page_title="CLIN-RAG: Clinical Evidence Retrieval",
        page_icon="🩺",
        layout="wide",
    )

    # 2. Sidebar Settings
    st.sidebar.markdown(
        """
        ## ⚙️ CLIN-RAG Settings
        Configure the retrieval parameters for the RAG pipeline.
        """
    )
    st.sidebar.markdown("---")
    
    k_cases: int = st.sidebar.slider(
        "Number of Reference Cases (k)",
        min_value=1,
        max_value=5,
        value=3,
        help="The number of structurally similar historical cases to retrieve as context for the generator."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Explainability (XAI)")
    heatmap_threshold: float = st.sidebar.slider(
        "XAI Heatmap Focus",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="0.0 shows all activations. 1.0 hides the heatmap entirely. Use this to isolate high-attention (red) areas.",
    )

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
        )

    if uploaded_file is not None:
        # Load and display the target image (preserve raw data integrity)
        try:
            target_image = Image.open(uploaded_file)
            target_image.load()
        except Exception as e:
            st.error(f"Failed to read image file: {e}")
            st.stop()

        # ---------------------------------------------------------------
        # Persist uploaded file to disk so the XAI util can read it.
        # We reuse the same temp path across slider re-runs.
        # ---------------------------------------------------------------
        xai_tmp_path = TEMP_UPLOAD_DIR / f"xai_{uploaded_file.name}"
        if not xai_tmp_path.exists():
            img_for_disk = Image.open(uploaded_file).convert("RGB")
            img_for_disk.save(xai_tmp_path)

        with target_col:
            st.subheader("Target Image")
            st.image(target_image, width='stretch', caption="Uploaded Radiograph")

        # ---------------------------------------------------------------
        # XAI Heatmap Overlay (reacts to slider without rerunning RAG)
        # ---------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔬 XAI Attention Heatmap")
        st.caption(
            "Visualises MedSigLIP patch activations.  "
            "Use the sidebar slider to isolate high-attention (red) regions."
        )

        try:
            heatmap_rgba: Image.Image = generate_attention_heatmap(
                image_path=str(xai_tmp_path),
                encoder_model=rag_sys.encoder,
                threshold=heatmap_threshold,
            )

            # Composite: convert target to RGBA, paste heatmap on top
            base: Image.Image = target_image.convert("RGBA")
            # Ensure both layers share the same dimensions
            if heatmap_rgba.size != base.size:
                heatmap_rgba = heatmap_rgba.resize(base.size, Image.BILINEAR)
            composite: Image.Image = Image.alpha_composite(base, heatmap_rgba)

            xai_col_overlay, xai_col_orig = st.columns(2, gap="large")
            with xai_col_overlay:
                st.image(
                    composite,
                    width='stretch',
                    caption=f"Attention Overlay  (threshold = {heatmap_threshold:.2f})",
                )
            with xai_col_orig:
                st.image(
                    target_image,
                    width='stretch',
                    caption="Original Radiograph (reference)",
                )
        except Exception as e:
            st.warning(f"XAI heatmap generation failed: {e}")

        st.markdown("---")
        
        # 5. Execution Logic
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            generate_clicked = st.button("Generate Clinical Report", type="primary", width='stretch')

        if generate_clicked:
            # We must write the uploaded file temporarily to disk because 
            # rag_sys.retrieve_similar_cases expects a file path.
            # We strictly avoid any image processing or filtering here.
            local_filename = uploaded_file.name
            tmp_path = TEMP_UPLOAD_DIR / local_filename
            # Load and force RGB conversion to strip alpha/indexed channels
            img = Image.open(uploaded_file).convert("RGB")
            img.save(tmp_path)

            try:
                with st.spinner("Retrieving historical cases and generating report..."):
                    # Execute RAG Pipeline
                    retrieved_cases: List[Dict[str, Any]] = rag_sys.retrieve_similar_cases(str(tmp_path), k=k_cases)
                    result: Dict[str, Any] = rag_sys.generate_report(str(tmp_path), retrieved_cases)
                    generated_report: str = result["generated_report"]

                # 6. Display Layout - Section 1: Generated Report
                st.subheader("2. AI-Generated Clinical Report")
                st.info("The following report was generated by MedGemma (4B) using the retrieved historical context.", icon="🤖")
                
                with st.container(border=True):
                    st.markdown(generated_report)

                st.markdown("---")

                # 7. Display Layout - Section 2: Clinical Evidence
                st.subheader("3. Clinical Evidence (Retrieved Precedents)")
                st.markdown("These are the most structurally similar historical cases retrieved from the FAISS database to ground the generated report.")

                # Create tabs or columns to organize the evidence cleanly
                if retrieved_cases:
                    tabs = st.tabs([f"Case {i+1} (UID: {c['uid']})" for i, c in enumerate(retrieved_cases)])
                    
                    for idx, case in enumerate(retrieved_cases):
                        with tabs[idx]:
                            # Fetch the actual historical image from the dataset
                            ref_image_path = IMAGES_DIR / case["filename"]
                            
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

            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {e}")
            finally:
                # Cleanup the temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


if __name__ == "__main__":
    main()
