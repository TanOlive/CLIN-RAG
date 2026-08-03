"""
CLIN-RAG . Step 4 -- Full Retrieval-Augmented Generation Pipeline
=================================================================
This module orchestrates the full multimodal RAG pipeline:
1. Encodes an unseen target chest X-ray.
2. Queries the FAISS inner-product index to retrieve structurally
   similar historical cases.
3. Augments a generation prompt with the clinical text from the retrieved
   precedent cases.
4. Queries MedGemma (google/medgemma-1.5-4b-it) to generate a structured
   radiological report formatted with Markdown headers.

Author:  CLIN-RAG Team
Created: 2026-08-03
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from encoder import ClinicalVisionEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
TOKEN_PATH: Path = PROJECT_ROOT / "data" / "Hugging_Face_Access_Token.txt"
REPORTS_CSV: Path = PROJECT_ROOT / "data" / "archive" / "indiana_reports.csv"
INDEX_PATH: Path = PROJECT_ROOT / "data" / "clinical_index.faiss"
MAPPING_PATH: Path = PROJECT_ROOT / "data" / "index_mapping.pkl"
IMAGES_DIR: Path = PROJECT_ROOT / "data" / "archive" / "images" / "images_normalized"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


class ClinicalRAGSystem:
    """End-to-End RAG system for Multimodal Clinical Report Generation."""

    def __init__(self) -> None:
        """Initialise the RAG system: models, index, mapping, and reports."""
        logger.info("Initialising Clinical RAG System...")

        # 1. Load Hugging Face Authentication Token
        if not TOKEN_PATH.exists():
            raise FileNotFoundError(f"Missing HF token file at {TOKEN_PATH}")
        self.hf_token: str = TOKEN_PATH.read_text().strip()

        # 2. Determine Optimal Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info("Selected generation compute device: %s", self.device)

        # 3. Load the Vision Encoder (MedSigLIP)
        self.encoder = ClinicalVisionEncoder()

        # 4. Load the FAISS Index and Mapping
        if not INDEX_PATH.exists() or not MAPPING_PATH.exists():
            raise FileNotFoundError(
                "FAISS index or mapping file missing. Please run build_index.py first."
            )
        # Workaround for FAISS C++ failing on Windows Unicode paths (like 'ü')
        import shutil
        import os
        tmp_read_path = "temp_clinical_index_read.faiss"
        shutil.copy2(str(INDEX_PATH), tmp_read_path)
        self.index: faiss.IndexFlatIP = faiss.read_index(tmp_read_path)
        os.remove(tmp_read_path)
        with open(MAPPING_PATH, "rb") as f:
            self.mapping: List[Dict[str, Any]] = pickle.load(f)
        logger.info("Loaded FAISS index with %d vectors.", self.index.ntotal)

        # 5. Load Clinical Reports
        if not REPORTS_CSV.exists():
            raise FileNotFoundError(f"Missing reports file at {REPORTS_CSV}")
        # Note: uid is kept exactly as read (no string manipulation or hyphen splits)
        self.reports_df: pd.DataFrame = pd.read_csv(REPORTS_CSV)
        # Ensure uid is an integer for exact matching
        self.reports_df["uid"] = pd.to_numeric(self.reports_df["uid"], errors="coerce")
        logger.info("Loaded %d clinical reports.", len(self.reports_df))

        # 6. Load MedGemma Generator Model
        # Using bfloat16 to fit the 4B parameter model efficiently on GPU VRAM.
        model_id = "google/medgemma-1.5-4b-it"
        logger.info("Loading generator model '%s' ...", model_id)
        
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        
        self.processor = AutoProcessor.from_pretrained(
            model_id, token=self.hf_token
        )
        self.generator = AutoModelForImageTextToText.from_pretrained(
            model_id,
            token=self.hf_token,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.generator.eval()
        logger.info("MedGemma generator loaded successfully.")

    def retrieve_similar_cases(
        self, image_path: str | Path, k: int = 3
    ) -> List[Dict[str, Any]]:
        """Encode the target image and retrieve top-k similar historical cases.

        Parameters
        ----------
        image_path : str | Path
            Path to the raw target chest X-ray image.
        k : int, optional
            Number of similar cases to retrieve, by default 3.

        Returns
        -------
        List[Dict[str, Any]]
            A list of retrieved cases containing metadata and report text.
        """
        logger.info("Retrieving top-%d similar cases for image: %s", k, image_path)

        # Encode image (encoder STRICTLY preserves raw data integrity; no blurring)
        query_vector: np.ndarray = self.encoder.encode_image(str(image_path))
        query_vector = query_vector[np.newaxis, :]  # Expand to shape (1, D)

        # Query FAISS Index
        scores, indices = self.index.search(query_vector, k)
        
        retrieved_cases = []
        for rank, (score, faiss_id) in enumerate(zip(scores[0], indices[0])):
            if faiss_id == -1:
                continue  # Not enough vectors in the index

            # Map FAISS ID back to the patient uid and filename
            case_meta: Dict[str, Any] = self.mapping[faiss_id]
            uid: int = case_meta["uid"]

            # Exact matching on uid without any hyphen logic
            report_row = self.reports_df[self.reports_df["uid"] == uid]
            
            findings = "Not available."
            impression = "Not available."
            
            if not report_row.empty:
                # Use the first matched report (in case of duplicates)
                row = report_row.iloc[0]
                # Handle possible NaN values in the text columns
                if pd.notna(row.get("findings")):
                    findings = str(row["findings"])
                if pd.notna(row.get("impression")):
                    impression = str(row["impression"])

            retrieved_cases.append(
                {
                    "rank": rank + 1,
                    "score": float(score),
                    "uid": uid,
                    "filename": case_meta["filename"],
                    "projection": case_meta["projection"],
                    "findings": findings,
                    "impression": impression,
                }
            )

        return retrieved_cases

    def generate_report(
        self, target_image_path: str | Path, retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a structured clinical report using MedGemma and RAG context.

        Parameters
        ----------
        target_image_path : str | Path
            Path to the target chest X-ray image.
        retrieved_cases : List[Dict[str, Any]]
            The context dictionary list returned by `retrieve_similar_cases`.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the retrieved cases and the generated Markdown report.
        """
        logger.info("Constructing prompt and generating report...")

        # Load raw image for the generator (no destructive preprocessing)
        path = Path(target_image_path)
        if not path.exists():
            raise FileNotFoundError(f"Target image not found: {path}")
        image: Image.Image = Image.open(path).convert("RGB")
        image.load()

        # Build Context String from Historical Cases
        context_blocks = []
        for case in retrieved_cases:
            context_blocks.append(
                f"- Precedent Case (UID: {case['uid']}):\n"
                f"  Findings: {case['findings']}\n"
                f"  Impression: {case['impression']}"
            )
        context_str = "\n\n".join(context_blocks)

        # Construct Prompt enforcing structured output
        prompt_text = (
            f"""You are a highly precise, deterministic clinical reporting AI. 
            Your sole task is to generate a structured chest X-ray report based EXACTLY and ONLY on the provided Historical Context.
            
            STRICT RULES:
            1. NO DISCLAIMERS: Never output warnings like "I am an AI", "Please note", or "Requires a medical professional".
            2. NO HALLUCINATIONS: Do not mention anatomical structures (e.g., "osseous structures", "bones") unless they are explicitly present in the provided historical context.
            3. FORMAT: Always use exactly the two markdown headers "### FINDINGS:" and "### IMPRESSION:".
            
            [EXAMPLE - NORMAL CASE]
            Context Findings: The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax.
            Context Impression: Normal chest x-XXXX.
            
            Output:
            ### FINDINGS:
            The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema, focal consolidation, pleural effusion, or pneumothorax.
            
            ### IMPRESSION:
            Normal chest.
            [END EXAMPLE]
            
            Now, generate the report for the current case following these strict rules.
            
            HISTORICAL CONTEXT:
            {context_str}
            
            Output:
            """
        )

        # MedGemma / PaliGemma formatting: usually expects <image> token prepended or
        # specific chat formatting. Using the processor's chat template if available,
        # otherwise applying the raw prompt.
        try:
            # Modern instruction-tuned multimodal models typically use chat templates
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            # Process the templated string and the image
            model_inputs = self.processor(
                text=inputs, images=image, return_tensors="pt"
            ).to(self.device)
            # Ensure bfloat16 for visual inputs if needed by the model dtype
            if self.device.type == "cuda":
                model_inputs["pixel_values"] = model_inputs["pixel_values"].to(torch.bfloat16)

        except Exception as e:
            # Fallback for models that do not support apply_chat_template or require direct string input
            logger.warning("Chat template failed (%s), using raw string format.", e)
            model_inputs = self.processor(
                text=prompt_text, images=image, return_tensors="pt"
            ).to(self.device)
            if self.device.type == "cuda":
                model_inputs["pixel_values"] = model_inputs["pixel_values"].to(torch.bfloat16)

        # Generate Output
        with torch.no_grad():
            output_ids = self.generator.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,  # Greedy decoding for clinical factual consistency
            )

        # Decode output, skipping the prompt tokens
        input_len = model_inputs["input_ids"].shape[1]
        generated_text = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()

        logger.info("Report generation completed successfully.")

        return {
            "retrieved_cases": retrieved_cases,
            "generated_report": generated_text,
        }


# ---------------------------------------------------------------------------
# Test Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke test the pipeline using the first available image
    print("=" * 70)
    print("CLIN-RAG . RAG Pipeline Test")
    print("=" * 70)

    try:
        rag_sys = ClinicalRAGSystem()

        # Pick a sample image to test the full pipeline
        # (Assuming the images directory is populated based on Step 1)
        test_images = list(IMAGES_DIR.glob("*.png"))
        if not test_images:
            print("No test images found in data/archive/images/images_normalized/")
        else:
            sample_image = test_images[10]
            print(f"\nTarget Image: {sample_image.name}")
            print("-" * 70)

            # 1. Retrieve
            cases = rag_sys.retrieve_similar_cases(sample_image, k=3)
            print(f"\nRetrieved {len(cases)} Historical Cases:")
            for c in cases:
                print(f" [Rank {c['rank']}] UID: {c['uid']} (Sim: {c['score']:.4f})")
                print(f"    Findings: {c['findings'][:80]}...")
                print(f"    Impression: {c['impression'][:80]}...\n")

            # 2. Generate
            result = rag_sys.generate_report(sample_image, cases)
            print("-" * 70)
            print("Generated MedGemma Report:\n")
            print(result["generated_report"])
            print("-" * 70)

    except Exception as e:
        logger.error("Pipeline test failed: %s", e, exc_info=True)
