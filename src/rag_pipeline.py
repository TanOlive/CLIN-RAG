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

import gc
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from threading import Thread

from src.encoder import ClinicalVisionEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
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
        
        # Track active threads to prevent OOMs from aborted Streamlit runs
        import threading
        self.active_generation_thread = None
        self.generation_lock = threading.Lock()
        
        # We explicitly set attn_implementation="eager" because the PyTorch SDPA 'math' 
        # fallback is extremely slow on some Windows setups during autoregressive generation.
        self.processor = AutoProcessor.from_pretrained(
            model_id, token=self.hf_token
        )
        self.generator = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-1.5-4b-it",
            device_map=self.device,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            token=self.hf_token,
            attn_implementation="eager",
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

        with self.generation_lock:
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

    # ------------------------------------------------------------------
    # Image-token detection helpers
    # ------------------------------------------------------------------

    def _detect_num_image_tokens(self, input_ids: torch.Tensor) -> int:
        """Count how many image-placeholder tokens are in the input sequence.

        MedGemma / Gemma 3 prepends projected SigLIP patch embeddings as
        contiguous placeholder tokens.  This method finds their count by
        checking the model config and tokenizer for the image token ID.

        Returns 0 when detection fails (graceful degradation for XAI).
        """
        # Robust multi-fallback chain for the image token ID
        image_token_id: Optional[int] = getattr(
            self.generator.config, "image_token_index", None
        )
        if image_token_id is None:
            image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            try:
                image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                    "<image_soft_token>"
                )
                if image_token_id == self.processor.tokenizer.unk_token_id:
                    image_token_id = None
            except Exception:
                image_token_id = None

        if image_token_id is None:
            logger.warning(
                "Could not determine image token ID — XAI attention map "
                "will not be available."
            )
            return 0

        count = int((input_ids[0] == image_token_id).sum().item())
        logger.info(
            "Detected %d image tokens (token_id=%d) in input sequence.",
            count,
            image_token_id,
        )
        return count



    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report_stream(
        self, target_image_path: str | Path, retrieved_cases: List[Dict[str, Any]]
    ):
        """Generator that streams the MedGemma report chunk by chunk."""
        # Wait for any previous background thread to finish BEFORE doing any PyTorch ops
        if getattr(self, "active_generation_thread", None) is not None and self.active_generation_thread.is_alive():
            logger.warning("Found an existing generation thread still running! Joining it before starting a new one to prevent CUDA OOM...")
            self.active_generation_thread.join()
            
            # Force cleanup of the previous thread's memory before allocating for the new one
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with self.generation_lock:
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
        if not retrieved_cases:
            prompt_text = (
                "You are an expert clinical AI. You are provided with a target chest X-ray.\n\n"
                "CRITICAL INSTRUCTION - REASONING FIRST:\n"
                "1. VISUAL PRIMACY: Your diagnosis must rely EXCLUSIVELY on what you see in the image.\n"
                "2. DO NOT output the report directly. You MUST analyze the image step-by-step first.\n\n"
                "MANDATORY PROTOCOL:\n"
                "You MUST output your internal reasoning inside <clinical_reasoning> tags BEFORE generating the report.\n\n"
                "Output Format:\n"
                "<clinical_reasoning>\n"
                "1. Visual Perception: [Detail exactly what you observe in the current image]\n"
                "2. Synthesis: [Conclude your diagnosis based on visual facts]\n"
                "</clinical_reasoning>\n\n"
                "### FINDINGS:\n"
                "[Synthesize findings here]\n\n"
                "### IMPRESSION:\n"
                "[Synthesize impression here]\n"
            )
        else:
            prompt_text = (
                "You are an expert clinical AI. You are provided with a target chest X-ray and historical reports from visually similar precedent cases.\n\n"
                "CRITICAL INSTRUCTION - THE OMISSION RULE:\n"
                "1. The historical cases are for medical reference only. \n"
                "2. VISUAL PRIMACY: Your final diagnosis must rely EXCLUSIVELY on what you see in the current image.\n"
                "3. NEGATIVE DISTRACTION AVOIDANCE: If the historical cases mention a pathology (e.g., 'granuloma', 'tube') that is NOT visible in the current image, DO NOT mention it. DO NOT state that it is missing. Simply ignore it.\n"
                "4. Do not adopt highly specific measurements from the historical text unless you can visually verify them.\n\n"
                f"<historical_context>\n{context_str}\n</historical_context>\n\n"
                "MANDATORY PROTOCOL:\n"
                "You MUST output your internal reasoning inside <clinical_reasoning> tags BEFORE generating the report.\n\n"
                "Output Format:\n"
                "<clinical_reasoning>\n"
                "1. Visual Perception: [Detail exactly what you observe in the current image]\n"
                "2. Synthesis: [Conclude your diagnosis based on visual facts, ignoring irrelevant historical context]\n"
                "</clinical_reasoning>\n\n"
                "### FINDINGS:\n"
                "[Synthesize findings here]\n\n"
                "### IMPRESSION:\n"
                "[Synthesize impression here]\n"
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

        # -----------------------------------------------------------------
        # Detect image tokens BEFORE generation (needed for XAI)
        # -----------------------------------------------------------------
        num_image_tokens: int = self._detect_num_image_tokens(
            model_inputs["input_ids"]
        )

        # -----------------------------------------------------------------
        # Extract Vision Tower Self-Attention
        # -----------------------------------------------------------------
        vision_attention: Optional[np.ndarray] = None
        if "pixel_values" in model_inputs:
            logger.info("Extracting Vision Tower self-attention...")
            with torch.no_grad():
                vision_outputs = self.generator.model.vision_tower(
                    model_inputs["pixel_values"],
                    output_attentions=True,
                )
                if vision_outputs.attentions is not None:
                    # Extract the final layer's attention matrix
                    # Shape: (batch_size, num_heads, seq_len, seq_len)
                    vision_attention = vision_outputs.attentions[-1].to(torch.float32).cpu().numpy()

        # -----------------------------------------------------------------
        # Generate Report with Streaming
        # -----------------------------------------------------------------
        logger.info("Running MedGemma generate with streaming...")

        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.no_grad():
            generation_kwargs = dict(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,  # Greedy decoding for clinical factual consistency
                streamer=streamer,
            )

            thread = Thread(target=self.generator.generate, kwargs=generation_kwargs)
            self.active_generation_thread = thread
            thread.start()

            try:
                for new_text in streamer:
                    yield {"status": "streaming", "text": new_text}
                thread.join()
            finally:
                # If the generator is interrupted (e.g. Streamlit rerun), we must free references
                del model_inputs
                del generation_kwargs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info("Report generation completed successfully.")

        yield {
            "status": "complete",
            "retrieved_cases": retrieved_cases,
            "vision_attention": vision_attention,
            "num_image_tokens": num_image_tokens,
        }

    def generate_report(
        self, target_image_path: str | Path, retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synchronous wrapper for generate_report_stream for backwards compatibility."""
        full_text = ""
        final_data = {}
        for chunk in self.generate_report_stream(target_image_path, retrieved_cases):
            if chunk["status"] == "streaming":
                full_text += chunk["text"]
            elif chunk["status"] == "complete":
                final_data = chunk
                
        return {
            "generated_report": full_text.strip(),
            "retrieved_cases": final_data.get("retrieved_cases", []),
            "vision_attention": final_data.get("vision_attention"),
            "num_image_tokens": final_data.get("num_image_tokens", 0),
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
