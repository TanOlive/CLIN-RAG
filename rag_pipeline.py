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
    # Attention extraction (VRAM-safe)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image_attention(
        attentions_tuple: tuple,
        num_image_tokens: int,
    ) -> Optional[np.ndarray]:
        """Aggregate generator self-attention from generated tokens → image tokens.

        Parameters
        ----------
        attentions_tuple
            ``outputs.attentions`` from ``generate(output_attentions=True,
            return_dict_in_generate=True)``.  Structure:
            ``tuple[step]( tuple[layer]( Tensor(B, H, Q, KV) ) )``
            With KV-cache, decode steps have ``Q = 1``.
        num_image_tokens : int
            Number of image placeholder positions at the start of the
            input sequence.

        Returns
        -------
        np.ndarray or None
            2-D float32 array of shape ``(grid, grid)`` representing
            spatial attention, or *None* on failure.
        """
        if not attentions_tuple:
            return None

        # Collect attention grids for each step
        attention_grids = []

        for step_idx, step_attns in enumerate(attentions_tuple):
            # Select the last 4 layers
            selected_layers = list(step_attns[-4:])
            
            # Filter valid layers that contain all image tokens
            valid_layers = [L for L in selected_layers if L.shape[-1] >= num_image_tokens]
            if not valid_layers:
                continue

            # Average the valid layers. L shape: (batch, heads, q_len, kv_len)
            avg_attn = torch.stack(valid_layers).mean(dim=0)
            
            if step_idx == 0:
                # Prefill step: use the last query row
                token_attn = avg_attn[0, :, -1, :num_image_tokens]
            else:
                # Decode step: use the only query row
                token_attn = avg_attn[0, :, 0, :num_image_tokens]

            # Aggregate heads and move to CPU
            step_aggregated = token_attn.float().mean(dim=0).cpu().numpy()
            
            # Reshape to a 2-D spatial grid
            grid_size = int(np.sqrt(num_image_tokens))
            if grid_size * grid_size == num_image_tokens:
                grid = step_aggregated.astype(np.float32).reshape(grid_size, grid_size)
            else:
                grid_size = int(np.ceil(np.sqrt(num_image_tokens)))
                padded = np.zeros(grid_size * grid_size, dtype=np.float32)
                padded[:num_image_tokens] = step_aggregated.astype(np.float32)
                grid = padded.reshape(grid_size, grid_size)
                
            attention_grids.append(grid)

        if not attention_grids:
            logger.warning("No decode-step attention contributions found.")
            return None

        # Return 3D array: (num_steps, grid_h, grid_w)
        return np.stack(attention_grids)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self, target_image_path: str | Path, retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a structured clinical report using MedGemma and RAG context.

        Parameters
        ----------
        target_image_path : str | Path
            Path to the target chest X-ray image.
        retrieved_cases : List[Dict[str, Any]]
            The context dictionary list returned by ``retrieve_similar_cases``.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - ``retrieved_cases`` – the input cases (pass-through).
            - ``generated_report`` – the Markdown-formatted report text.
            - ``attention_map_2d`` – a 2-D ``np.ndarray`` (or *None*) of
              aggregated self-attention from generated tokens to image
              tokens, suitable for XAI heatmap visualisation.
            - ``num_image_tokens`` – count of image placeholder tokens.
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

        # -----------------------------------------------------------------
        # Detect image tokens BEFORE generation (needed for XAI)
        # -----------------------------------------------------------------
        num_image_tokens: int = self._detect_num_image_tokens(
            model_inputs["input_ids"]
        )

        # -----------------------------------------------------------------
        # Generate with attention capture
        # -----------------------------------------------------------------
        # output_attentions=True forces eager (non-SDPA) attention so that
        # per-head weight matrices are returned.  With KV-cache each decode
        # step stores only a (B, H, 1, KV) tensor, keeping VRAM overhead
        # manageable (~1-2 GB for Gemma-3 4B with 8 heads, 34 layers).
        logger.info("Running MedGemma generate with attention capture...")

        with torch.no_grad():
            outputs = self.generator.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,  # Greedy decoding for clinical factual consistency
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # -----------------------------------------------------------------
        # Extract image attention (CPU-safe, step-by-step)
        # -----------------------------------------------------------------
        attention_map_3d: Optional[np.ndarray] = None

        if num_image_tokens > 0 and outputs.attentions is not None:
            attention_map_3d = self._extract_image_attention(
                outputs.attentions, num_image_tokens
            )

        # -----------------------------------------------------------------
        # Decode the generated report and Individual Tokens
        # -----------------------------------------------------------------
        input_len = model_inputs["input_ids"].shape[1]
        output_ids = outputs.sequences
        generated_token_ids = output_ids[0][input_len:]
        
        # Cleanly decode tokens for UI presentation
        generated_tokens = []
        for tid in generated_token_ids:
            # Decode single token ID
            decoded_str = self.processor.decode([tid], skip_special_tokens=True)
            # Remove SentencePiece underscore meta characters if present
            decoded_str = decoded_str.replace(' ', ' ')
            generated_tokens.append(decoded_str)

        generated_text = self.processor.decode(
            generated_token_ids, skip_special_tokens=True
        ).strip()

        # Aggressively free the massive attention tensors
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Report generation completed successfully.")

        # --- LLM Zero-Shot Entity Extraction ---
        clinical_entities = self._extract_clinical_entities(target_image_path, generated_text)
        matched_indices = self._match_entities_to_tokens(clinical_entities, generated_tokens)
        logger.info(f"Extracted clinical entities: {clinical_entities}")

        return {
            "retrieved_cases": retrieved_cases,
            "generated_report": generated_text,
            "generated_tokens": generated_tokens,
            "attention_map_3d": attention_map_3d,
            "num_image_tokens": num_image_tokens,
            "clinical_entities": clinical_entities,
            "clinical_indices": matched_indices,
        }

    def _extract_clinical_entities(self, target_image_path: str, report_text: str) -> List[str]:
        logger.info("Running LLM zero-shot entity extraction...")
        prompt_text = (
            "Extract all key anatomical structures and pathological findings from the following text. "
            "Return ONLY a comma-separated list of the base terms without adjectives (e.g., 'lungs', 'opacity', 'pleural effusion').\n"
            f"Text: {report_text}\n"
            "Terms:"
        )
        
        image = Image.open(target_image_path).convert("RGB")
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
            messages, add_generation_prompt=True, tokenize=False
        )
        
        model_inputs = self.processor(
            text=inputs, images=image, return_tensors="pt"
        ).to(self.device)
        if self.device.type == "cuda":
            model_inputs["pixel_values"] = model_inputs["pixel_values"].to(torch.bfloat16)
            
        with torch.no_grad():
            outputs = self.generator.generate(
                **model_inputs,
                max_new_tokens=64,
                do_sample=False,
            )
            
        input_len = model_inputs["input_ids"].shape[1]
        generated_token_ids = outputs[0][input_len:]
        extracted_text = self.processor.decode(generated_token_ids, skip_special_tokens=True).strip()
        
        # Parse the comma-separated string
        entities = [e.strip().lower() for e in extracted_text.split(',') if e.strip()]
        return entities

    def _match_entities_to_tokens(self, entities: List[str], generated_tokens: List[str]) -> List[int]:
        matched_indices = set()
        for i, token in enumerate(generated_tokens):
            clean_token = token.strip().lower()
            if not clean_token or not clean_token.isalpha() or len(clean_token) < 2:
                continue
                
            for entity in entities:
                if clean_token in entity or entity in clean_token:
                    matched_indices.add(i)
                    break
        return sorted(list(matched_indices))


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
