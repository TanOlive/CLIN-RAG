"""
CLIN-RAG · Step 2 – Vision Embedding Pipeline (MedSigLIP)
==========================================================
Generates dense image embeddings from chest X-rays using Google's
MedSigLIP-448 vision encoder.

Architecture
------------
``ClinicalVisionEncoder`` wraps the gated ``google/medsiglip-448`` model
from Hugging Face.  It exposes two public methods:

* ``encode_image``  – embeds a single image  → 1-D numpy vector
* ``encode_batch``  – embeds N images at once → (N, D) numpy matrix

Both return **L2-normalised** vectors suitable for direct FAISS indexing.

Critical design rule
--------------------
**No destructive preprocessing is applied to the raw image data.**
The only transforms are those performed internally by the SigLIP
``AutoProcessor`` (deterministic resize to 448 × 448 and channel-wise
normalisation with published mean/std).  In particular, Gaussian blur
and any other smoothing / filtering operations are strictly forbidden to
preserve high-frequency micro-structures that the MedSigLIP encoder was
trained to exploit.

Author:  CLIN-RAG Team
Created: 2026-08-03
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hugging Face model identifier for MedSigLIP at 448 × 448 resolution.
MODEL_ID: str = "google/medsiglip-448"

#: Default location of the Hugging Face access token (project-local).
_TOKEN_PATH: Path = Path(__file__).resolve().parent / "data" / "Hugging_Face_Access_Token.txt"

#: Directory containing the normalised Indiana University X-ray PNGs.
_IMAGES_DIR: Path = (
    Path(__file__).resolve().parent / "data" / "archive" / "images" / "images_normalized"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_hf_token(path: Path = _TOKEN_PATH) -> str:
    """Read and return the Hugging Face access token from a local file.

    Parameters
    ----------
    path : Path
        Absolute path to the token text file.  The file is expected to
        contain a single line with the token string.

    Returns
    -------
    str
        The stripped token string ready for use with ``huggingface_hub``.

    Raises
    ------
    FileNotFoundError
        If the token file does not exist at the expected location.
    ValueError
        If the token file is empty or contains only whitespace.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Hugging Face token file not found at {path}. "
            "Please create a file containing your HF access token."
        )
    token: str = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Token file at {path} is empty.")
    logger.info("Hugging Face token loaded from %s", path)
    return token


def _resolve_device() -> torch.device:
    """Detect and return the optimal hardware accelerator.

    Selection priority: CUDA → MPS (Apple Silicon) → CPU.

    Returns
    -------
    torch.device
        The selected PyTorch device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Selected compute device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Core Encoder
# ---------------------------------------------------------------------------


class ClinicalVisionEncoder:
    """MedSigLIP-based image encoder for clinical chest X-rays.

    This class loads the ``google/medsiglip-448`` vision–language model
    and exposes methods to produce L2-normalised image embeddings.

    Parameters
    ----------
    model_id : str, optional
        Hugging Face model identifier.  Defaults to ``google/medsiglip-448``.
    token_path : Path or None, optional
        Path to the Hugging Face access-token file.  When *None*, the
        default project location is used.
    device : torch.device or None, optional
        Compute device.  When *None*, the best available accelerator is
        auto-detected (CUDA → MPS → CPU).

    Attributes
    ----------
    model : SiglipModel
        The loaded SigLIP model in evaluation mode.
    processor : AutoProcessor
        The corresponding image processor (resize + normalise only).
    device : torch.device
        The active compute device.
    embedding_dim : int
        Dimensionality of the output embedding vector.

    Examples
    --------
    >>> encoder = ClinicalVisionEncoder()
    >>> vec = encoder.encode_image("data/archive/images/images_normalized/1_IM-0001-4001.dcm.png")
    >>> vec.shape
    (768,)
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        token_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        # --- Token -----------------------------------------------------------
        effective_path: Path = token_path if token_path is not None else _TOKEN_PATH
        token: str = _read_hf_token(effective_path)

        # --- Device ----------------------------------------------------------
        self.device: torch.device = device if device is not None else _resolve_device()

        # --- Model & Processor -----------------------------------------------
        logger.info("Loading MedSigLIP model '%s' …", model_id)

        self.processor: AutoProcessor = AutoProcessor.from_pretrained(
            model_id,
            token=token,
        )

        self.model: SiglipModel = SiglipModel.from_pretrained(
            model_id,
            token=token,
        )
        # Move model to the chosen device and switch to eval mode so that
        # dropout / batch-norm layers are deactivated.
        self.model = self.model.to(self.device).eval()

        # Store embedding dimensionality for downstream consumers.
        self.embedding_dim: int = self.model.config.vision_config.hidden_size

        logger.info(
            "MedSigLIP ready — device=%s, embedding_dim=%d",
            self.device,
            self.embedding_dim,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_raw_image(filepath: str) -> Image.Image:
        """Load an image from disk **without any destructive preprocessing**.

        The image is opened with Pillow, converted to RGB (required by
        SigLIP), and returned.  No Gaussian blur, sharpening, histogram
        equalisation, or any other filtering is applied.

        Parameters
        ----------
        filepath : str
            Path to the image file (typically a ``.png``).

        Returns
        -------
        PIL.Image.Image
            The loaded image in RGB mode.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img: Image.Image = Image.open(path).convert("RGB")
        # Materialise pixel data so the file handle is released.
        img.load()
        return img

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_image(self, image_path: str) -> np.ndarray:
        """Embed a single medical image and return a normalised 1-D vector.

        Processing pipeline (all non-destructive):
        1. Load the raw image via PIL (RGB conversion only).
        2. Pass through the ``AutoProcessor`` which applies a deterministic
           resize to 448 × 448 and channel-wise mean/std normalisation.
        3. Forward through the SigLIP vision tower to obtain the pooled
           image embedding.
        4. L2-normalise the vector on the unit sphere.

        Parameters
        ----------
        image_path : str
            Path to the medical image file.

        Returns
        -------
        np.ndarray
            A 1-D float32 array of shape ``(embedding_dim,)`` with unit
            L2 norm, ready for FAISS indexing.
        """
        image: Image.Image = self._load_raw_image(image_path)

        # Processor output is a BatchEncoding with key "pixel_values"
        # of shape (1, 3, 448, 448).
        inputs = self.processor(images=image, return_tensors="pt")
        # Move every tensor in the batch to the target device.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # get_image_features may return a raw tensor (older transformers)
            # or a BaseModelOutputWithPooling (transformers ≥ 5.x).  We
            # handle both cases to stay forward-compatible.
            output = self.model.get_image_features(**inputs)
            features: torch.Tensor = (
                output if isinstance(output, torch.Tensor) else output.pooler_output
            )

        # L2-normalise so that cosine similarity reduces to a dot product,
        # which FAISS IndexFlatIP can exploit directly.
        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        # Detach from the computation graph, move to CPU, and convert to
        # a 1-D numpy array with float32 dtype.
        embedding: np.ndarray = features.squeeze(0).cpu().numpy().astype(np.float32)
        return embedding

    def encode_batch(
        self,
        image_paths: Sequence[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Embed multiple images efficiently in mini-batches.

        Parameters
        ----------
        image_paths : Sequence[str]
            An ordered collection of file paths to medical images.
        batch_size : int, optional
            Number of images processed in a single forward pass.
            Tune this based on available GPU memory.  Defaults to 16.

        Returns
        -------
        np.ndarray
            A float32 matrix of shape ``(N, embedding_dim)`` where each
            row is an L2-normalised embedding.  The row order matches
            the input ``image_paths`` order.
        """
        all_embeddings: List[np.ndarray] = []
        total: int = len(image_paths)

        for start in range(0, total, batch_size):
            end: int = min(start + batch_size, total)
            batch_paths: Sequence[str] = image_paths[start:end]

            # Load all images for the current batch.
            images: List[Image.Image] = [
                self._load_raw_image(p) for p in batch_paths
            ]

            # The processor accepts a list of PIL images and returns a
            # stacked tensor of shape (batch, 3, 448, 448).
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.get_image_features(**inputs)
                features: torch.Tensor = (
                    output if isinstance(output, torch.Tensor) else output.pooler_output
                )

            features = torch.nn.functional.normalize(features, p=2, dim=-1)
            batch_np: np.ndarray = features.cpu().numpy().astype(np.float32)
            all_embeddings.append(batch_np)

            logger.info(
                "Encoded batch %d–%d / %d",
                start + 1,
                end,
                total,
            )

        # Vertically stack all mini-batch results into a single matrix.
        return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("CLIN-RAG · MedSigLIP Encoder — Smoke Test")
    print("=" * 60)

    # Instantiate the encoder (downloads model on first run).
    encoder = ClinicalVisionEncoder()

    # Pick the first available image from the normalised directory.
    sample_images: list[Path] = sorted(_IMAGES_DIR.glob("*.png"))
    if not sample_images:
        print(f"[ERROR] No .png files found in {_IMAGES_DIR}")
        sys.exit(1)

    test_path: str = str(sample_images[0])
    print(f"\nTest image : {test_path}")
    print(f"Device     : {encoder.device}")
    print(f"Model      : {MODEL_ID}")

    # --- Single image test ---------------------------------------------------
    embedding: np.ndarray = encoder.encode_image(test_path)
    norm: float = float(np.linalg.norm(embedding))

    print(f"\n[Single]  shape = {embedding.shape}")
    print(f"[Single]  dtype = {embedding.dtype}")
    print(f"[Single]  L2 norm = {norm:.6f}  (expected ~= 1.0)")
    print(f"[Single]  first 5 values = {embedding[:5]}")

    # --- Batch test (first 3 images) -----------------------------------------
    batch_paths: list[str] = [str(p) for p in sample_images[:3]]
    batch_embeddings: np.ndarray = encoder.encode_batch(batch_paths, batch_size=2)

    print(f"\n[Batch]   shape = {batch_embeddings.shape}")
    print(f"[Batch]   dtype = {batch_embeddings.dtype}")
    norms: np.ndarray = np.linalg.norm(batch_embeddings, axis=1)
    print(f"[Batch]   L2 norms = {norms}")

    print("\n[OK] Smoke test passed.")
