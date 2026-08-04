"""
CLIN-RAG · XAI Utilities – Attention Heatmap Overlays
=====================================================
Provides two complementary XAI visualisation functions:

1. **Encoder-based** (``generate_attention_heatmap``):
   Extracts patch-level activation magnitudes from the MedSigLIP vision
   encoder.  Useful as a lightweight, pre-generation diagnostic.

2. **Generator-based** (``process_generator_attention``):
   Takes the aggregated 2-D spatial attention map produced by
   ``ClinicalRAGSystem.generate_report`` (self-attention from
   generated text tokens → prepended image tokens in MedGemma)
   and converts it into a thresholded RGBA heatmap overlay.
   This reflects the actual diagnostic reasoning of the generator.

Critical Constraints
--------------------
* **No Gaussian blur** is applied to the heatmap, the alpha mask, or
  the final composite – ever.
* Upscaling uses strictly ``cv2.INTER_LINEAR`` (bilinear).

Author:  CLIN-RAG Team
Created: 2026-08-04
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from encoder import ClinicalVisionEncoder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_attention_heatmap(
    image_path: str,
    encoder_model: ClinicalVisionEncoder,
    threshold: float = 0.0,
) -> Image.Image:
    """Produce a thresholded RGBA heatmap from MedSigLIP patch activations.

    Parameters
    ----------
    image_path : str
        Absolute path to the target chest X-ray image file.
    encoder_model : ClinicalVisionEncoder
        An already-initialised encoder whose ``.model``, ``.processor``,
        and ``.device`` attributes are used directly.
    threshold : float, optional
        Activation cut-off in [0.0, 1.0].  Normalised activation values
        **below** this threshold are rendered fully transparent (alpha = 0),
        exposing the original radiograph underneath.  ``0.0`` shows the
        full heatmap; ``1.0`` hides it entirely.

    Returns
    -------
    PIL.Image.Image
        An RGBA image of the **same pixel dimensions** as the original
        radiograph, suitable for direct compositing via
        ``Image.alpha_composite``.
    """
    # ------------------------------------------------------------------
    # 1. Load the raw image (preserving original dimensions for later)
    # ------------------------------------------------------------------
    original_image: Image.Image = encoder_model._load_raw_image(image_path)
    orig_w, orig_h = original_image.size  # PIL uses (width, height)

    # ------------------------------------------------------------------
    # 2. Preprocess & forward-pass through the vision tower
    # ------------------------------------------------------------------
    inputs = encoder_model.processor(
        images=original_image, return_tensors="pt"
    )
    inputs = {k: v.to(encoder_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        vision_outputs = encoder_model.model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
        )

    # last_hidden_state shape: (1, num_patches, hidden_dim)
    last_hidden: torch.Tensor = vision_outputs.last_hidden_state

    # ------------------------------------------------------------------
    # 3. Compute per-patch activation magnitude (L2 norm)
    # ------------------------------------------------------------------
    # Shape: (num_patches,)
    patch_activations: torch.Tensor = torch.norm(
        last_hidden.squeeze(0), p=2, dim=-1
    )

    # Determine the spatial grid size from the model config
    vision_cfg = encoder_model.model.config.vision_config
    image_size: int = vision_cfg.image_size   # e.g. 448
    patch_size: int = vision_cfg.patch_size    # e.g. 14
    grid_size: int = image_size // patch_size  # e.g. 32

    # Reshape to 2-D spatial grid
    activation_grid: np.ndarray = (
        patch_activations.cpu().numpy().reshape(grid_size, grid_size)
    )

    # ------------------------------------------------------------------
    # 4. Normalise activations strictly to [0.0, 1.0]
    # ------------------------------------------------------------------
    a_min: float = float(activation_grid.min())
    a_max: float = float(activation_grid.max())

    if a_max - a_min > 1e-8:
        activation_grid = (activation_grid - a_min) / (a_max - a_min)
    else:
        # Degenerate case: uniform activations → flat mid-grey
        activation_grid = np.full_like(activation_grid, 0.5)

    # ------------------------------------------------------------------
    # 5. Upscale to original image dimensions (bilinear, NO blur)
    # ------------------------------------------------------------------
    heatmap_full: np.ndarray = cv2.resize(
        activation_grid.astype(np.float32),
        (orig_w, orig_h),             # (width, height) for cv2.resize
        interpolation=cv2.INTER_LINEAR,
    )

    # ------------------------------------------------------------------
    # 6. Apply JET colourmap → BGR → RGB
    # ------------------------------------------------------------------
    heatmap_uint8: np.ndarray = (heatmap_full * 255).astype(np.uint8)
    heatmap_bgr: np.ndarray = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb: np.ndarray = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # 7. Build the alpha channel with dynamic thresholding
    # ------------------------------------------------------------------
    #   • activations >= threshold → semi-transparent (alpha ≈ 0.55 × 255)
    #   • activations <  threshold → fully transparent (alpha = 0)
    # A moderate base opacity lets the radiograph detail remain visible
    # even where the heatmap is active.
    BASE_ALPHA: int = 140  # ~55 % opacity for active regions

    alpha: np.ndarray = np.where(
        heatmap_full >= threshold,
        np.uint8(BASE_ALPHA),
        np.uint8(0),
    ).astype(np.uint8)

    # ------------------------------------------------------------------
    # 8. Compose the RGBA image
    # ------------------------------------------------------------------
    rgba: np.ndarray = np.dstack([heatmap_rgb, alpha])
    heatmap_pil: Image.Image = Image.fromarray(rgba, mode="RGBA")

    logger.info(
        "XAI heatmap generated for %s (grid=%d×%d, threshold=%.2f)",
        Path(image_path).name,
        grid_size,
        grid_size,
        threshold,
    )
    return heatmap_pil


# ---------------------------------------------------------------------------
# Generator-based XAI (MedGemma self-attention → image tokens)
# ---------------------------------------------------------------------------


def process_generator_attention(
    raw_attention_2d: np.ndarray,
    original_image_size: Tuple[int, int],
    threshold: float = 0.0,
) -> Image.Image:
    """Convert an aggregated 2-D attention grid into a thresholded RGBA heatmap.

    This function takes the spatial attention map produced by
    ``ClinicalRAGSystem._extract_image_attention`` (MedGemma
    self-attention: generated tokens → prepended image tokens)
    and renders it as an RGBA overlay ready for compositing.

    Parameters
    ----------
    raw_attention_2d : np.ndarray
        2-D float array of shape ``(grid_h, grid_w)`` produced by the
        generator’s attention extraction pipeline.
    original_image_size : tuple[int, int]
        ``(width, height)`` of the original radiograph in pixels
        (PIL convention).
    threshold : float, optional
        Activation cut-off in ``[0.0, 1.0]``.  Normalised values
        **below** this threshold are rendered fully transparent
        (alpha = 0).  ``0.0`` shows the full heatmap; ``1.0`` hides
        it entirely.

    Returns
    -------
    PIL.Image.Image
        RGBA image at ``original_image_size`` ready for
        ``Image.alpha_composite``.
    """
    orig_w, orig_h = original_image_size

    # ------------------------------------------------------------------
    # 1. Normalise to [0.0, 1.0]
    # ------------------------------------------------------------------
    grid = raw_attention_2d.astype(np.float64)
    
    # Statistical Sink Obliteration
    p99 = np.percentile(grid, 99)
    median_val = np.median(grid)
    grid[grid > p99] = median_val

    a_min, a_max = float(grid.min()), float(grid.max())

    if a_max - a_min > 1e-8:
        grid = (grid - a_min) / (a_max - a_min)
    else:
        grid = np.full_like(grid, 0.5)

    grid = grid.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Upscale to original image dimensions (bilinear, NO blur)
    # ------------------------------------------------------------------
    heatmap_full: np.ndarray = cv2.resize(
        grid,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # ------------------------------------------------------------------
    # 3. Apply JET colourmap → BGR → RGB
    # ------------------------------------------------------------------
    heatmap_uint8: np.ndarray = (heatmap_full * 255).astype(np.uint8)
    heatmap_bgr: np.ndarray = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb: np.ndarray = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # 4. Build the alpha channel with dynamic thresholding
    # ------------------------------------------------------------------
    BASE_ALPHA: int = 140  # ~55 % opacity for active regions

    if threshold >= 1.0:
        alpha = np.zeros_like(heatmap_full, dtype=np.uint8)
    else:
        alpha = np.where(
            heatmap_full >= threshold,
            np.uint8(BASE_ALPHA),
            np.uint8(0),
        ).astype(np.uint8)

    # ------------------------------------------------------------------
    # 5. Compose the RGBA image
    # ------------------------------------------------------------------
    rgba: np.ndarray = np.dstack([heatmap_rgb, alpha])
    heatmap_pil: Image.Image = Image.fromarray(rgba, mode="RGBA")

    logger.info(
        "Generator XAI heatmap processed (grid=%d×%d, target=%d×%d, threshold=%.2f)",
        raw_attention_2d.shape[1],
        raw_attention_2d.shape[0],
        orig_w,
        orig_h,
        threshold,
    )
    return heatmap_pil
