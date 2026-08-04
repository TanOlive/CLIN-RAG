"""
CLIN-RAG . Step 3 -- FAISS Vector Database & Indexing
======================================================
Builds a persistent FAISS inner-product index over the MedSigLIP
embeddings of every chest X-ray image in the Indiana University dataset.

Workflow
--------
1. Parse ``indiana_projections.csv`` to obtain the list of image
   filenames and their associated patient ``uid`` values.
2. Pre-validate that each referenced image file actually exists on disk.
3. Instantiate :class:`ClinicalVisionEncoder` (MedSigLIP-448).
4. Encode all valid images in mini-batches, accumulating L2-normalised
   embeddings of shape ``(N, 1152)``.
5. Insert the embedding matrix into a ``faiss.IndexFlatIP`` (inner-product
   index, which is equivalent to cosine similarity for unit-norm vectors).
6. Persist the index to ``data/clinical_index.faiss`` and a companion
   mapping file to ``data/index_mapping.pkl`` so that each integer FAISS
   ID can be resolved back to its source filename and patient uid.

Error Handling
--------------
* Missing image files are logged and skipped -- they do not crash the
  pipeline.
* Corrupt or unreadable images (e.g. truncated PNGs) are caught at
  encode time, logged, and skipped.

Author:  CLIN-RAG Team
Created: 2026-08-03
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from encoder import ClinicalVisionEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Project root (parent of this script).
PROJECT_ROOT: Path = Path(__file__).resolve().parent

#: Path to the projections metadata CSV.
PROJECTIONS_CSV: Path = PROJECT_ROOT / "data" / "archive" / "indiana_projections.csv"

#: Directory containing the normalised chest X-ray PNGs.
IMAGES_DIR: Path = PROJECT_ROOT / "data" / "archive" / "images" / "images_normalized"

#: Output path for the FAISS index file.
INDEX_OUTPUT: Path = PROJECT_ROOT / "data" / "clinical_index.faiss"

#: Output path for the FAISS-ID-to-metadata mapping.
MAPPING_OUTPUT: Path = PROJECT_ROOT / "data" / "index_mapping.pkl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Data Loading & Validation
# ---------------------------------------------------------------------------


def load_projection_records(csv_path: Path = PROJECTIONS_CSV) -> pd.DataFrame:
    """Load the projections CSV and return a clean DataFrame.

    The CSV contains three columns: ``uid`` (int), ``filename`` (str),
    and ``projection`` (str, e.g. 'Frontal' or 'Lateral').  No string
    cleaning or hyphen-based splitting is applied -- the identifiers are
    used exactly as they appear in the file.

    Parameters
    ----------
    csv_path : Path
        Absolute path to ``indiana_projections.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['uid', 'filename', 'projection']``.

    Raises
    ------
    FileNotFoundError
        If the CSV file is missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Projections CSV not found: {csv_path}")

    df: pd.DataFrame = pd.read_csv(csv_path)

    # Ensure the uid column is typed as a native Python int.
    df["uid"] = df["uid"].astype(int)

    # Exclude test cases if mapping exists
    test_meta_path = PROJECT_ROOT / "data" / "test_samples" / "test_metadata.csv"
    if test_meta_path.exists():
        test_df = pd.read_csv(test_meta_path)
        test_uids = set(test_df["uid"].astype(int))
        initial_count = len(df)
        df = df[~df["uid"].isin(test_uids)].copy()
        logger.info(
            "Excluded %d projection records matching test samples (UID mapping).",
            initial_count - len(df),
        )

    logger.info(
        "Loaded %d projection records (%d unique patients) from %s",
        len(df),
        df["uid"].nunique(),
        csv_path.name,
    )
    return df


def validate_image_paths(
    df: pd.DataFrame,
    images_dir: Path = IMAGES_DIR,
) -> tuple[pd.DataFrame, int]:
    """Filter the projections DataFrame to rows whose image files exist.

    Parameters
    ----------
    df : pd.DataFrame
        The raw projections DataFrame (must contain ``filename``).
    images_dir : Path
        Directory where normalised image files are stored.

    Returns
    -------
    tuple[pd.DataFrame, int]
        A tuple of ``(valid_df, n_missing)`` where *valid_df* contains
        only the rows with existing files and *n_missing* is the count
        of rows that were dropped.
    """
    exists_mask: pd.Series = df["filename"].apply(
        lambda fn: (images_dir / fn).exists()
    )

    n_missing: int = int((~exists_mask).sum())
    if n_missing > 0:
        missing_files: list[str] = df.loc[~exists_mask, "filename"].tolist()
        for fn in missing_files[:10]:  # Log first 10 to avoid flooding.
            logger.warning("Image file missing, will be skipped: %s", fn)
        if n_missing > 10:
            logger.warning("... and %d more missing files.", n_missing - 10)

    valid_df: pd.DataFrame = df.loc[exists_mask].reset_index(drop=True)
    logger.info(
        "Validated images: %d found, %d missing.",
        len(valid_df),
        n_missing,
    )
    return valid_df, n_missing


# ---------------------------------------------------------------------------
# Encoding Pipeline
# ---------------------------------------------------------------------------


def encode_images_safe(
    encoder: ClinicalVisionEncoder,
    image_paths: List[str],
    batch_size: int = 16,
) -> tuple[np.ndarray, List[int]]:
    """Encode images with graceful per-image error handling.

    Unlike ``encoder.encode_batch``, this function catches errors for
    individual images (e.g. corrupt / truncated PNGs) so that a single
    bad file does not invalidate an entire batch.

    Strategy
    --------
    * Attempt to process each mini-batch as a whole via the encoder.
    * If a batch-level exception occurs, fall back to encoding each
      image in that batch individually, skipping any that fail.

    Parameters
    ----------
    encoder : ClinicalVisionEncoder
        The initialised MedSigLIP encoder.
    image_paths : List[str]
        Ordered list of absolute image paths.
    batch_size : int
        Forward-pass mini-batch size.

    Returns
    -------
    tuple[np.ndarray, List[int]]
        ``(embeddings, valid_indices)`` where *embeddings* is an
        ``(M, D)`` float32 matrix of the successfully encoded images
        and *valid_indices* lists their original positions in
        ``image_paths`` (so the caller can align them with metadata).
    """
    all_embeddings: List[np.ndarray] = []
    valid_indices: List[int] = []
    total: int = len(image_paths)

    progress = tqdm(
        total=total,
        desc="Encoding images",
        unit="img",
        ncols=100,
    )

    for start in range(0, total, batch_size):
        end: int = min(start + batch_size, total)
        batch_paths: List[str] = image_paths[start:end]
        batch_indices: List[int] = list(range(start, end))

        try:
            # Attempt full-batch encoding (fast path).
            batch_embeddings: np.ndarray = encoder.encode_batch(
                batch_paths, batch_size=len(batch_paths)
            )
            # Shape: (batch_len, embedding_dim)
            all_embeddings.append(batch_embeddings)
            valid_indices.extend(batch_indices)

        except Exception:
            # Batch failed -- fall back to per-image encoding so we
            # only lose the truly broken images, not the whole batch.
            logger.warning(
                "Batch %d-%d failed; falling back to per-image encoding.",
                start,
                end,
            )
            for idx, path in zip(batch_indices, batch_paths):
                try:
                    emb: np.ndarray = encoder.encode_image(path)
                    # encode_image returns shape (D,); expand to (1, D).
                    all_embeddings.append(emb[np.newaxis, :])
                    valid_indices.append(idx)
                except Exception as img_err:
                    logger.warning(
                        "Skipping corrupt image at index %d (%s): %s",
                        idx,
                        Path(path).name,
                        img_err,
                    )

        progress.update(end - start)

    progress.close()

    if not all_embeddings:
        logger.error("No images were successfully encoded.")
        return np.empty((0, encoder.embedding_dim), dtype=np.float32), []

    # Stack all (variable-length) chunks into a single (M, D) matrix.
    embeddings: np.ndarray = np.vstack(all_embeddings)

    logger.info(
        "Encoding complete: %d / %d images succeeded.  Matrix shape: %s",
        len(valid_indices),
        total,
        embeddings.shape,
    )
    return embeddings, valid_indices


# ---------------------------------------------------------------------------
# FAISS Index Construction
# ---------------------------------------------------------------------------


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from a matrix of embeddings.

    Because all embeddings are L2-normalised to unit length, the inner
    product is algebraically equivalent to cosine similarity.  This
    allows the downstream retrieval step to use ``index.search(query, k)``
    and interpret the returned scores directly as cosine similarities.

    Parameters
    ----------
    embeddings : np.ndarray
        Float32 matrix of shape ``(N, D)`` with unit-norm rows.

    Returns
    -------
    faiss.IndexFlatIP
        A populated FAISS index containing *N* vectors of dimension *D*.
    """
    n_vectors, dim = embeddings.shape
    logger.info(
        "Building FAISS IndexFlatIP: %d vectors, %d dimensions.",
        n_vectors,
        dim,
    )

    index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # Bulk-insert all vectors at once.

    assert index.ntotal == n_vectors, (
        f"Index size mismatch: expected {n_vectors}, got {index.ntotal}"
    )
    logger.info("FAISS index built successfully. ntotal = %d", index.ntotal)
    return index


# ---------------------------------------------------------------------------
# Mapping Construction
# ---------------------------------------------------------------------------


def build_index_mapping(
    df: pd.DataFrame,
    valid_indices: List[int],
) -> List[Dict[str, Any]]:
    """Create a FAISS-ID-to-metadata mapping.

    The mapping is a list where ``mapping[faiss_id]`` yields a dictionary
    containing the source ``filename``, ``uid``, and ``projection`` for
    the vector stored at that position in the FAISS index.

    Parameters
    ----------
    df : pd.DataFrame
        The validated projections DataFrame (post-filtering).
    valid_indices : List[int]
        Indices into *df* for the images that were successfully encoded.

    Returns
    -------
    List[Dict[str, Any]]
        Ordered list (indexed by FAISS integer ID) of metadata dicts.
        Each dict has keys ``'faiss_id'``, ``'uid'``, ``'filename'``,
        and ``'projection'``.
    """
    mapping: List[Dict[str, Any]] = []

    for faiss_id, df_idx in enumerate(valid_indices):
        row: pd.Series = df.iloc[df_idx]
        mapping.append(
            {
                "faiss_id": faiss_id,
                "uid": int(row["uid"]),
                "filename": str(row["filename"]),
                "projection": str(row["projection"]),
            }
        )

    logger.info("Index mapping contains %d entries.", len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_artifacts(
    index: faiss.IndexFlatIP,
    mapping: List[Dict[str, Any]],
    index_path: Path = INDEX_OUTPUT,
    mapping_path: Path = MAPPING_OUTPUT,
) -> None:
    """Write the FAISS index and mapping to disk.

    Parameters
    ----------
    index : faiss.IndexFlatIP
        The populated FAISS index.
    mapping : List[Dict[str, Any]]
        The FAISS-ID-to-metadata mapping list.
    index_path : Path
        Destination for the FAISS index binary file.
    mapping_path : Path
        Destination for the pickled mapping file.
    """
    # Ensure the parent directories exist.
    index_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    # Workaround: FAISS's C++ FileIOWriter fails on Windows paths containing
    # Unicode characters (e.g., 'ü' in 'Prüfungsleistung'). We write to a 
    # safe ASCII relative temporary path, then move it via Python's shutil.
    import shutil
    tmp_path = "temp_clinical_index.faiss"
    faiss.write_index(index, tmp_path)
    shutil.move(tmp_path, str(index_path))
    
    logger.info("FAISS index saved to %s", index_path)

    with open(mapping_path, "wb") as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Index mapping saved to %s", mapping_path)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """End-to-end indexing pipeline: parse, encode, index, save."""
    print("=" * 70)
    print("CLIN-RAG . FAISS Index Builder")
    print("=" * 70)

    wall_start: float = time.time()

    # ------------------------------------------------------------------
    # 1. Load projection records
    # ------------------------------------------------------------------
    df: pd.DataFrame = load_projection_records()

    # ------------------------------------------------------------------
    # 2. Validate image file existence
    # ------------------------------------------------------------------
    valid_df, n_missing = validate_image_paths(df)

    if valid_df.empty:
        logger.error("No valid image files found. Aborting.")
        sys.exit(1)

    # Build the ordered list of absolute image paths.
    image_paths: List[str] = [
        str(IMAGES_DIR / fn) for fn in valid_df["filename"]
    ]

    # ------------------------------------------------------------------
    # 3. Initialise the MedSigLIP encoder
    # ------------------------------------------------------------------
    encoder = ClinicalVisionEncoder()

    # ------------------------------------------------------------------
    # 4. Encode all images (with graceful error handling)
    # ------------------------------------------------------------------
    embeddings, valid_indices = encode_images_safe(
        encoder,
        image_paths,
        batch_size=16,
    )

    if embeddings.shape[0] == 0:
        logger.error("Encoding produced zero embeddings. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Build the FAISS index
    # ------------------------------------------------------------------
    index: faiss.IndexFlatIP = build_faiss_index(embeddings)

    # ------------------------------------------------------------------
    # 6. Build the ID-to-metadata mapping
    # ------------------------------------------------------------------
    mapping: List[Dict[str, Any]] = build_index_mapping(valid_df, valid_indices)

    # ------------------------------------------------------------------
    # 7. Persist artifacts
    # ------------------------------------------------------------------
    save_artifacts(index, mapping)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    wall_elapsed: float = time.time() - wall_start
    minutes, seconds = divmod(wall_elapsed, 60)

    print("\n" + "-" * 70)
    print("Pipeline Summary")
    print("-" * 70)
    print(f"  Total projection records : {len(df):,}")
    print(f"  Missing image files      : {n_missing:,}")
    print(f"  Successfully encoded     : {embeddings.shape[0]:,}")
    print(f"  Embedding dimension      : {embeddings.shape[1]}")
    print(f"  FAISS index vectors      : {index.ntotal:,}")
    print(f"  Index file               : {INDEX_OUTPUT}")
    print(f"  Mapping file             : {MAPPING_OUTPUT}")
    print(f"  Wall time                : {int(minutes)}m {seconds:.1f}s")
    print("-" * 70)
    print("[OK] Indexing complete.")


if __name__ == "__main__":
    main()
