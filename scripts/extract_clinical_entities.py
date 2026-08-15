import os
import json
import logging
import pandas as pd
from collections import Counter
from pathlib import Path

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_CSV = PROJECT_ROOT / "data" / "archive" / "indiana_reports.csv"
OUTPUT_JSON = PROJECT_ROOT / "data" / "archive" / "indiana_reports_clinical_entities.json"

# Basic anatomical terms to filter out if they stand alone
STOP_WORDS = {
    "normal", "lung", "lungs", "pleura", "heart", "mediastinum", 
    "diaphragm", "bone", "bones", "spine", "rib", "ribs", 
    "chest", "thoracic", "aorta", "cardiac silhouette",
    "pulmonary", "soft tissue", "trachea", "clavicle", "scapula"
}

def extract_entities():
    if not REPORTS_CSV.exists():
        logger.error(f"Reports file not found at {REPORTS_CSV}")
        return

    logger.info(f"Loading reports from {REPORTS_CSV}")
    df = pd.read_csv(REPORTS_CSV)

    if "Problems" not in df.columns:
        logger.error("Column 'Problems' not found in dataset.")
        return

    entity_counter = Counter()

    for problems_str in df["Problems"].dropna():
        # Semicolon-separated
        entities = str(problems_str).split(";")
        
        for entity in entities:
            # Lowercase and strip whitespace
            clean_entity = entity.lower().strip()
            
            # Filter out empty strings, "normal", and standalone anatomical terms
            if not clean_entity:
                continue
            if clean_entity in STOP_WORDS:
                continue
                
            entity_counter[clean_entity] += 1

    # Select Top 50 most frequent
    top_50 = [entity for entity, count in entity_counter.most_common(50)]
    
    logger.info(f"Extracted {len(top_50)} clinical entities. Top 5: {top_50[:5]}")

    # Save to JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(top_50, f, indent=4)
        
    logger.info(f"Successfully saved clinical entities to {OUTPUT_JSON}")

if __name__ == "__main__":
    extract_entities()
