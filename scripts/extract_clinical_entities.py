import os
import json
import logging
import pandas as pd
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Basic anatomical terms to filter out if they stand alone
STOP_WORDS = config.ENTITIES_STOP_WORDS

def extract_entities():
    if not config.REPORTS_CSV.exists():
        logger.error(f"Reports file not found at {config.REPORTS_CSV}")
        return

    logger.info(f"Loading reports from {config.REPORTS_CSV}")
    df = pd.read_csv(config.REPORTS_CSV)

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

    # Select Top N most frequent
    top_50 = [entity for entity, count in entity_counter.most_common(config.ENTITIES_TOP_K)]
    
    logger.info(f"Extracted {len(top_50)} clinical entities. Top 5: {top_50[:5]}")

    # Save to JSON
    config.ENTITIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.ENTITIES_PATH, "w") as f:
        json.dump(top_50, f, indent=4)
        
    logger.info(f"Successfully saved clinical entities to {config.ENTITIES_PATH}")

if __name__ == "__main__":
    extract_entities()
