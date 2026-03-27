"""
prepare_domain_data.py — Prepares the Domain Categorizer training dataset.
Maps DS-1 (LiveCareer) categories to the project's standard 7 domains.
"""

import logging
import pandas as pd
from pathlib import Path
from src.config import RAW_DIR, PROCESSED_DIR, DOMAIN_LABELS
from src.preprocessing.data_loader import load_livecareer_resumes
from src.preprocessing.text_cleaner import quick_clean

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Category to Domain mapping
# Maps DS-1 Category names (string) -> Domain Index (int)
CATEGORY_MAP = {
    "INFORMATION-TECHNOLOGY": 0,
    "ENGINEERING": 0,
    
    "HR": 1,
    "BUSINESS-DEVELOPMENT": 1,
    "CONSULTANT": 1,
    "BPO": 1,
    "PUBLIC-RELATIONS": 1,
    "SALES": 1,
    "AGRICULTURE": 1,
    "AUTOMOBILE": 1,
    "AVIATION": 1,
    "CHEF": 1,
    "CONSTRUCTION": 1,
    
    "ARTS": 2,
    "DESIGNER": 2,
    "DIGITAL-MEDIA": 2,
    "APPAREL": 2,
    
    "HEALTHCARE": 3,
    "FITNESS": 3,
    
    "ACCOUNTANT": 4,
    "BANKING": 4,
    "FINANCE": 4,
    
    "ADVOCATE": 5,
    
    "TEACHER": 6
}

def main():
    logger.info("Starting Domain Data Preparation...")
    
    # 1. Load DS-1 Resumes
    df = load_livecareer_resumes()
    if df.empty:
        logger.error("No resumes found in data/raw/resume_dataset. Aborting.")
        return

    logger.info("Loaded %d resumes from DS-1.", len(df))

    # 2. Map Categories to Domain Indices
    logger.info("Mapping categories to domain indices...")
    df['domain_idx'] = df['category'].map(CATEGORY_MAP)
    
    # Drop rows that couldn't be mapped (if any)
    initial_count = len(df)
    df = df.dropna(subset=['domain_idx'])
    df['domain_idx'] = df['domain_idx'].astype(int)
    
    if len(df) < initial_count:
        logger.warning("Dropped %d rows due to unmapped categories.", initial_count - len(df))

    # 3. Clean Text
    logger.info("Cleaning resume text (this may take a minute)...")
    # We use quick_clean for the classifier. 
    # The classifier handles the dense representation, but we want to strip HTML/rubbish first.
    df['clean_text'] = df['resume_text'].apply(quick_clean)
    
    # Drop rows that became empty after cleaning
    df = df[df['clean_text'].str.len() > 50]

    # 4. Save processed dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "domain_training_data.csv"
    
    # Keep only relevant columns to save space
    df_final = df[['clean_text', 'domain_idx']]
    
    logger.info("Final dataset size: %d rows.", len(df_final))
    logger.info("Domain distribution:\n%s", df_final['domain_idx'].value_counts())
    
    df_final.to_csv(out_path, index=False)
    logger.info("Saved domain training data to %s", out_path)

if __name__ == "__main__":
    main()
