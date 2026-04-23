import pandas as pd
from .config import (
    DATA_PATH, INPUT_COLUMNS, HIGH_CONFIDENCE_STATUS,
    RELAXED_REVIEW_STATUS, CANCER_KEYWORDS,
    ONCOGENIC_LABELS, SOMATIC_IMPACT_POSITIVE
)

def load_clean_data(row_limit=None):
    """
    Loads ClinVar data with a multi-tiered cancer-specific labeling strategy.
    
    Label Construction (CancerLabel):
      - Tier 1 (Gold):   Oncogenicity ∈ {"Oncogenic", "Likely oncogenic"} → 1
      - Tier 2 (Silver): SomaticClinicalImpact ∈ {"Tier I", "Tier II"}   → 1
      - Tier 3 (Bronze): ClinSigSimple==1 AND PhenotypeList matches cancer keywords → 1
      - Negative:        ClinSigSimple==0 → 0
      - Excluded:        Everything else (uncertain, unclassified)
    
    Review Status Filtering:
      - Tier 1/2 variants: Accept RELAXED_REVIEW_STATUS (includes "conflicting")
      - Tier 3/Negative:   Require HIGH_CONFIDENCE_STATUS only
    """
    reader = pd.read_csv(
        DATA_PATH, 
        sep='\t', 
        usecols=INPUT_COLUMNS, 
        chunksize=100000, 
        low_memory=False
    )
    
    cancer_pattern = '|'.join(CANCER_KEYWORDS)
    data_list = []
    current_count = 0
    
    for chunk in reader:
        # Filter for GRCh38 assembly only
        chunk = chunk[chunk['Assembly'] == 'GRCh38']
        
        # --- Tier 1 (Gold): Direct oncogenicity annotation ---
        is_oncogenic = chunk['Oncogenicity'].isin(ONCOGENIC_LABELS)
        
        # --- Tier 2 (Silver): Somatic clinical impact ---
        has_somatic_impact = chunk['SomaticClinicalImpact'].isin(SOMATIC_IMPACT_POSITIVE)
        
        # --- Tier 3 (Bronze): ClinSigSimple + cancer phenotype ---
        is_pathogenic = chunk['ClinSigSimple'] == 1
        has_cancer_phenotype = chunk['PhenotypeList'].str.contains(
            cancer_pattern, case=False, na=False
        )
        
        # --- Negative class: Reliably benign ---
        is_benign = chunk['ClinSigSimple'] == 0
        
        # Review status filtering (relaxed for Tier 1/2, strict for Tier 3/Negative)
        has_relaxed_review = chunk['ReviewStatus'].isin(RELAXED_REVIEW_STATUS)
        has_strict_review = chunk['ReviewStatus'].isin(HIGH_CONFIDENCE_STATUS)
        
        # Positive variants (any tier)
        tier1_positive = is_oncogenic & has_relaxed_review
        tier2_positive = has_somatic_impact & has_relaxed_review & (~tier1_positive)
        tier3_positive = is_pathogenic & has_cancer_phenotype & has_strict_review & (~tier1_positive) & (~tier2_positive)
        
        all_positive = tier1_positive | tier2_positive | tier3_positive
        
        # Negative variants (strict review only)
        all_negative = is_benign & has_strict_review
        
        # Combine and assign CancerLabel
        positives = chunk[all_positive].copy()
        positives['CancerLabel'] = 1
        
        negatives = chunk[all_negative].copy()
        negatives['CancerLabel'] = 0
        
        filtered = pd.concat([positives, negatives])
        
        # Drop rows with missing critical fields
        filtered = filtered.dropna(subset=[
            'GeneID', 'PositionVCF', 'ReferenceAlleleVCF', 
            'AlternateAlleleVCF', 'Type', 'Chromosome'
        ])
        
        data_list.append(filtered)
        current_count += len(filtered)
        
        if row_limit is not None and current_count >= row_limit:
            break
            
    df = pd.concat(data_list).reset_index(drop=True)
    
    # Remove duplicates to prevent leakage
    df = df.drop_duplicates(subset=['VariationID'])
    
    # Fill missing OriginSimple with 'unknown'
    df['OriginSimple'] = df['OriginSimple'].fillna('unknown')
    
    # Report label distribution
    n_pos = (df['CancerLabel'] == 1).sum()
    n_neg = (df['CancerLabel'] == 0).sum()
    print(f"Dataset successfully loaded. Final size: {df.shape[0]} variants.")
    print(f"  Oncogenic (positive): {n_pos} ({n_pos/len(df)*100:.1f}%)")
    print(f"  Benign (negative):    {n_neg} ({n_neg/len(df)*100:.1f}%)")
    
    return df
