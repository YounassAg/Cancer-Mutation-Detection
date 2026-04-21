import pandas as pd
from .config import DATA_PATH, INPUT_COLUMNS, HIGH_CONFIDENCE_STATUS

def load_clean_data(row_limit=1000000):
    """
    Loads ClinVar data using chunked reading to handle memory constraints.
    Filters for high-confidence variants (2+ stars) and GRCh38 assembly.
    """
    reader = pd.read_csv(
        DATA_PATH, 
        sep='\t', 
        usecols=INPUT_COLUMNS, 
        chunksize=100000, 
        low_memory=False
    )
    
    data_list = []
    current_count = 0
    
    for chunk in reader:
        # Filter for high quality ground truth and assembly
        chunk = chunk[
            (chunk['Assembly'] == 'GRCh38') & 
            (chunk['ReviewStatus'].isin(HIGH_CONFIDENCE_STATUS))
        ]
        
        # Cancer Specific Filtering:
        # 1. Keep all Benign (0)
        # 2. Keep Pathogenic (1) ONLY if PhenotypeList contains cancer keywords
        from .config import CANCER_KEYWORDS
        pattern = '|'.join(CANCER_KEYWORDS)
        
        is_benign = chunk['ClinSigSimple'] == 0
        is_pathogenic = chunk['ClinSigSimple'] == 1
        has_cancer_phenotype = chunk['PhenotypeList'].str.contains(pattern, case=False, na=False)
        
        filtered = chunk[is_benign | (is_pathogenic & has_cancer_phenotype)].dropna()
        
        data_list.append(filtered)
        current_count += len(filtered)
        
        if current_count >= row_limit:
            break
            
    df = pd.concat(data_list).reset_index(drop=True)
    
    # Remove duplicates to prevent leakage
    df = df.drop_duplicates(subset=['VariationID'])
    
    print(f"Dataset successfully loaded. Final size: {df.shape[0]} variants.")
    return df
