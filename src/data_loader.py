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
        # Filter for high quality ground truth
        filtered = chunk[
            (chunk['Assembly'] == 'GRCh38') & 
            (chunk['ClinSigSimple'].isin([0, 1])) & 
            (chunk['ReviewStatus'].isin(HIGH_CONFIDENCE_STATUS))
        ].dropna()
        
        data_list.append(filtered)
        current_count += len(filtered)
        
        if current_count >= row_limit:
            break
            
    df = pd.concat(data_list).reset_index(drop=True)
    
    # Remove duplicates to prevent leakage
    df = df.drop_duplicates(subset=['VariationID'])
    
    print(f"Dataset successfully loaded. Final size: {df.shape[0]} variants.")
    return df
