import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_mutation_type(ref, alt):
    """
    Classifies a mutation as Transition (Ti) or Transversion (Tv).
    Ti: A <-> G, C <-> T
    Tv: others
    """
    ref = str(ref)[:1].upper()
    alt = str(alt)[:1].upper()
    
    ti_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
    
    if (ref, alt) in ti_pairs:
        return 1 # Transition
    return 0 # Transversion

class FeatureEngineer:
    def __init__(self):
        self.gene_le = LabelEncoder()
        self.type_le = LabelEncoder()
        self.chrom_le = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.status_map = {
            'criteria provided, multiple submitters, no conflicts': 2,
            'reviewed by expert panel': 3,
            'practice guideline': 4
        }
        
        self.base_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

    def fit_transform(self, df):
        # Encodings
        df['GeneIdx'] = self.gene_le.fit_transform(df['GeneID'].astype(str))
        df['TypeIdx'] = self.type_le.fit_transform(df['Type'].astype(str))
        df['ChromIdx'] = self.chrom_le.fit_transform(df['Chromosome'].astype(str))
        
        # Medical Evidence Quality
        df['Stars'] = df['ReviewStatus'].map(self.status_map).fillna(1)
        
        # Ti/Tv Analysis
        df['IsTransition'] = df.apply(
            lambda x: get_mutation_type(x['ReferenceAlleleVCF'], x['AlternateAlleleVCF']), 
            axis=1
        )
        
        # Nucleotide Encoding
        df['RefEnc'] = df['ReferenceAlleleVCF'].apply(lambda x: self.base_map.get(str(x)[:1].upper(), 0))
        df['AltEnc'] = df['AlternateAlleleVCF'].apply(lambda x: self.base_map.get(str(x)[:1].upper(), 0))
        
        # Gene Pathogenicity Frequency (Self-Knowledge)
        gene_path_freq = df.groupby('GeneID')['ClinSigSimple'].mean().to_dict()
        self.gene_path_map = gene_path_freq
        df['GenePathFreq'] = df['GeneID'].map(self.gene_path_map).fillna(0)
        
        # Prepare inputs
        X_num = df[['PositionVCF', 'RefEnc', 'AltEnc', 'NumberSubmitters', 'Stars', 'IsTransition', 'GenePathFreq']].values
        X_num = self.scaler.fit_transform(X_num)
        
        return {
            'gene': df['GeneIdx'].values,
            'type': df['TypeIdx'].values,
            'chrom': df['ChromIdx'].values,
            'numeric': X_num
        }

    def transform(self, df):
        # Handle unseen categorical labels by safe mapping
        def safe_encode(le, values):
            classes_set = set(le.classes_)
            # Map unseen to the first class or a placeholder if added
            return [val if val in classes_set else le.classes_[0] for val in values]

        df['GeneIdx'] = self.gene_le.transform(safe_encode(self.gene_le, df['GeneID'].astype(str)))
        df['TypeIdx'] = self.type_le.transform(safe_encode(self.type_le, df['Type'].astype(str)))
        df['ChromIdx'] = self.chrom_le.transform(safe_encode(self.chrom_le, df['Chromosome'].astype(str)))
        
        df['Stars'] = df['ReviewStatus'].map(self.status_map).fillna(1)
        df['IsTransition'] = df.apply(
            lambda x: get_mutation_type(x['ReferenceAlleleVCF'], x['AlternateAlleleVCF']), 
            axis=1
        )
        # Apply Gene Pathogenicity Frequency
        df['GenePathFreq'] = df['GeneID'].map(self.gene_path_map).fillna(0)
        
        df['RefEnc'] = df['ReferenceAlleleVCF'].apply(lambda x: self.base_map.get(str(x)[:1].upper(), 0))
        df['AltEnc'] = df['AlternateAlleleVCF'].apply(lambda x: self.base_map.get(str(x)[:1].upper(), 0))
        
        X_num = df[['PositionVCF', 'RefEnc', 'AltEnc', 'NumberSubmitters', 'Stars', 'IsTransition', 'GenePathFreq']].values
        X_num = self.scaler.transform(X_num)
        
        return {
            'gene': df['GeneIdx'].values,
            'type': df['TypeIdx'].values,
            'chrom': df['ChromIdx'].values,
            'numeric': X_num
        }
