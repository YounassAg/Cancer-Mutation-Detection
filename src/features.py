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
        self.origin_le = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.status_map = {
            'criteria provided, conflicting classifications': 1,
            'criteria provided, multiple submitters, no conflicts': 2,
            'reviewed by expert panel': 3,
            'practice guideline': 4
        }
        
        self.base_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

    def fit_transform(self, df):
        # Encodings
        df = df.copy()
        df['GeneIdx'] = self.gene_le.fit_transform(df['GeneID'].astype(str))
        df['TypeIdx'] = self.type_le.fit_transform(df['Type'].astype(str))
        df['ChromIdx'] = self.chrom_le.fit_transform(df['Chromosome'].astype(str))
        df['OriginIdx'] = self.origin_le.fit_transform(df['OriginSimple'].astype(str).fillna('unknown'))
        
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
        
        # Allele Length Difference (captures indel characteristics)
        df['AlleleLength'] = df.apply(
            lambda x: len(str(x['AlternateAlleleVCF'])) - len(str(x['ReferenceAlleleVCF'])),
            axis=1
        )
        
        # Hierarchical Positional Encoding
        # Overcomes float32 precision limits (16.7M) and scaler distortion
        df['Pos_MB'] = df['PositionVCF'] // 1000000
        df['Pos_KB'] = (df['PositionVCF'] % 1000000) // 1000
        df['Pos_B'] = df['PositionVCF'] % 1000
        
        # Store gene symbol mapping for inference reports
        gene_symbol_map = df.groupby('GeneID')['GeneSymbol'].first().to_dict()
        self.gene_symbol_map = gene_symbol_map
        
        # Prepare numeric inputs (10 features)
        X_num = df[[
            'Pos_MB', 'Pos_KB', 'Pos_B', 'RefEnc', 'AltEnc', 
            'NumberSubmitters', 'Stars', 'IsTransition', 'AlleleLength',
            'PositionVCF'  # Keeping scaled PositionVCF as a global context
        ]].values.copy()
        X_num = self.scaler.fit_transform(X_num)
        
        return {
            'gene': df['GeneIdx'].values,
            'type': df['TypeIdx'].values,
            'chrom': df['ChromIdx'].values,
            'origin': df['OriginIdx'].values,
            'numeric': X_num
        }

    def transform(self, df):
        df = df.copy()
        
        # Handle unseen categorical labels by safe mapping
        def safe_encode(le, values):
            classes_set = set(le.classes_)
            return [val if val in classes_set else le.classes_[0] for val in values]

        df['GeneIdx'] = self.gene_le.transform(safe_encode(self.gene_le, df['GeneID'].astype(str)))
        df['TypeIdx'] = self.type_le.transform(safe_encode(self.type_le, df['Type'].astype(str)))
        df['ChromIdx'] = self.chrom_le.transform(safe_encode(self.chrom_le, df['Chromosome'].astype(str)))
        df['OriginIdx'] = self.origin_le.transform(safe_encode(self.origin_le, df['OriginSimple'].astype(str).fillna('unknown')))
        
        df['Stars'] = df['ReviewStatus'].map(self.status_map).fillna(1)
        df['IsTransition'] = df.apply(
            lambda x: get_mutation_type(x['ReferenceAlleleVCF'], x['AlternateAlleleVCF']), 
            axis=1
        )
        df['RefEnc'] = df['ReferenceAlleleVCF'].apply(lambda x: self.base_map.get(str(x)[:1].upper(), 0))
        df['AltEnc'] = df['AlternateAlleleVCF'].apply(lambda x: self.base_map.get(str(x)[:1].upper(), 0))
        
        df['AlleleLength'] = df.apply(
            lambda x: len(str(x['AlternateAlleleVCF'])) - len(str(x['ReferenceAlleleVCF'])),
            axis=1
        )
        
        # Hierarchical Positional Encoding
        df['Pos_MB'] = df['PositionVCF'] // 1000000
        df['Pos_KB'] = (df['PositionVCF'] % 1000000) // 1000
        df['Pos_B'] = df['PositionVCF'] % 1000
        
        X_num = df[[
            'Pos_MB', 'Pos_KB', 'Pos_B', 'RefEnc', 'AltEnc', 
            'NumberSubmitters', 'Stars', 'IsTransition', 'AlleleLength',
            'PositionVCF'
        ]].values.copy()
        X_num = self.scaler.transform(X_num)
        
        return {
            'gene': df['GeneIdx'].values,
            'type': df['TypeIdx'].values,
            'chrom': df['ChromIdx'].values,
            'origin': df['OriginIdx'].values,
            'numeric': X_num
        }
