"""
Data Loader Module - Charge et explore le dataset ClinVar
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_clinvar_data(filepath: str = None, nrows: int = None) -> pd.DataFrame:
    """
    Charge le dataset ClinVar depuis le fichier CSV.
    
    Args:
        filepath: Chemin vers le fichier CSV (par défaut: data/clinvar_conflicting.csv)
        nrows: Nombre de lignes à charger (None = toutes)
    
    Returns:
        DataFrame pandas avec les données ClinVar
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "clinvar_conflicting.csv"
    
    print(f"Chargement des données depuis: {filepath}")
    df = pd.read_csv(filepath, nrows=nrows)
    print(f"Dataset chargé: {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """
    Exploration rapide des données.
    
    Args:
        df: DataFrame ClinVar
    
    Returns:
        Dictionnaire avec les statistiques d'exploration
    """
    print("\n" + "="*60)
    print("EXPLORATION DES DONNÉES")
    print("="*60)
    
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    print(f"\nDimensions: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"Mémoire utilisée: {stats['memory_usage_mb']:.2f} MB")
    
    # Vérifier la colonne cible
    target_col = 'CLNSIGINCL'
    if target_col in df.columns:
        print(f"\nDistribution de la cible ({target_col}):")
        print(df[target_col].value_counts(dropna=False))
    
    return stats


def get_features_info() -> dict:
    """
    Retourne les informations sur les 7 features sélectionnées.
    
    Returns:
        Dictionnaire avec les noms et descriptions des features
    """
    features_info = {
        'CADD_PHRED': {
            'description': 'Score de pathogénicité combiné (0-100)',
            'importance': 'Très élevée - Score expert #1',
            'threshold': '> 20 = probablement pathogène'
        },
        'CADD_RAW': {
            'description': 'Version brute du score CADD',
            'importance': 'Élevée - Complète CADD_PHRED',
            'threshold': 'Variable selon le contexte'
        },
        'SIFT': {
            'description': 'Score d\'impact sur la protéine (0-1)',
            'importance': 'Très élevée - Score expert #2',
            'threshold': '< 0.05 = délétère (mauvais)'
        },
        'PolyPhen': {
            'description': 'Prédiction d\'effet sur acide aminé (0-1)',
            'importance': 'Très élevée - Score expert #3',
            'threshold': '> 0.85 = probablement dangereux'
        },
        'AF_EXAC': {
            'description': 'Fréquence population EXAC (0-1)',
            'importance': 'Élevée - Rareté de la mutation',
            'threshold': 'Faible = plus suspecte'
        },
        'AF_TGP': {
            'description': 'Fréquence population 1000 Genomes (0-1)',
            'importance': 'Élevée - Confirmation de rareté',
            'threshold': 'Faible = plus suspecte'
        },
        'BLOSUM62': {
            'description': 'Score de changement d\'acide aminé',
            'importance': 'Moyenne - Sévérité du changement',
            'threshold': 'Négatif = changement grave'
        }
    }
    return features_info


if __name__ == "__main__":
    # Test du module
    df = load_clinvar_data(nrows=1000)
    explore_data(df)
    
    features = get_features_info()
    print("\n" + "="*60)
    print("FEATURES SÉLECTIONNÉES")
    print("="*60)
    for feat, info in features.items():
        print(f"\n{feat}:")
        print(f"  Description: {info['description']}")
        print(f"  Importance: {info['importance']}")
