"""
Preprocessing Module - Nettoyage et préparation des données
Adapté pour le dataset clinvar_conflicting avec CLASS comme target
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle


# Les 7 features sélectionnées
# SIFT et PolyPhen sont encodés numériquement
SELECTED_FEATURES = [
    'CADD_PHRED',
    'CADD_RAW',
    'SIFT',
    'PolyPhen',
    'AF_EXAC',
    'AF_TGP',
    'BLOSUM62'
]

# Target est la colonne CLASS (0 = Benign, 1 = Pathogenic)
TARGET_COLUMN = 'CLASS'


def encode_sift(value):
    """Encode SIFT en valeurs numériques."""
    if pd.isna(value):
        return np.nan
    value = str(value).lower()
    if 'deleterious' in value:
        return 1.0  # Mauvais = pathogène potentiel
    elif 'tolerated' in value:
        return 0.0  # Bon = probablement bénin
    return np.nan


def encode_polyphen(value):
    """Encode PolyPhen en valeurs numériques."""
    if pd.isna(value):
        return np.nan
    value = str(value).lower()
    if 'probably_damaging' in value:
        return 1.0
    elif 'possibly_damaging' in value:
        return 0.5
    elif 'benign' in value:
        return 0.0
    return np.nan


def clean_clinvar_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Nettoie le dataset ClinVar pour la classification binaire.
    
    Args:
        df: DataFrame brut ClinVar
        verbose: Afficher les logs
    
    Returns:
        DataFrame nettoyé avec features et target
    """
    if verbose:
        print("\n" + "="*60)
        print("NETTOYAGE DES DONNÉES")
        print("="*60)
        print(f"Données brutes: {df.shape[0]:,} lignes")
    
    # Copie pour éviter de modifier l'original
    df_clean = df.copy()
    
    # 1. Vérifier la colonne target (CLASS)
    if TARGET_COLUMN not in df_clean.columns:
        raise ValueError(f"Colonne target '{TARGET_COLUMN}' non trouvée!")
    
    # Supprimer les lignes avec NaN dans CLASS
    df_clean = df_clean[df_clean[TARGET_COLUMN].notna()].copy()
    
    # 2. Créer la cible binaire (CLASS est déjà 0 ou 1)
    df_clean['target'] = df_clean[TARGET_COLUMN].astype(int)
    
    if verbose:
        target_counts = df_clean['target'].value_counts()
        print(f"\nDistribution de la cible (CLASS):")
        print(f"  Benign (0):     {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df_clean)*100:.1f}%)")
        print(f"  Pathogenic (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
    
    # 3. Sélectionner et encoder les features
    available_features = [f for f in SELECTED_FEATURES if f in df_clean.columns]
    missing_features = [f for f in SELECTED_FEATURES if f not in df_clean.columns]
    
    if missing_features and verbose:
        print(f"\n⚠️ Features manquantes: {missing_features}")
    
    # 4. Encoder SIFT et PolyPhen en valeurs numériques
    if 'SIFT' in available_features:
        df_clean['SIFT'] = df_clean['SIFT'].apply(encode_sift)
        if verbose:
            print(f"\nSIFT encodé: {df_clean['SIFT'].value_counts().to_dict()}")
    
    if 'PolyPhen' in available_features:
        df_clean['PolyPhen'] = df_clean['PolyPhen'].apply(encode_polyphen)
        if verbose:
            print(f"PolyPhen encodé: {df_clean['PolyPhen'].value_counts().to_dict()}")
    
    # 5. Convertir les autres features en numérique
    for feat in available_features:
        if feat not in ['SIFT', 'PolyPhen']:  # Déjà encodés
            df_clean[feat] = pd.to_numeric(df_clean[feat], errors='coerce')
    
    # 6. Gérer les valeurs manquantes
    for feat in available_features:
        missing_count = df_clean[feat].isnull().sum()
        if missing_count > 0:
            median_val = df_clean[feat].median()
            df_clean[feat] = df_clean[feat].fillna(median_val)
            if verbose:
                print(f"  {feat}: {missing_count:,} valeurs manquantes → médiane ({median_val:.4f})")
    
    # Sélectionner uniquement les colonnes utiles
    df_clean = df_clean[available_features + ['target']].copy()
    
    # Supprimer les lignes avec des NaN résiduels
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    final_rows = len(df_clean)
    
    if verbose and initial_rows != final_rows:
        print(f"\nLignes supprimées (NaN résiduels): {initial_rows - final_rows:,}")
    
    if verbose:
        print(f"\nDataset final: {df_clean.shape[0]:,} lignes × {df_clean.shape[1]} colonnes")
        print(f"Features utilisées: {available_features}")
    
    return df_clean, available_features


def prepare_train_test_data(df_clean: pd.DataFrame, 
                             test_size: float = 0.2,
                             random_state: int = 42,
                             verbose: bool = True) -> tuple:
    """
    Prépare les données pour l'entraînement.
    """
    if verbose:
        print("\n" + "="*60)
        print("PRÉPARATION TRAIN/TEST")
        print("="*60)
    
    # Séparer X et y
    feature_cols = [c for c in df_clean.columns if c != 'target']
    X = df_clean[feature_cols].values
    y = df_clean['target'].values
    
    if verbose:
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
    
    # Split train/test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    if verbose:
        print(f"\nTrain set: {X_train.shape[0]:,} samples")
        print(f"  Benign: {np.sum(y_train == 0):,} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
        print(f"  Pathogenic: {np.sum(y_train == 1):,} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
        print(f"\nTest set: {X_test.shape[0]:,} samples")
        print(f"  Benign: {np.sum(y_test == 0):,} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
        print(f"  Pathogenic: {np.sum(y_test == 1):,} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
    
    # Normalisation (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        print(f"\nNormalisation effectuée (StandardScaler)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def save_preprocessed_data(X_train, X_test, y_train, y_test, 
                           scaler, feature_names,
                           output_dir: str = None):
    """Sauvegarde les données prétraitées."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(X_train, columns=feature_names).to_csv(
        output_dir / "X_train.csv", index=False
    )
    pd.DataFrame(X_test, columns=feature_names).to_csv(
        output_dir / "X_test.csv", index=False
    )
    pd.DataFrame({'target': y_train}).to_csv(
        output_dir / "y_train.csv", index=False
    )
    pd.DataFrame({'target': y_test}).to_csv(
        output_dir / "y_test.csv", index=False
    )
    
    with open(output_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nDonnées sauvegardées dans: {output_dir}")


if __name__ == "__main__":
    from data_loader import load_clinvar_data
    
    df = load_clinvar_data(nrows=5000)
    df_clean, features = clean_clinvar_data(df)
    
    if len(df_clean) > 0:
        X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test_data(df_clean)
        save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, feature_names)
    else:
        print("Pas assez de données pour le prétraitement.")
