"""
Training Module - Entraînement des 3 modèles
"""
import numpy as np
import pickle
from pathlib import Path
from tensorflow import keras

from models import create_xgboost_model, create_simple_nn, create_advanced_nn


def calculate_class_weights(y_train):
    """Calcule les poids de classe pour gérer le déséquilibre."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


def train_xgboost(X_train, y_train, X_val=None, y_val=None, 
                  random_state: int = 42, verbose: bool = True):
    """
    Entraîne le modèle XGBoost.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        X_val: Features de validation (optionnel)
        y_val: Cible de validation (optionnel)
        random_state: Seed
        verbose: Afficher les logs
    
    Returns:
        Modèle entraîné
    """
    if verbose:
        print("\n" + "="*60)
        print("ENTRAÎNEMENT XGBOOST")
        print("="*60)
    
    # Calculer scale_pos_weight pour gérer le déséquilibre
    # scale_pos_weight = nombre de négatifs / nombre de positifs
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    if verbose:
        print(f"Distribution des classes: {n_neg} négatifs, {n_pos} positifs")
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    model = create_xgboost_model(random_state=random_state)
    model.set_params(scale_pos_weight=scale_pos_weight)
    
    if verbose:
        print(f"Paramètres: n_estimators=200, max_depth=8, learning_rate=0.05")
        print(f"Samples d'entraînement: {len(X_train):,}")
    
    # Entraînement
    model.fit(X_train, y_train, verbose=False)
    
    if verbose:
        print("✓ Entraînement terminé")
    
    return model


def train_simple_nn(X_train, y_train, 
                    epochs: int = 50,
                    batch_size: int = 32,
                    validation_split: float = 0.2,
                    random_state: int = 42,
                    verbose: bool = True):
    """
    Entraîne le Neural Network Simple.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        epochs: Nombre d'époques
        batch_size: Taille des batches
        validation_split: Fraction pour validation
        random_state: Seed
        verbose: Afficher les logs
    
    Returns:
        Tuple (modèle entraîné, history)
    """
    if verbose:
        print("\n" + "="*60)
        print("ENTRAÎNEMENT NEURAL NETWORK SIMPLE")
        print("="*60)
        print(f"Architecture: 128 → 64 → 32 → 1")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Validation split: {validation_split}")
    
    # Calculer les poids de classe
    class_weights = calculate_class_weights(y_train)
    if verbose:
        print(f"Class weights: {class_weights}")
    
    # Créer le modèle
    input_dim = X_train.shape[1]
    model = create_simple_nn(input_dim=input_dim, random_state=random_state)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            verbose=1 if verbose else 0,
            mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='max',
            verbose=1 if verbose else 0
        )
    ]
    
    # Entraînement avec class weights
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2 if verbose else 0
    )
    
    if verbose:
        final_epoch = len(history.history['loss'])
        final_auc = history.history['auc'][-1]
        final_val_auc = history.history['val_auc'][-1]
        print(f"\n✓ Entraînement terminé après {final_epoch} époques")
        print(f"  Train AUC: {final_auc:.4f}")
        print(f"  Val AUC: {final_val_auc:.4f}")
    
    return model, history


def train_advanced_nn(X_train, y_train,
                      epochs: int = 100,
                      batch_size: int = 32,
                      validation_split: float = 0.2,
                      random_state: int = 42,
                      verbose: bool = True):
    """
    Entraîne le Neural Network Avancé.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        epochs: Nombre d'époques
        batch_size: Taille des batches
        validation_split: Fraction pour validation
        random_state: Seed
        verbose: Afficher les logs
    
    Returns:
        Tuple (modèle entraîné, history)
    """
    if verbose:
        print("\n" + "="*60)
        print("ENTRAÎNEMENT NEURAL NETWORK AVANCÉ")
        print("="*60)
        print(f"Architecture: 128 → 64 → 32 → 16 → 1")
        print(f"BatchNorm + L2 regularization + LR scheduling")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    # Calculer les poids de classe
    class_weights = calculate_class_weights(y_train)
    if verbose:
        print(f"Class weights: {class_weights}")
    
    # Créer le modèle
    input_dim = X_train.shape[1]
    model = create_advanced_nn(input_dim=input_dim, random_state=random_state)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            verbose=1 if verbose else 0,
            mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            mode='max',
            verbose=1 if verbose else 0
        )
    ]
    
    # Entraînement avec class weights
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2 if verbose else 0
    )
    
    if verbose:
        final_epoch = len(history.history['loss'])
        final_auc = history.history['auc'][-1]
        final_val_auc = history.history['val_auc'][-1]
        print(f"\n✓ Entraînement terminé après {final_epoch} époques")
        print(f"  Train AUC: {final_auc:.4f}")
        print(f"  Val AUC: {final_val_auc:.4f}")
    
    return model, history


def save_models(xgb_model, nn_simple, nn_advanced, 
                nn_simple_history, nn_advanced_history,
                output_dir: str = None):
    """
    Sauvegarde tous les modèles entraînés.
    
    Args:
        xgb_model: Modèle XGBoost
        nn_simple: NN Simple
        nn_advanced: NN Avancé
        nn_simple_history: History du NN simple
        nn_advanced_history: History du NN avancé
        output_dir: Répertoire de sortie
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "models"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder XGBoost
    with open(output_dir / "xgboost_model.pkl", 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"✓ XGBoost sauvegardé: {output_dir / 'xgboost_model.pkl'}")
    
    # Sauvegarder NN Simple
    nn_simple.save(output_dir / "nn_simple.keras")
    print(f"✓ NN Simple sauvegardé: {output_dir / 'nn_simple.keras'}")
    
    # Sauvegarder NN Avancé
    nn_advanced.save(output_dir / "nn_advanced.keras")
    print(f"✓ NN Avancé sauvegardé: {output_dir / 'nn_advanced.keras'}")
    
    # Sauvegarder les histories
    histories = {
        'nn_simple': nn_simple_history.history,
        'nn_advanced': nn_advanced_history.history
    }
    with open(output_dir / "training_histories.pkl", 'wb') as f:
        pickle.dump(histories, f)


def train_all_models(X_train, y_train, 
                     xgb_params: dict = None,
                     nn_simple_params: dict = None,
                     nn_advanced_params: dict = None,
                     save: bool = True,
                     output_dir: str = None,
                     random_state: int = 42):
    """
    Entraîne les 3 modèles et les sauvegarde.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        xgb_params: Paramètres pour XGBoost
        nn_simple_params: Paramètres pour NN simple
        nn_advanced_params: Paramètres pour NN avancé
        save: Sauvegarder les modèles
        output_dir: Répertoire de sortie
        random_state: Seed
    
    Returns:
        Dictionnaire avec les modèles et leurs histories
    """
    results = {}
    
    # 1. XGBoost
    xgb_params = xgb_params or {}
    xgb_model = train_xgboost(X_train, y_train, random_state=random_state, **xgb_params)
    results['xgboost'] = {'model': xgb_model, 'history': None}
    
    # 2. NN Simple
    nn_simple_params = nn_simple_params or {}
    nn_simple, history_simple = train_simple_nn(
        X_train, y_train, random_state=random_state, **nn_simple_params
    )
    results['nn_simple'] = {'model': nn_simple, 'history': history_simple}
    
    # 3. NN Avancé
    nn_advanced_params = nn_advanced_params or {}
    nn_advanced, history_advanced = train_advanced_nn(
        X_train, y_train, random_state=random_state, **nn_advanced_params
    )
    results['nn_advanced'] = {'model': nn_advanced, 'history': history_advanced}
    
    # Sauvegarder
    if save:
        save_models(xgb_model, nn_simple, nn_advanced, 
                    history_simple, history_advanced, output_dir)
    
    return results


if __name__ == "__main__":
    # Test avec données fictives
    print("Test de l'entraînement avec données fictives...")
    
    X_dummy = np.random.randn(1000, 7)
    y_dummy = np.random.randint(0, 2, 1000)
    
    results = train_all_models(
        X_dummy, y_dummy,
        nn_simple_params={'epochs': 5, 'batch_size': 32},
        nn_advanced_params={'epochs': 5, 'batch_size': 32},
        save=False
    )
    
    print("\n✓ Test d'entraînement réussi!")
