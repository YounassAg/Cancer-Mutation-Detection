"""
Train Optimized Module - Entraînement optimisé des 2 modèles avec GridSearch et tuning

Modèle 1: XGBoost avec GridSearchCV (5-fold)
Modèle 2: Neural Network Simple avec Early Stopping + ReduceLROnPlateau
"""
import numpy as np
import pickle
import time
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
import tensorflow as tf

from models import create_simple_nn


# ============================================================================
# UTILITAIRES
# ============================================================================

def calculate_class_weights(y_train):
    """Calcule les poids de classe pour gérer le déséquilibre."""
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


# ============================================================================
# XGBOOST AVEC GRIDSEARCHCV
# ============================================================================

def train_xgboost_optimized(X_train, y_train, random_state: int = 42, verbose: bool = True):
    """
    Entraîne XGBoost avec GridSearchCV pour optimiser les hyperparamètres.
    
    GridSearch sur :
      - max_depth: [4, 6, 8]
      - learning_rate: [0.01, 0.05, 0.1]
      - n_estimators: [100, 200, 300]
    
    Utilise scale_pos_weight pour le déséquilibre de classes.
    Cross-validation 5-fold.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        random_state: Seed
        verbose: Afficher les logs
    
    Returns:
        Tuple (meilleur modèle, résultats GridSearch)
    """
    from xgboost import XGBClassifier
    
    if verbose:
        print("\n" + "="*60)
        print("ENTRAÎNEMENT XGBOOST OPTIMISÉ (GridSearchCV)")
        print("="*60)
    
    # Calculer scale_pos_weight pour gérer le déséquilibre
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    if verbose:
        print(f"Distribution des classes: {n_neg} négatifs, {n_pos} positifs")
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Modèle de base avec scale_pos_weight
    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='auc'
    )
    
    # Grille de recherche
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300]
    }
    
    if verbose:
        total_combos = 1
        for v in param_grid.values():
            total_combos *= len(v)
        print(f"\nGrille de recherche: {total_combos} combinaisons × 5 folds = {total_combos * 5} entraînements")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
    
    # GridSearchCV avec 5-fold
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=1 if verbose else 0,
        n_jobs=-1,
        return_train_score=True
    )
    
    if verbose:
        print(f"\nDébut du GridSearchCV...")
        start_time = time.time()
    
    grid_search.fit(X_train, y_train)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\n✓ GridSearchCV terminé en {elapsed:.1f}s")
        print(f"\n{'─'*40}")
        print(f"MEILLEURS HYPERPARAMÈTRES XGBOOST:")
        print(f"{'─'*40}")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\n  Meilleur ROC-AUC (CV): {grid_search.best_score_:.4f}")
        print(f"{'─'*40}")
    
    best_model = grid_search.best_estimator_
    
    # Résultats détaillés
    grid_results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    return best_model, grid_results


# ============================================================================
# NEURAL NETWORK SIMPLE OPTIMISÉ
# ============================================================================

def train_nn_simple_optimized(X_train, y_train,
                               batch_sizes_to_test: list = None,
                               epochs: int = 150,
                               validation_split: float = 0.2,
                               random_state: int = 42,
                               verbose: bool = True):
    """
    Entraîne le NN Simple optimisé avec Early Stopping et ReduceLROnPlateau.
    
    Teste différentes valeurs de batch_size et garde le meilleur.
    
    Améliorations:
      - Early Stopping patience=15
      - ReduceLROnPlateau
      - Test batch_size: [16, 32, 64]
      - epochs: 150
      - L2 regularization (0.001)
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        batch_sizes_to_test: Liste de batch_size à tester (défaut: [16, 32, 64])
        epochs: Nombre max d'époques
        validation_split: Fraction pour validation
        random_state: Seed
        verbose: Afficher les logs
    
    Returns:
        Tuple (meilleur modèle, meilleur history, résultats de tous les tests)
    """
    if batch_sizes_to_test is None:
        batch_sizes_to_test = [16, 32, 64]
    
    if verbose:
        print("\n" + "="*60)
        print("ENTRAÎNEMENT NEURAL NETWORK SIMPLE OPTIMISÉ")
        print("="*60)
        print(f"Architecture: 64 → 32 → 1 (avec L2=0.001)")
        print(f"Epochs max: {epochs}")
        print(f"Batch sizes à tester: {batch_sizes_to_test}")
        print(f"Early Stopping patience: 15")
        print(f"ReduceLROnPlateau: factor=0.5, patience=5")
    
    # Calculer les poids de classe
    class_weights = calculate_class_weights(y_train)
    if verbose:
        print(f"Class weights: {class_weights}")
    
    input_dim = X_train.shape[1]
    
    best_model = None
    best_history = None
    best_val_auc = -1
    best_batch_size = None
    all_results = {}
    
    for batch_size in batch_sizes_to_test:
        if verbose:
            print(f"\n{'─'*40}")
            print(f"Test avec batch_size = {batch_size}")
            print(f"{'─'*40}")
        
        # Recréer le modèle à chaque test
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        model = create_simple_nn(input_dim=input_dim, random_state=random_state)
        
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
                patience=5,
                min_lr=1e-6,
                mode='max',
                verbose=1 if verbose else 0
            )
        ]
        
        # Entraînement
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2 if verbose else 0
        )
        
        # Récupérer le meilleur val_auc
        val_auc = max(history.history['val_auc'])
        final_epoch = len(history.history['loss'])
        
        if verbose:
            print(f"  → batch_size={batch_size}: val_AUC={val_auc:.4f} (arrêté epoch {final_epoch})")
        
        all_results[batch_size] = {
            'val_auc': val_auc,
            'epochs_run': final_epoch,
            'history': history.history
        }
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model
            best_history = history
            best_batch_size = batch_size
    
    if verbose:
        print(f"\n{'─'*40}")
        print(f"MEILLEURS HYPERPARAMÈTRES NN SIMPLE:")
        print(f"{'─'*40}")
        print(f"  batch_size: {best_batch_size}")
        print(f"  epochs effectifs: {len(best_history.history['loss'])}")
        print(f"  L2 regularization: 0.001")
        print(f"  Early Stopping patience: 15")
        print(f"  Meilleur val_AUC: {best_val_auc:.4f}")
        print(f"{'─'*40}")
        
        # Résumé des tests
        print(f"\nRésumé des tests batch_size:")
        for bs, res in all_results.items():
            marker = " ← BEST" if bs == best_batch_size else ""
            print(f"  batch_size={bs}: val_AUC={res['val_auc']:.4f}, epochs={res['epochs_run']}{marker}")
    
    nn_results = {
        'best_batch_size': best_batch_size,
        'best_val_auc': best_val_auc,
        'all_batch_results': all_results
    }
    
    return best_model, best_history, nn_results


# ============================================================================
# SAUVEGARDE
# ============================================================================

def save_optimized_models(xgb_model, nn_simple, nn_simple_history,
                          xgb_grid_results, nn_results,
                          output_dir: str = None):
    """
    Sauvegarde les 2 modèles optimisés et leurs résultats.
    
    Args:
        xgb_model: Meilleur modèle XGBoost
        nn_simple: Meilleur NN Simple
        nn_simple_history: History du meilleur NN
        xgb_grid_results: Résultats GridSearchCV
        nn_results: Résultats des tests NN
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
    
    # Sauvegarder les histories (NN Simple uniquement)
    histories = {
        'nn_simple': nn_simple_history.history
    }
    with open(output_dir / "training_histories.pkl", 'wb') as f:
        pickle.dump(histories, f)
    print(f"✓ Historiques sauvegardés: {output_dir / 'training_histories.pkl'}")
    
    # Sauvegarder les résultats d'optimisation
    optim_results = {
        'xgboost': {
            'best_params': xgb_grid_results['best_params'],
            'best_cv_score': float(xgb_grid_results['best_score'])
        },
        'nn_simple': {
            'best_batch_size': nn_results['best_batch_size'],
            'best_val_auc': float(nn_results['best_val_auc'])
        }
    }
    with open(output_dir / "optimization_results.pkl", 'wb') as f:
        pickle.dump(optim_results, f)
    print(f"✓ Résultats optimisation sauvegardés: {output_dir / 'optimization_results.pkl'}")


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def train_all_optimized(X_train, y_train,
                         random_state: int = 42,
                         save: bool = True,
                         output_dir: str = None):
    """
    Entraîne les 2 modèles optimisés et les sauvegarde.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        random_state: Seed
        save: Sauvegarder les modèles
        output_dir: Répertoire de sortie
    
    Returns:
        Dictionnaire avec les modèles et résultats d'optimisation
    """
    results = {}
    
    # 1. XGBoost avec GridSearchCV
    xgb_model, xgb_grid_results = train_xgboost_optimized(
        X_train, y_train, random_state=random_state
    )
    results['xgboost'] = {
        'model': xgb_model,
        'grid_results': xgb_grid_results
    }
    
    # 2. NN Simple optimisé
    nn_simple, nn_history, nn_results = train_nn_simple_optimized(
        X_train, y_train, random_state=random_state
    )
    results['nn_simple'] = {
        'model': nn_simple,
        'history': nn_history,
        'tuning_results': nn_results
    }
    
    # Sauvegarder
    if save:
        save_optimized_models(
            xgb_model, nn_simple, nn_history,
            xgb_grid_results, nn_results,
            output_dir
        )
    
    return results


if __name__ == "__main__":
    # Test avec données fictives
    print("Test de l'entraînement optimisé avec données fictives...")
    
    X_dummy = np.random.randn(1000, 7)
    y_dummy = np.random.randint(0, 2, 1000)
    
    results = train_all_optimized(
        X_dummy, y_dummy,
        save=False
    )
    
    print("\n✓ Test d'entraînement optimisé réussi!")
