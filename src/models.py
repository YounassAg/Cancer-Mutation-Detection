"""
Models Module - Définition des 2 architectures ML (XGBoost + Neural Network Simple)
"""
import numpy as np
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def create_xgboost_model(random_state: int = 42, n_jobs: int = -1) -> XGBClassifier:
    """
    Crée le modèle XGBoost (Baseline).
    
    Architecture optimisée pour le dataset ClinVar.
    
    Args:
        random_state: Seed pour reproductibilité
        n_jobs: Nombre de threads (-1 = tous les coeurs)
    
    Returns:
        Modèle XGBClassifier
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric='auc'
    )
    return model


def create_simple_nn(input_dim: int = 7, random_state: int = 42) -> keras.Model:
    """
    Crée le Neural Network Simple (MLP) avec 2 couches cachées.
    
    Architecture:
    - Input: 7 features
    - Hidden Layer 1: 64 neurones + ReLU + L2(0.001) + Dropout(0.3)
    - Hidden Layer 2: 32 neurones + ReLU + L2(0.001) + Dropout(0.2)
    - Output: 1 neurone + Sigmoid
    
    Args:
        input_dim: Nombre de features d'entrée
        random_state: Seed pour reproductibilité
    
    Returns:
        Modèle Keras compilé
    """
    tf.random.set_seed(random_state)
    np.random.seed(random_state)
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(64, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ], name='NN_Simple')
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def get_model_summary(model, model_name: str = "Model"):
    """
    Affiche un résumé du modèle.
    
    Args:
        model: Modèle à résumer
        model_name: Nom du modèle
    """
    print(f"\n{'='*60}")
    print(f"ARCHITECTURE: {model_name}")
    print(f"{'='*60}")
    model.summary()


if __name__ == "__main__":
    # Test des modèles
    print("Test des 2 architectures...")
    
    # 1. XGBoost
    xgb = create_xgboost_model()
    print(f"\n✓ XGBoost créé: {xgb}")
    
    # 2. NN Simple
    nn_simple = create_simple_nn(input_dim=7)
    get_model_summary(nn_simple, "Neural Network Simple")
