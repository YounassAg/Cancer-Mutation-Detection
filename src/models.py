"""
Models Module - Définition des 3 architectures ML
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
    Crée le Neural Network Simple (MLP) optimisé.
    
    Architecture:
    - Input: 7 features
    - Hidden Layer 1: 128 neurones + ReLU + Dropout(0.3)
    - Hidden Layer 2: 64 neurones + ReLU + Dropout(0.2)
    - Hidden Layer 3: 32 neurones + ReLU + Dropout(0.1)
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
        
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
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


def create_advanced_nn(input_dim: int = 7, random_state: int = 42) -> keras.Model:
    """
    Crée le Neural Network Avancé avec Batch Normalization et L2.
    
    Architecture selon CONTEXT.md:
    - Input: 7 features
    - Dense(128) + ReLU + BatchNorm + Dropout(0.3) + L2(0.001)
    - Dense(64) + ReLU + BatchNorm + Dropout(0.3) + L2(0.001)
    - Dense(32) + ReLU + BatchNorm + Dropout(0.2) + L2(0.001)
    - Dense(16) + ReLU + Dropout(0.2)
    - Output: 1 neurone + Sigmoid
    
    Args:
        input_dim: Nombre de features d'entrée
        random_state: Seed pour reproductibilité
    
    Returns:
        Modèle Keras compilé avec learning rate schedule
    """
    # Fixer le seed
    tf.random.set_seed(random_state)
    np.random.seed(random_state)
    
    # Learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001),
                     name='hidden_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(0.3, name='dropout_1'),
        
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001),
                     name='hidden_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(0.3, name='dropout_2'),
        
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001),
                     name='hidden_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(0.2, name='dropout_3'),
        
        layers.Dense(16, activation='relu', name='hidden_4'),
        layers.Dropout(0.2, name='dropout_4'),
        
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='NN_Advanced')
    
    # Compilation avec optimizer custom
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
    print("Test des 3 architectures...")
    
    # 1. XGBoost
    xgb = create_xgboost_model()
    print(f"\n✓ XGBoost créé: {xgb}")
    
    # 2. NN Simple
    nn_simple = create_simple_nn(input_dim=7)
    get_model_summary(nn_simple, "Neural Network Simple")
    
    # 3. NN Avancé
    nn_advanced = create_advanced_nn(input_dim=7)
    get_model_summary(nn_advanced, "Neural Network Avancé")
