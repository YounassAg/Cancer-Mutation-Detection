"""
Evaluate Optimized Module - Évaluation et comparaison des 2 modèles optimisés
(XGBoost vs Neural Network Simple)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score,
    f1_score, confusion_matrix, precision_recall_curve, auc as pr_auc,
    accuracy_score
)
import json


# ============================================================================
# ÉVALUATION D'UN MODÈLE
# ============================================================================

def evaluate_model(y_true, y_pred_proba, model_name: str, threshold: float = 0.5):
    """
    Évalue un modèle et calcule toutes les métriques.
    
    Args:
        y_true: Cibles réelles
        y_pred_proba: Probabilités prédites
        model_name: Nom du modèle
        threshold: Seuil de classification
    
    Returns:
        Dictionnaire avec les métriques
    """
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc_score = pr_auc(recall_vals, precision_vals)
    
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def print_evaluation(results: dict):
    """Affiche les résultats d'évaluation formatés."""
    print(f"\n{'='*60}")
    print(f"RÉSULTATS: {results['model_name']}")
    print(f"{'='*60}")
    print(f"ROC-AUC              : {results['roc_auc']:.4f}")
    print(f"Precision-Recall AUC : {results['pr_auc']:.4f}")
    print(f"Accuracy             : {results['accuracy']:.4f}")
    print(f"Precision            : {results['precision']:.4f}")
    print(f"Recall (Sensibilité) : {results['recall']:.4f}")
    print(f"F1-Score             : {results['f1_score']:.4f}")
    print(f"Specificity          : {results['specificity']:.4f}")
    print(f"\nMatrice de Confusion:")
    print(f"  True Negatives  (TN): {results['tn']:6d}  (correctement Benign)")
    print(f"  False Positives (FP): {results['fp']:6d}  (faux Pathogenic)")
    print(f"  False Negatives (FN): {results['fn']:6d}  (faux Benign)")
    print(f"  True Positives  (TP): {results['tp']:6d}  (correctement Pathogenic)")
    print(f"{'='*60}")


# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_roc_curves(results_list: list, y_true, output_path: str = None):
    """
    Trace les courbes ROC pour les 2 modèles.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e']
    
    for idx, results in enumerate(results_list):
        y_pred = results['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = results['roc_auc']
        
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], linewidth=2,
                label=f"{results['model_name']} (AUC = {roc_auc:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('Courbes ROC - XGBoost vs NN Simple', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Courbes ROC sauvegardées: {output_path}")
    
    plt.close(fig)
    return fig


def plot_confusion_matrices(results_list: list, output_path: str = None):
    """
    Trace les matrices de confusion pour les 2 modèles.
    """
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (results, ax) in enumerate(zip(results_list, axes)):
        cm = results['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Benign (0)', 'Pathogenic (1)'],
                    yticklabels=['Benign (0)', 'Pathogenic (1)'],
                    cbar_kws={'shrink': 0.8})
        ax.set_title(results['model_name'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Vrai Label', fontsize=11)
        ax.set_xlabel('Prédit', fontsize=11)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matrices de confusion sauvegardées: {output_path}")
    
    plt.close(fig)
    return fig


def plot_feature_importance(xgb_model, feature_names: list, output_path: str = None):
    """
    Trace l'importance des features pour XGBoost.
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))
    ax.barh(importance['feature'], importance['importance'], color=colors)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (XGBoost - Meilleur GridSearch)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance sauvegardée: {output_path}")
    
    plt.close(fig)
    return fig


def plot_training_history(nn_simple_history, output_path: str = None):
    """
    Trace l'historique d'entraînement du NN Simple uniquement.
    
    Args:
        nn_simple_history: History dict du NN simple
        output_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(nn_simple_history['loss'], label='Train Loss', linewidth=2, color='#1f77b4')
    axes[0].plot(nn_simple_history['val_loss'], label='Val Loss', linewidth=2, 
                 linestyle='--', color='#ff7f0e')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Binary Crossentropy Loss', fontsize=12)
    axes[0].set_title('Évolution de la Perte (NN Simple)', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC
    axes[1].plot(nn_simple_history['auc'], label='Train AUC', linewidth=2, color='#1f77b4')
    axes[1].plot(nn_simple_history['val_auc'], label='Val AUC', linewidth=2, 
                 linestyle='--', color='#ff7f0e')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title('Évolution de l\'AUC (NN Simple)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Historique d'entraînement sauvegardé: {output_path}")
    
    plt.close(fig)
    return fig


# ============================================================================
# COMPARAISON ET SAUVEGARDE
# ============================================================================

def create_comparison_table(results_list: list, output_path: str = None):
    """
    Crée un tableau comparatif XGBoost vs NN Simple.
    """
    comparison = pd.DataFrame([
        {
            'Modèle': r['model_name'],
            'ROC-AUC': f"{r['roc_auc']:.4f}",
            'PR-AUC': f"{r['pr_auc']:.4f}",
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1_score']:.4f}",
            'Specificity': f"{r['specificity']:.4f}",
            'Sensitivity': f"{r['sensitivity']:.4f}"
        }
        for r in results_list
    ])
    
    roc_aucs = [r['roc_auc'] for r in results_list]
    best_idx = np.argmax(roc_aucs)
    
    print("\n" + "="*80)
    print("COMPARAISON FINALE : XGBoost vs NN Simple")
    print("="*80)
    print(comparison.to_string(index=False))
    print(f"\n🏆 Meilleur modèle (ROC-AUC): {results_list[best_idx]['model_name']}")
    print(f"   ROC-AUC: {results_list[best_idx]['roc_auc']:.4f}")
    print("="*80)
    
    if output_path:
        comparison.to_csv(output_path, index=False)
        print(f"✓ Tableau comparatif sauvegardé: {output_path}")
    
    return comparison


def print_best_hyperparameters(xgb_grid_results, nn_results):
    """
    Affiche un résumé des meilleurs hyperparamètres trouvés.
    """
    print("\n" + "="*60)
    print("RÉSUMÉ DES MEILLEURS HYPERPARAMÈTRES")
    print("="*60)
    
    print(f"\n📊 XGBoost (GridSearchCV 5-fold):")
    for param, value in xgb_grid_results['best_params'].items():
        print(f"   {param}: {value}")
    print(f"   Meilleur ROC-AUC (CV): {xgb_grid_results['best_score']:.4f}")
    
    print(f"\n🧠 Neural Network Simple:")
    print(f"   batch_size: {nn_results['best_batch_size']}")
    print(f"   L2 regularization: 0.001")
    print(f"   Early Stopping patience: 15")
    print(f"   ReduceLROnPlateau: factor=0.5, patience=5")
    print(f"   Meilleur val_AUC: {nn_results['best_val_auc']:.4f}")
    print("="*60)


def save_metrics_json(results_list: list, xgb_grid_results: dict, 
                      nn_results: dict, output_path: str):
    """
    Sauvegarde toutes les métriques et hyperparamètres en JSON.
    """
    metrics = {}
    for r in results_list:
        metrics[r['model_name']] = {
            'roc_auc': float(r['roc_auc']),
            'pr_auc': float(r['pr_auc']),
            'accuracy': float(r['accuracy']),
            'precision': float(r['precision']),
            'recall': float(r['recall']),
            'f1_score': float(r['f1_score']),
            'specificity': float(r['specificity']),
            'sensitivity': float(r['sensitivity']),
            'confusion_matrix': r['confusion_matrix'].tolist()
        }
    
    # Ajouter les meilleurs hyperparamètres
    metrics['best_hyperparameters'] = {
        'xgboost': {
            k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in xgb_grid_results['best_params'].items()
        },
        'xgboost_best_cv_score': float(xgb_grid_results['best_score']),
        'nn_simple': {
            'batch_size': int(nn_results['best_batch_size']),
            'l2_regularization': 0.001,
            'early_stopping_patience': 15,
            'best_val_auc': float(nn_results['best_val_auc'])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Métriques JSON sauvegardées: {output_path}")


# ============================================================================
# PIPELINE D'ÉVALUATION COMPLET
# ============================================================================

def evaluate_all_optimized(xgb_model, nn_simple,
                            X_test, y_test, feature_names,
                            nn_simple_history=None,
                            xgb_grid_results=None,
                            nn_results=None,
                            output_dir: str = None):
    """
    Évalue les 2 modèles optimisés et génère toutes les visualisations.
    
    Args:
        xgb_model: Modèle XGBoost (meilleur de GridSearch)
        nn_simple: Modèle NN Simple (meilleur batch_size)
        X_test: Features de test
        y_test: Cibles de test
        feature_names: Noms des features
        nn_simple_history: History du NN simple
        xgb_grid_results: Résultats GridSearchCV
        nn_results: Résultats tuning NN
        output_dir: Répertoire de sortie
    
    Returns:
        Liste des résultats d'évaluation
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("ÉVALUATION DES 2 MODÈLES OPTIMISÉS")
    print("="*60)
    
    # Prédictions
    y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
    y_pred_nn_simple = nn_simple.predict(X_test, verbose=0)[:, 0]
    
    # Évaluer chaque modèle
    results_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost")
    results_nn_simple = evaluate_model(y_test, y_pred_nn_simple, "NN Simple")
    
    results_list = [results_xgb, results_nn_simple]
    
    # Afficher les résultats
    for r in results_list:
        print_evaluation(r)
    
    # Afficher les meilleurs hyperparamètres
    if xgb_grid_results and nn_results:
        print_best_hyperparameters(xgb_grid_results, nn_results)
    
    # Générer les visualisations
    print("\n" + "="*60)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*60)
    
    # 1. Courbes ROC
    plot_roc_curves(results_list, y_test, output_dir / "roc_curves.png")
    
    # 2. Matrices de confusion
    plot_confusion_matrices(results_list, output_dir / "confusion_matrices.png")
    
    # 3. Feature importance
    plot_feature_importance(xgb_model, feature_names, output_dir / "feature_importance.png")
    
    # 4. Training history (NN Simple uniquement)
    if nn_simple_history:
        history_data = nn_simple_history if isinstance(nn_simple_history, dict) else nn_simple_history.history
        plot_training_history(history_data, output_dir / "training_history.png")
    
    # 5. Comparaison finale
    comparison = create_comparison_table(results_list, output_dir / "model_comparison.csv")
    
    # 6. Métriques JSON
    save_metrics_json(
        results_list, 
        xgb_grid_results or {'best_params': {}, 'best_score': 0},
        nn_results or {'best_batch_size': 32, 'best_val_auc': 0},
        output_dir / "metrics.json"
    )
    
    print(f"\n✓ Tous les résultats sauvegardés dans: {output_dir}")
    
    return results_list


if __name__ == "__main__":
    # Test avec données fictives
    print("Test d'évaluation avec données fictives...")
    
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.rand(1000)
    
    results = evaluate_model(y_true, y_pred, "Test Model")
    print_evaluation(results)
    
    print("\n✓ Test d'évaluation réussi!")
