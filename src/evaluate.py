"""
Evaluation Module - Évaluation et comparaison des modèles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score, recall_score,
    f1_score, confusion_matrix, precision_recall_curve, auc as pr_auc,
    accuracy_score, classification_report
)
import json


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
    # Convertir en binaire
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Calculer les métriques
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc_score = pr_auc(recall_vals, precision_vals)
    
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Spécificité et sensibilité
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    results = {
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
    
    return results


def print_evaluation(results: dict):
    """
    Affiche les résultats d'évaluation formatés.
    
    Args:
        results: Dictionnaire de résultats
    """
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


def plot_roc_curves(results_list: list, y_true, output_path: str = None):
    """
    Trace les courbes ROC pour tous les modèles.
    
    Args:
        results_list: Liste des résultats de modèles
        y_true: Cibles réelles
        output_path: Chemin pour sauvegarder la figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, results in enumerate(results_list):
        y_pred = results['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = results['roc_auc']
        
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], linewidth=2,
                label=f"{results['model_name']} (AUC = {roc_auc:.3f})")
    
    # Diagonale aléatoire
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Courbes ROC sauvegardées: {output_path}")
    
    return fig


def plot_confusion_matrices(results_list: list, output_path: str = None):
    """
    Trace les matrices de confusion pour tous les modèles.
    
    Args:
        results_list: Liste des résultats de modèles
        output_path: Chemin pour sauvegarder la figure
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
    
    return fig


def plot_feature_importance(xgb_model, feature_names: list, output_path: str = None):
    """
    Trace l'importance des features pour XGBoost.
    
    Args:
        xgb_model: Modèle XGBoost entraîné
        feature_names: Noms des features
        output_path: Chemin pour sauvegarder la figure
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))
    ax.barh(importance['feature'], importance['importance'], color=colors)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance sauvegardée: {output_path}")
    
    return fig


def plot_training_history(nn_simple_history, nn_advanced_history, output_path: str = None):
    """
    Trace l'historique d'entraînement des réseaux de neurones.
    
    Args:
        nn_simple_history: History du NN simple
        nn_advanced_history: History du NN avancé
        output_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(nn_simple_history['loss'], label='NN Simple - Train', linewidth=2)
    axes[0].plot(nn_simple_history['val_loss'], label='NN Simple - Val', linewidth=2, linestyle='--')
    axes[0].plot(nn_advanced_history['loss'], label='NN Avancé - Train', linewidth=2)
    axes[0].plot(nn_advanced_history['val_loss'], label='NN Avancé - Val', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Binary Crossentropy Loss', fontsize=12)
    axes[0].set_title('Évolution de la Perte', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC
    axes[1].plot(nn_simple_history['auc'], label='NN Simple - Train', linewidth=2)
    axes[1].plot(nn_simple_history['val_auc'], label='NN Simple - Val', linewidth=2, linestyle='--')
    axes[1].plot(nn_advanced_history['auc'], label='NN Avancé - Train', linewidth=2)
    axes[1].plot(nn_advanced_history['val_auc'], label='NN Avancé - Val', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title('Évolution de l\'AUC', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Historique d'entraînement sauvegardé: {output_path}")
    
    return fig


def create_comparison_table(results_list: list, output_path: str = None):
    """
    Crée un tableau comparatif des modèles.
    
    Args:
        results_list: Liste des résultats
        output_path: Chemin pour sauvegarder le CSV
    
    Returns:
        DataFrame comparatif
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
    
    # Identifier le meilleur modèle (ROC-AUC)
    roc_aucs = [r['roc_auc'] for r in results_list]
    best_idx = np.argmax(roc_aucs)
    
    print("\n" + "="*80)
    print("COMPARAISON DES MODÈLES")
    print("="*80)
    print(comparison.to_string(index=False))
    print(f"\n🏆 Meilleur modèle (ROC-AUC): {results_list[best_idx]['model_name']}")
    print(f"   ROC-AUC: {results_list[best_idx]['roc_auc']:.4f}")
    print("="*80)
    
    if output_path:
        comparison.to_csv(output_path, index=False)
        print(f"✓ Tableau comparatif sauvegardé: {output_path}")
    
    return comparison


def save_metrics_json(results_list: list, output_path: str):
    """
    Sauvegarde toutes les métriques en JSON.
    
    Args:
        results_list: Liste des résultats
        output_path: Chemin de sortie
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
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Métriques JSON sauvegardées: {output_path}")


def evaluate_all_models(xgb_model, nn_simple, nn_advanced,
                        X_test, y_test, feature_names,
                        nn_simple_history=None, nn_advanced_history=None,
                        output_dir: str = None):
    """
    Évalue tous les modèles et génère les visualisations.
    
    Args:
        xgb_model: Modèle XGBoost
        nn_simple: Modèle NN simple
        nn_advanced: Modèle NN avancé
        X_test: Features de test
        y_test: Cibles de test
        feature_names: Noms des features
        nn_simple_history: History NN simple
        nn_advanced_history: History NN avancé
        output_dir: Répertoire de sortie
    
    Returns:
        Liste des résultats
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("ÉVALUATION DES MODÈLES")
    print("="*60)
    
    # Prédictions
    y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
    y_pred_nn_simple = nn_simple.predict(X_test, verbose=0)[:, 0]
    y_pred_nn_advanced = nn_advanced.predict(X_test, verbose=0)[:, 0]
    
    # Évaluer chaque modèle
    results_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost")
    results_nn_simple = evaluate_model(y_test, y_pred_nn_simple, "NN Simple")
    results_nn_advanced = evaluate_model(y_test, y_pred_nn_advanced, "NN Avancé")
    
    results_list = [results_xgb, results_nn_simple, results_nn_advanced]
    
    # Afficher les résultats
    for r in results_list:
        print_evaluation(r)
    
    # Générer les visualisations
    print("\n" + "="*60)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*60)
    
    plot_roc_curves(results_list, y_test, output_dir / "roc_curves.png")
    plot_confusion_matrices(results_list, output_dir / "confusion_matrices.png")
    plot_feature_importance(xgb_model, feature_names, output_dir / "feature_importance.png")
    
    if nn_simple_history and nn_advanced_history:
        plot_training_history(nn_simple_history, nn_advanced_history, 
                            output_dir / "training_history.png")
    
    # Tableau comparatif
    comparison = create_comparison_table(results_list, output_dir / "model_comparison.csv")
    
    # Sauvegarder les métriques JSON
    save_metrics_json(results_list, output_dir / "metrics.json")
    
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
