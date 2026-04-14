"""
Main Pipeline - Exécution complète du projet
Ce script exécute l'ensemble du pipeline ML:
1. Chargement des données
2. Prétraitement
3. Entraînement des 3 modèles
4. Évaluation et comparaison
5. Prédiction d'exemple
"""
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import des modules locaux
from data_loader import load_clinvar_data, explore_data, get_features_info
from preprocessing import clean_clinvar_data, prepare_train_test_data, save_preprocessed_data
from train import train_all_models
from evaluate import evaluate_all_models, evaluate_model, print_evaluation


def main(data_path: str = None, 
         nrows: int = None,
         test_size: float = 0.2,
         random_state: int = 42,
         save_models: bool = True,
         output_dir: str = None):
    """
    Pipeline complet d'entraînement et d'évaluation.
    
    Args:
        data_path: Chemin vers le CSV ClinVar
        nrows: Nombre de lignes à charger (None = toutes)
        test_size: Fraction pour le test
        random_state: Seed pour reproductibilité
        save_models: Sauvegarder les modèles entraînés
        output_dir: Répertoire de sortie
    
    Returns:
        Dictionnaire avec les résultats
    """
    print("\n" + "="*70)
    print("CANCER MUTATION DETECTION - PIPELINE ML COMPLET")
    print("Prédiction de mutations génétiques pathogènes (Cancer)")
    print("="*70)
    
    # =========================================================================
    # ÉTAPE 1: CHARGEMENT DES DONNÉES
    # =========================================================================
    print("\n📥 ÉTAPE 1: CHARGEMENT DES DONNÉES")
    print("-" * 60)
    
    df = load_clinvar_data(data_path, nrows=nrows)
    explore_data(df)
    
    # =========================================================================
    # ÉTAPE 2: PRÉTRAITEMENT
    # =========================================================================
    print("\n🔧 ÉTAPE 2: PRÉTRAITEMENT")
    print("-" * 60)
    
    df_clean, features = clean_clinvar_data(df)
    
    if len(df_clean) == 0:
        print("❌ ERREUR: Pas assez de données après nettoyage!")
        return None
    
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test_data(
        df_clean, test_size=test_size, random_state=random_state
    )
    
    # Sauvegarder les données prétraitées
    if save_models:
        save_preprocessed_data(X_train, X_test, y_train, y_test, 
                               scaler, feature_names, output_dir)
    
    # =========================================================================
    # ÉTAPE 3: ENTRAÎNEMENT
    # =========================================================================
    print("\n🚀 ÉTAPE 3: ENTRAÎNEMENT DES 3 MODÈLES")
    print("-" * 60)
    
    # Paramètres d'entraînement
    xgb_params = {}
    nn_simple_params = {'epochs': 50, 'batch_size': 32, 'validation_split': 0.2}
    nn_advanced_params = {'epochs': 100, 'batch_size': 32, 'validation_split': 0.2}
    
    # Entraîner tous les modèles
    training_results = train_all_models(
        X_train, y_train,
        xgb_params=xgb_params,
        nn_simple_params=nn_simple_params,
        nn_advanced_params=nn_advanced_params,
        save=save_models,
        output_dir=output_dir,
        random_state=random_state
    )
    
    xgb_model = training_results['xgboost']['model']
    nn_simple = training_results['nn_simple']['model']
    nn_advanced = training_results['nn_advanced']['model']
    nn_simple_history = training_results['nn_simple']['history']
    nn_advanced_history = training_results['nn_advanced']['history']
    
    # =========================================================================
    # ÉTAPE 4: ÉVALUATION
    # =========================================================================
    print("\n📊 ÉTAPE 4: ÉVALUATION ET COMPARAISON")
    print("-" * 60)
    
    eval_results = evaluate_all_models(
        xgb_model, nn_simple, nn_advanced,
        X_test, y_test, feature_names,
        nn_simple_history.history if nn_simple_history else None,
        nn_advanced_history.history if nn_advanced_history else None,
        output_dir=output_dir
    )
    
    # =========================================================================
    # ÉTAPE 5: PRÉDICTION D'EXEMPLE
    # =========================================================================
    print("\n🔮 ÉTAPE 5: PRÉDICTION SUR UN CAS RÉEL (SIMULÉ)")
    print("-" * 60)
    
    # Exemple de mutation pathogène
    example_pathogenic = np.array([[
        28.0,      # CADD_PHRED (élevé = pathogène)
        3.5,       # CADD_RAW
        0.01,      # SIFT (bas = délétère)
        0.95,      # PolyPhen (élevé = dangereux)
        0.0001,    # AF_EXAC (rare)
        0.0002,    # AF_TGP (rare)
        -3.0       # BLOSUM62 (négatif = grave)
    ]])
    
    # Exemple de mutation bénigne
    example_benign = np.array([[
        5.0,       # CADD_PHRED (faible)
        0.5,       # CADD_RAW
        0.8,       # SIFT (élevé = toléré)
        0.1,       # PolyPhen (faible)
        0.35,      # AF_EXAC (commun)
        0.40,      # AF_TGP (commun)
        2.0        # BLOSUM62 (positif)
    ]])
    
    # Normaliser les exemples
    example_pathogenic_scaled = scaler.transform(example_pathogenic)
    example_benign_scaled = scaler.transform(example_benign)
    
    # Prédictions
    for name, example, example_scaled in [
        ("MUTATION PATHOGÈNE (Cancer)", example_pathogenic, example_pathogenic_scaled),
        ("MUTATION BÉNIGNE (Inoffensive)", example_benign, example_benign_scaled)
    ]:
        print(f"\n{name}:")
        print(f"  Features: CADD_PHRED={example[0][0]:.1f}, SIFT={example[0][2]:.2f}, PolyPhen={example[0][3]:.2f}")
        
        pred_xgb = xgb_model.predict_proba(example_scaled)[0, 1]
        pred_nn_simple = nn_simple.predict(example_scaled, verbose=0)[0, 0]
        pred_nn_advanced = nn_advanced.predict(example_scaled, verbose=0)[0, 0]
        pred_avg = (pred_xgb + pred_nn_simple + pred_nn_advanced) / 3
        
        print(f"  Probabilité Pathogène:")
        print(f"    XGBoost:    {pred_xgb:.1%}")
        print(f"    NN Simple:  {pred_nn_simple:.1%}")
        print(f"    NN Avancé:  {pred_nn_advanced:.1%}")
        print(f"    Moyenne:    {pred_avg:.1%}")
        
        verdict = "PATHOGÈNE ⚠️" if pred_avg > 0.5 else "BÉNIGNE ✓"
        print(f"  → VERDICT: {verdict}")
    
    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    print("\n" + "="*70)
    print("RÉSUMÉ FINAL")
    print("="*70)
    
    # Trouver le meilleur modèle
    roc_aucs = [r['roc_auc'] for r in eval_results]
    best_idx = np.argmax(roc_aucs)
    best_model = eval_results[best_idx]
    
    print(f"\n📈 Meilleur modèle: {best_model['model_name']}")
    print(f"   ROC-AUC: {best_model['roc_auc']:.4f}")
    
    # Vérifier si objectif atteint (80%+ ROC-AUC)
    if best_model['roc_auc'] >= 0.80:
        print(f"\n✅ OBJECTIF ATTEINT: ROC-AUC ≥ 80% ({best_model['roc_auc']:.1%})")
    else:
        print(f"\n⚠️  Objectif non atteint: ROC-AUC < 80% ({best_model['roc_auc']:.1%})")
    
    print(f"\n📁 Résultats sauvegardés dans: {output_dir or 'models/ et results/'}")
    print("="*70)
    
    return {
        'models': {
            'xgboost': xgb_model,
            'nn_simple': nn_simple,
            'nn_advanced': nn_advanced
        },
        'results': eval_results,
        'scaler': scaler,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    # Parser les arguments
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cancer Mutation Detection - Pipeline ML'
    )
    parser.add_argument('--data', type=str, default=None,
                       help='Chemin vers le fichier CSV ClinVar')
    parser.add_argument('--nrows', type=int, default=None,
                       help='Nombre de lignes à charger (défaut: toutes)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction pour le test (défaut: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed aléatoire (défaut: 42)')
    parser.add_argument('--output', type=str, default=None,
                       help='Répertoire de sortie')
    parser.add_argument('--no-save', action='store_true',
                       help='Ne pas sauvegarder les modèles')
    
    args = parser.parse_args()
    
    # Exécuter le pipeline
    results = main(
        data_path=args.data,
        nrows=args.nrows,
        test_size=args.test_size,
        random_state=args.seed,
        save_models=not args.no_save,
        output_dir=args.output
    )
    
    if results is None:
        sys.exit(1)
