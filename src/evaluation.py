import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from .config import TARGET_RECALL

def evaluate_medical_utility(model, X_test, y_test):
    """
    Evaluates the model with a focus on clinical safety.
    Optimizes the decision threshold to satisfy medical sensitivity requirements.
    """
    y_probs = model.predict(
        [X_test['gene'], X_test['type'], X_test['chrom'], X_test['origin'], X_test['numeric']]
    )
    
    # Calculate Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Identify optimal medical threshold for target recall
    valid_idx = np.where(recalls >= TARGET_RECALL)[0]
    if len(valid_idx) > 0:
        target_idx = valid_idx[-1]
        optimized_threshold = thresholds[target_idx]
    else:
        # Fallback: use the threshold that maximizes F1 score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        target_idx = np.argmax(f1_scores)
        optimized_threshold = thresholds[min(target_idx, len(thresholds) - 1)]
        print(f"⚠️  Could not reach target recall of {TARGET_RECALL:.0%}. Using best F1 threshold instead.")
    
    y_pred = (y_probs >= optimized_threshold).astype(int)
    
    target_names = ['Non-Oncogenic (0)', 'Oncogenic (1)']
    print(f"\n--- Clinical Performance Appraisal (Threshold: {optimized_threshold:.4f}) ---")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return y_pred, y_probs, optimized_threshold

def plot_visual_assessment(y_test, y_pred, y_probs):
    """
    Generates medical utility charts: Confusion Matrix and PR-Curve.
    """
    plt.figure(figsize=(12, 5))
    
    # Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Reds',
        xticklabels=['Non-Oncogenic', 'Oncogenic'],
        yticklabels=['Non-Oncogenic', 'Oncogenic']
    )
    plt.title("Cancer Mutation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recalls, precisions, label="Model")
    plt.axvline(x=TARGET_RECALL, color='r', linestyle='--', alpha=0.5, label=f'Target Recall ({TARGET_RECALL:.0%})')
    plt.title("Precision-Recall Trade-off")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Confidence)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def report_inference(model, engineer, variant_data, threshold):
    """
    Biologist-friendly inference report for a single variant.
    Explains the model's decision in clinical terms.
    """
    # Create a copy to avoid modifying the original and add defaults for missing columns
    data = variant_data.copy()
    defaults = {
        'ReviewStatus': 'criteria provided, multiple submitters, no conflicts', 
        'VariationID': 0,
        'NumberSubmitters': 10,
        'OriginSimple': 'unknown',
        'GeneSymbol': '',
        'CancerLabel': 0,  # Dummy for feature engineering (not used in prediction)
        'Oncogenicity': '-',
        'SomaticClinicalImpact': '-',
        'PhenotypeList': '',
        'ClinSigSimple': 0,
        'Assembly': 'GRCh38'
    }
    for key, val in defaults.items():
        if key not in data:
            data[key] = val

    # Prepare input
    X = engineer.transform(pd.DataFrame([data]))
    prob = model.predict(
        [X['gene'], X['type'], X['chrom'], X['origin'], X['numeric']], 
        verbose=0
    )[0][0]
    
    # Interpretation logic
    prediction = "PATHOGENIC (ONCOGENIC)" if prob >= threshold else "BENIGN (HARMLESS)"
    risk_level = "HIGH" if prob > 0.8 else "MODERATE" if prob >= threshold else "LOW"
    
    # Resolve gene symbol
    gene_id = variant_data.get('GeneID')
    gene_symbol = variant_data.get('GeneSymbol', '')
    if not gene_symbol and gene_id in engineer.gene_symbol_map:
        gene_symbol = engineer.gene_symbol_map[gene_id]
    gene_display = f"{gene_symbol} (ID: {gene_id})" if gene_symbol else f"Gene {gene_id}"
    
    print("\n" + "="*50)
    print("      CANCER MUTATION CLINICAL REPORT")
    print("="*50)
    print(f"VARIANT: {gene_display} | Pos: {variant_data.get('PositionVCF')}")
    print(f"DNA CHANGE: {variant_data.get('ReferenceAlleleVCF')} -> {variant_data.get('AlternateAlleleVCF')}")
    print(f"ORIGIN: {data.get('OriginSimple', 'unknown')}")
    print("-" * 50)
    print(f"FINAL CLASSIFICATION: {prediction}")
    print(f"CONFIDENCE SCORE: {prob*100:.1f}%")
    print(f"RISK LEVEL: {risk_level}")
    print("-" * 50)
    
    # Biological Explanations
    print("BIOLOGICAL CONTEXT:")
    ti_tv = "Transition" if X['numeric'][0][7] > 0 else "Transversion"
    print(f" - Mutation Type: {variant_data.get('Type')} ({ti_tv})")
    
    print("\nCLINICAL ADVICE:")
    if prediction == "PATHOGENIC (ONCOGENIC)":
        print(" [!] High priority for clinical follow-up.")
        print(" [!] Mutation shows genomic signatures common in cancer drivers.")
        if risk_level == "HIGH":
            print(" [!!] Very high confidence — strongly associated with oncogenic activity.")
    else:
        print(" [ok] Likely a common variation or harmless passenger mutation.")
    print("="*50)
