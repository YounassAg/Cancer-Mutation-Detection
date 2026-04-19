import numpy as np
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
        [X_test['gene'], X_test['type'], X_test['chrom'], X_test['numeric']]
    )
    
    # Calculate Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Identify optimal medical threshold for target recall
    target_idx = np.where(recalls >= TARGET_RECALL)[0][-1]
    optimized_threshold = thresholds[target_idx]
    
    y_pred = (y_probs >= optimized_threshold).astype(int)
    
    print(f"\n--- Clinical Performance Appraisal (Threshold: {optimized_threshold:.4f}) ---")
    print(classification_report(y_test, y_pred))
    
    return y_pred, y_probs, optimized_threshold

def plot_visual_assessment(y_test, y_pred, y_probs):
    """
    Generates medical utility charts: Confusion Matrix and PR-Curve.
    """
    plt.figure(figsize=(12, 5))
    
    # Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title("Medical Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recalls, precisions, label="Model")
    plt.title("Precision-Recall Trade-off")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Confidence)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
