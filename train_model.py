from src.gpu_setup import setup_gpu
setup_gpu()

from src.data_loader import load_clean_data
from src.features import FeatureEngineer
from src.training import train_pipeline
from src.evaluation import evaluate_medical_utility, plot_visual_assessment
from src.config import RANDOM_STATE, TEST_SIZE
from sklearn.model_selection import train_test_split
import os

def main():
    """
    Main entry point for the Cancer Mutation Detection pipeline.
    Orchestrates data loading, feature engineering, training, and medical evaluation.
    """
    print("--- Cancer Mutation Classification - Analysis Pipeline ---")
    
    # 1. Load data
    df = load_clean_data()
    
    # 2. Split data
    train_df, test_df = train_test_split(
        df, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=df['ClinSigSimple']
    )
    
    # 3. Feature Engineering
    engineer = FeatureEngineer()
    X_train = engineer.fit_transform(train_df)
    X_test = engineer.transform(test_df)
    
    y_train = train_df['ClinSigSimple'].values
    y_test = test_df['ClinSigSimple'].values
    
    # 4. Training
    model, history = train_pipeline(X_train, y_train, engineer)
    
    # 5. Evaluate Clinical Utility
    y_pred, y_probs, threshold = evaluate_medical_utility(model, X_test, y_test)
    
    # Save the professional model for deployment
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/mutation_classifier.h5')
    print("\nModel saved successfully in 'models/' directory.")

if __name__ == "__main__":
    main()
