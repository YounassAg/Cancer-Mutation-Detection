# Cancer Mutation Classification

A deep-learning-based classification pipeline designed to distinguish between **Pathogenic** and **Benign** genetic mutations using high-confidence clinical evidence from the ClinVar database.

## Project Vision
Traditional genetic testing is often slow and cost-prohibitive. This project leverages neural networks to predict clinical significance with a focus on **Medical Sensitivity**. In clinical diagnostics, missing a pathogenic variant (False Negative) is far more dangerous than a false alarm. This model is optimized to catch at least **85% of dangerous mutations**.

### Technical Highlights
- **Focal Loss Integration**: Addresses class imbalance by forcing the model to focus on "hard" examples.
- **Ti/Tv Analysis**: Incorporates biological Transition/Transversion ratios as predictive features.
- **Modular Pipeline**: Decoupled architecture for data processing, feature engineering, and model evaluation.
- **Medical Threshold Tuning**: Decision boundaries are automatically adjusted based on target clinical sensitivity.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure
- `src/`: Core logic modules (Config, Data Loading, Features, Model, etc.)
- `models/`: Saved model weights (`.h5` format)
- `notebooks/`: Original research and experimentation
- `train_model.py`: Main execution script to train and evaluate the model

## Usage

To train the model and evaluate its medical utility:

```bash
python train_model.py
```

## Clinical Performance
The model is tuned to prioritize **Recall (Sensitivity)** for pathogenic cases. 

| Metric | Target | Current |
| :--- | :--- | :--- |
| Pathogenic Recall | > 85% | 85% |
| Pathogenic Precision | > 50% | ~54% |
| Overall Accuracy | > 80% | ~81% |
