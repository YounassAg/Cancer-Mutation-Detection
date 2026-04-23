# Cancer Mutation Detection

A deep-learning-based classification pipeline designed to identify **Oncogenic** cancer driver mutations using clinical evidence from the ClinVar database.

## Project Vision
Traditional genetic testing is often slow and cost-prohibitive. This project leverages neural networks to predict oncogenicity with a focus on **Medical Sensitivity**. In clinical diagnostics, missing an oncogenic driver mutation (False Negative) is far more dangerous than a false alarm. This model is optimized to catch at least **90% of dangerous mutations**.

### Technical Highlights
- **Multi-Tiered Cancer Labeling**: Uses ClinVar's `Oncogenicity` column (Tier 1), `SomaticClinicalImpact` (Tier 2), and cancer-phenotype-matched pathogenicity (Tier 3) for accurate cancer-specific labels.
- **Focal Loss + Class Weights**: Dual strategy to address severe class imbalance (~96% benign / ~4% oncogenic).
- **Biological Feature Engineering**: Ti/Tv analysis, somatic/germline origin encoding, gene-level oncogenic frequencies, and allele length differences.
- **Multi-Input Embedding Architecture**: Embedding layers for high-cardinality categorical features (Gene, Chromosome, Variant Type, Origin).
- **Medical Threshold Tuning**: Decision boundaries are automatically adjusted based on target clinical sensitivity (90% recall).

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure
- `src/`: Core logic modules (Config, Data Loading, Features, Model, Training, Evaluation)
- `models/`: Saved model weights (`.h5` format)
- `notebooks/`: Research and experimentation notebook
- `train_model.py`: Main execution script to train and evaluate the model

## Usage

To train the model and evaluate its medical utility:

```bash
python train_model.py
```

## Clinical Performance
The model is tuned to prioritize **Recall (Sensitivity)** for oncogenic variants.

| Metric | Target |
| :--- | :--- |
| Oncogenic Recall | > 90% |
| Oncogenic Precision | > 40% |
| Overall Accuracy | > 90% |

## Data Labeling Strategy

| Tier | Source Column | Positive Criteria | Review Status |
| :--- | :--- | :--- | :--- |
| 1 (Gold) | `Oncogenicity` | "Oncogenic", "Likely oncogenic" | Relaxed (includes conflicting) |
| 2 (Silver) | `SomaticClinicalImpact` | "Tier I - Strong", "Tier II - Potential" | Relaxed |
| 3 (Bronze) | `ClinSigSimple` + `PhenotypeList` | Pathogenic + cancer keywords | Strict (high confidence only) |
| Negative | `ClinSigSimple` | 0 (Benign) | Strict |
