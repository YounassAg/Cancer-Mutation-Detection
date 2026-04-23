import os

# Data configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "variant_summary.txt")
MODEL_DIR = "models"
LOG_DIR = "logs"

# High-confidence review status filter
HIGH_CONFIDENCE_STATUS = [
    'criteria provided, multiple submitters, no conflicts',
    'reviewed by expert panel',
    'practice guideline'
]

# Feature columns
INPUT_COLUMNS = [
    'Type', 'Chromosome', 'PositionVCF', 
    'ReferenceAlleleVCF', 'AlternateAlleleVCF', 
    'GeneID', 'NumberSubmitters', 'ReviewStatus', 
    'VariationID', 'Assembly', 'ClinSigSimple', 'PhenotypeList'
]

# Cancer filtering
CANCER_KEYWORDS = [
    'cancer', 'tumor', 'carcinoma', 'leukemia', 'lymphoma', 
    'sarcoma', 'glioma', 'oncology', 'neoplasm', 'malignant', 
    'adenocarcinoma', 'melanoma', 'oncogenic'
]

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SPLIT = 0.15
BATCH_SIZE = 2048
EPOCHS = 100

# Target medical sensitivity
TARGET_RECALL = 0.85