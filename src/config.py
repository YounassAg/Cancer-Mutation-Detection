import os

# Data configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "variant_summary.txt")
MODEL_DIR = "models"
LOG_DIR = "logs"

# High-confidence review status filter (for general variants)
HIGH_CONFIDENCE_STATUS = [
    'criteria provided, multiple submitters, no conflicts',
    'reviewed by expert panel',
    'practice guideline'
]

# Relaxed review status — includes "conflicting classifications"
# Many well-known cancer drivers (e.g., BRAF V600E) have conflicting
# germline classifications but clear oncogenicity annotations.
RELAXED_REVIEW_STATUS = HIGH_CONFIDENCE_STATUS + [
    'criteria provided, conflicting classifications'
]

# Feature columns — expanded to include oncogenicity and origin data
INPUT_COLUMNS = [
    'Type', 'Chromosome', 'PositionVCF',
    'ReferenceAlleleVCF', 'AlternateAlleleVCF',
    'GeneID', 'GeneSymbol', 'NumberSubmitters', 'ReviewStatus',
    'VariationID', 'Assembly', 'ClinSigSimple', 'PhenotypeList',
    'Oncogenicity', 'SomaticClinicalImpact', 'OriginSimple'
]

# Cancer phenotype keywords (for Tier 3 labeling fallback)
CANCER_KEYWORDS = [
    'cancer', 'tumor', 'carcinoma', 'leukemia', 'lymphoma',
    'sarcoma', 'glioma', 'oncology', 'neoplasm', 'malignant',
    'adenocarcinoma', 'melanoma', 'oncogenic'
]

# Oncogenicity labels from ClinVar (Tier 1 — Gold standard)
ONCOGENIC_LABELS = ['Oncogenic', 'Likely oncogenic']

# Somatic clinical impact tiers (Tier 2 — Silver standard)
SOMATIC_IMPACT_POSITIVE = ['Tier I - Strong', 'Tier II - Potential']

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SPLIT = 0.15
BATCH_SIZE = 512   # Reduced from 2048 for better gradient signal on rare positives
EPOCHS = 100

# Target medical sensitivity (increased for safety-critical cancer classification)
TARGET_RECALL = 0.90