# Dataset Documentation

This directory contains the primary dataset used for the Cancer Mutation Detection project.

## Main File

- **`variant_summary.txt`**: This is a large, tab-separated values (TSV) file containing genetic variant data sourced from the [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) database. Due to its size (~4GB), it is loaded in chunks during processing.

## Content Overview

The dataset provides comprehensive annotations for human genetic variations, detailing their genomic location, type, and clinical significance. For this project, the data is heavily filtered to extract signals related to cancer and oncogenicity.

### Key Columns Utilized

During the data loading phase (handled by `src/data_loader.py` and configured in `src/config.py`), the following key columns are extracted from the dataset:

- **Variant Identification**: `VariationID`, `Assembly` (Filtered strictly for `GRCh38`)
- **Genomic Location & Type**: `Chromosome`, `PositionVCF`, `ReferenceAlleleVCF`, `AlternateAlleleVCF`, `Type`, `GeneID`, `GeneSymbol`
- **Clinical Annotations**:
  - `Oncogenicity`: ClinVar's direct assessment of the variant's potential to cause cancer.
  - `SomaticClinicalImpact`: Information regarding the variant's impact in somatic tissues (tumors).
  - `ClinSigSimple`: Simplified clinical significance (e.g., 0 for Benign, 1 for Pathogenic).
  - `PhenotypeList`: Associated diseases or phenotypes.
- **Metadata**: `NumberSubmitters`, `ReviewStatus`, `OriginSimple` (e.g., germline, somatic)

## Label Engineering Strategy

The project employs a multi-tiered strategy to construct a binary `CancerLabel` (1 for Oncogenic/Positive, 0 for Benign/Negative) based on clinical evidence, prioritizing medical sensitivity.

### Positive Class (Oncogenic)
- **Tier 1 (Gold)**: `Oncogenicity` explicitly marked as "Oncogenic" or "Likely oncogenic".
- **Tier 2 (Silver)**: `SomaticClinicalImpact` marked as "Tier I - Strong" or "Tier II - Potential".
- **Tier 3 (Bronze)**: `ClinSigSimple` indicates pathogenic (1) **AND** `PhenotypeList` contains cancer-related keywords (e.g., cancer, tumor, carcinoma). Requires strict high-confidence review status.

*Note: For Tier 1 and 2, the review status filter is relaxed to include "conflicting classifications" to ensure well-known cancer drivers (which sometimes have conflicting germline classifications) are not missed.*

### Negative Class (Benign)
- Variants with `ClinSigSimple` as Benign (0) OR `Oncogenicity` explicitly marked as "Benign".
- Strict review status (high confidence only) is enforced unless the `Oncogenicity` column explicitly confirms it is benign.

### Data Processing Note
Missing `OriginSimple` values are imputed with 'unknown'. Duplicate entries are dropped based on `VariationID` to prevent data leakage between training, validation, and test sets.
