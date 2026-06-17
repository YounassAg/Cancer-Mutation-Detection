"""
Cancer Mutation Detection — Flask Web Application
Serves a browser-based interface for testing the pre-trained mutation classifier.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model import focal_loss
from src.features import FeatureEngineer

# Suppress verbose TF logging before importing keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model

# Paths
MODEL_PATH = os.path.join(PROJECT_ROOT, 'notebooks', 'models', 'mutation_classifier.h5')
ENGINEER_PATH = os.path.join(PROJECT_ROOT, 'notebooks', 'models', 'feature_engineer.pkl')
DEFAULT_THRESHOLD = 0.3619

EXAMPLE_VARIANTS = [
    {'label':'KRAS','description':'Common oncogene in lung & colorectal cancer','GeneID':3845,'GeneSymbol':'KRAS','PositionVCF':25245350,'ReferenceAlleleVCF':'C','AlternateAlleleVCF':'T','Type':'snp','Chromosome':'12','OriginSimple':'somatic'},
    {'label':'TP53','description':'Tumor suppressor — most mutated gene in cancer','GeneID':7157,'GeneSymbol':'TP53','PositionVCF':7674220,'ReferenceAlleleVCF':'C','AlternateAlleleVCF':'T','Type':'snp','Chromosome':'17','OriginSimple':'somatic'},
    {'label':'EGFR','description':'Key target in non-small cell lung cancer','GeneID':1956,'GeneSymbol':'EGFR','PositionVCF':55181378,'ReferenceAlleleVCF':'T','AlternateAlleleVCF':'G','Type':'snp','Chromosome':'7','OriginSimple':'somatic'},
    {'label':'BRAF','description':'V600E — driver in melanoma & thyroid cancer','GeneID':673,'GeneSymbol':'BRAF','PositionVCF':140753336,'ReferenceAlleleVCF':'A','AlternateAlleleVCF':'T','Type':'snp','Chromosome':'7','OriginSimple':'somatic'},
    {'label':'PIK3CA','description':'Frequent in breast & endometrial cancers','GeneID':5290,'GeneSymbol':'PIK3CA','PositionVCF':179218303,'ReferenceAlleleVCF':'G','AlternateAlleleVCF':'A','Type':'snp','Chromosome':'3','OriginSimple':'somatic'},
]

app = Flask(__name__)

print('[*] Loading mutation classifier model ...')
model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.75)})
print('[*] Loading feature engineer ...')
with open(ENGINEER_PATH, 'rb') as f:
    engineer = pickle.load(f)
print('[OK] Model and feature engineer loaded successfully.')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/example-variants')
def example_variants():
    return jsonify(EXAMPLE_VARIANTS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        required = ['GeneID','GeneSymbol','PositionVCF','ReferenceAlleleVCF','AlternateAlleleVCF','Type','Chromosome','OriginSimple']
        missing = [f for f in required if f not in data or data[f] in (None, '')]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

        variant = {
            'GeneID': int(data['GeneID']),
            'GeneSymbol': str(data['GeneSymbol']).strip(),
            'PositionVCF': int(data['PositionVCF']),
            'ReferenceAlleleVCF': str(data['ReferenceAlleleVCF']).strip().upper(),
            'AlternateAlleleVCF': str(data['AlternateAlleleVCF']).strip().upper(),
            'Type': str(data['Type']).strip(),
            'Chromosome': str(data['Chromosome']).strip(),
            'OriginSimple': str(data['OriginSimple']).strip(),
        }

        defaults = {
            'ReviewStatus': 'criteria provided, multiple submitters, no conflicts',
            'VariationID': 0, 'NumberSubmitters': 10, 'Oncogenicity': '-',
            'SomaticClinicalImpact': '-', 'PhenotypeList': '', 'ClinSigSimple': 0,
            'Assembly': 'GRCh38', 'CancerLabel': 0,
        }
        for key, val in defaults.items():
            variant.setdefault(key, val)

        X = engineer.transform(pd.DataFrame([variant]))
        prob = float(model.predict(
            [X['gene'], X['type'], X['chrom'], X['origin'], X['numeric']], verbose=0
        )[0][0])

        threshold = DEFAULT_THRESHOLD
        is_oncogenic = prob >= threshold
        classification = 'PATHOGENIC (ONCOGENIC)' if is_oncogenic else 'BENIGN (NON-ONCOGENIC)'
        risk_level = 'HIGH' if prob > 0.8 else ('MODERATE' if is_oncogenic else 'LOW')
        ti_tv = 'Transition' if X['numeric'][0][7] > 0 else 'Transversion'

        advice = []
        if is_oncogenic:
            advice.append('High priority for clinical follow-up.')
            advice.append('Mutation shows genomic signatures common in cancer drivers.')
            if risk_level == 'HIGH':
                advice.append('Very high confidence — strongly associated with oncogenic activity.')
        else:
            advice.append('Likely a common variation or harmless passenger mutation.')

        result = {
            'classification': classification,
            'is_oncogenic': is_oncogenic,
            'confidence': round(prob * 100, 1),
            'risk_level': risk_level,
            'threshold': round(threshold * 100, 1),
            'mutation_type': f"{variant['Type']} ({ti_tv})",
            'advice': advice,
            'variant': {
                'gene_display': f"{variant['GeneSymbol']} (ID: {variant['GeneID']})",
                'position': variant['PositionVCF'],
                'dna_change': f"{variant['ReferenceAlleleVCF']} → {variant['AlternateAlleleVCF']}",
                'origin': variant['OriginSimple'],
                'chromosome': variant['Chromosome'],
            },
        }
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
