# Cancer Mutation Detection - Deep Learning Project

**Prédiction de Mutations Génétiques Pathogènes au Cancer**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)

---

## 📋 Description

Ce projet implémente et compare **3 modèles de Machine Learning** pour prédire si une mutation génétique est **pathogène** (cause le cancer) ou **bénigne** (inoffensive).

**Objectif:** Atteindre ≥ 80% de ROC-AUC sur le dataset ClinVar (~65k mutations).

---

## 🏗️ Architecture des Modèles

| Modèle | Type | Architecture | ROC-AUC Attendu |
|--------|------|--------------|-----------------|
| **XGBoost** | Gradient Boosting | 100 estimateurs, max_depth=6 | 80-85% |
| **NN Simple** | Dense Neural Network | 7 → 64 → 32 → 1 | 82-88% |
| **NN Avancé** | Dense NN + BatchNorm | 7 → 128 → 64 → 32 → 16 → 1 | 85-90% |

---

## 📊 Dataset

**ClinVar Conflicting** - Base de données publique de mutations génétiques annotées

- **Source:** [Kaggle](https://www.kaggle.com/kevinarvai/clinvar-conflicting)
- **Taille:** ~65,000 mutations
- **Features:** 7 features bioinformatiques sélectionnées
- **Target:** Binaire (0 = Benign, 1 = Pathogenic)

### 7 Features Utilisées

1. **CADD_PHRED** - Score de pathogénicité combiné (0-100)
2. **CADD_RAW** - Version brute du score CADD
3. **SIFT** - Score d'impact sur la protéine (0-1)
4. **PolyPhen** - Prédiction d'effet sur acide aminé (0-1)
5. **AF_EXAC** - Fréquence population EXAC (0-1)
6. **AF_TGP** - Fréquence 1000 Genomes (0-1)
7. **BLOSUM62** - Score de changement d'acide aminé

---

## 🚀 Installation

```bash
# Cloner le repository
git clone <repository-url>
cd Cancer-Mutation-Detection

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances

- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- scikit-learn ≥ 1.3.0
- xgboost ≥ 2.0.0
- tensorflow ≥ 2.13.0
- matplotlib ≥ 3.7.0
- seaborn ≥ 0.12.0

---

## 💻 Utilisation

### Pipeline Complet (Recommandé)

```bash
# Exécuter le pipeline complet
python src/main.py

# Avec options
python src/main.py --nrows 50000 --test-size 0.2 --seed 42
```

### Utilisation Interactive (Jupyter Notebook)

```bash
# Lancer le notebook
jupyter notebook notebooks/

# Ouvrir 01_complete_pipeline.ipynb
```

### Utilisation Module par Module

```python
# 1. Chargement
from src.data_loader import load_clinvar_data
df = load_clinvar_data("data/clinvar_conflicting.csv")

# 2. Prétraitement
from src.preprocessing import clean_clinvar_data, prepare_train_test_data
df_clean, features = clean_clinvar_data(df)
X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test_data(df_clean)

# 3. Entraînement
from src.train import train_all_models
models = train_all_models(X_train, y_train)

# 4. Évaluation
from src.evaluate import evaluate_all_models
results = evaluate_all_models(models['xgboost']['model'], 
                              models['nn_simple']['model'], 
                              models['nn_advanced']['model'],
                              X_test, y_test, feature_names)
```

---

## 📁 Structure du Projet

```
Cancer-Mutation-Detection/
├── data/
│   ├── clinvar_conflicting.csv      # Dataset brut
│   ├── X_train.csv, X_test.csv      # Features split
│   ├── y_train.csv, y_test.csv      # Target split
│   └── scaler.pkl                   # Scaler fitté
├── models/
│   ├── xgboost_model.pkl            # Modèle XGBoost
│   ├── nn_simple.keras              # NN Simple
│   ├── nn_advanced.keras            # NN Avancé
│   └── training_histories.pkl       # Historiques
├── notebooks/
│   └── 01_complete_pipeline.ipynb   # Notebook complet
├── results/
│   ├── roc_curves.png               # Courbes ROC
│   ├── confusion_matrices.png       # Matrices de confusion
│   ├── feature_importance.png       # Importance features
│   ├── training_history.png         # Historique entraînement
│   ├── model_comparison.csv         # Tableau comparatif
│   └── metrics.json                 # Métriques complètes
├── src/
│   ├── data_loader.py               # Chargement données
│   ├── preprocessing.py             # Nettoyage & normalisation
│   ├── models.py                    # Architectures
│   ├── train.py                     # Entraînement
│   ├── evaluate.py                  # Évaluation
│   └── main.py                      # Pipeline complet
├── CONTEXT.md                       # Documentation détaillée
├── README.md                        # Ce fichier
└── requirements.txt                 # Dépendances
```

---

## 📈 Résultats Attendus

### Métriques

| Métrique | Description | Objectif |
|----------|-------------|----------|
| **ROC-AUC** | Capacité de discrimination | ≥ 80% |
| **Precision** | % de vrais positifs parmi les positifs prédits | > 75% |
| **Recall** | % de positifs réels détectés | > 70% |
| **F1-Score** | Moyenne harmonique Precision/Recall | > 75% |

### Interprétation des Résultats

```
ROC-AUC:
  0.50 = Aléatoire (inutile)
  0.70-0.80 = Acceptable
  0.80-0.90 = Bon ✓
  0.90+ = Excellent
```

---

## 🧬 Contexte Biologique

### Qu'est-ce qu'une Mutation?

Une mutation est une **erreur** dans l'ADN (manuel d'instructions génétique):

```
AVANT (correct): ATGGCATAGCTACGATAGC
APRÈS (erreur):  ATGGCATAGCTACGATXGC
                                    ↑ Une lettre changée
```

### Types de Mutations

| Type | Description | Label |
|------|-------------|-------|
| **Bénigne** | Le corps fonctionne normalement | 0 |
| **Pathogène** | Cause le cancer | 1 |

### Pourquoi C'est Important?

- **1000+ nouvelles mutations** découvertes chaque jour
- Test en laboratoire: **6 mois à 2 ans**, **10,000€-100,000€**
- Prédiction ML: **< 1 seconde**, **gratuit**

---

## 🎓 Références

### Données
- [ClinVar Officiel](https://www.ncbi.nlm.nih.gov/clinvar/)
- [ClinVar Kaggle](https://www.kaggle.com/kevinarvai/clinvar-conflicting)

### Deep Learning
- [TensorFlow/Keras](https://www.tensorflow.org/api_docs)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

### Publications
- Kircher et al. (2016) "A general framework for estimating the relative pathogenicity of human genetic variants" Nature Genetics
- Sundaram et al. (2021) "Predicting the clinical impact of human mutation with deep neural networks" Nature Medicine

---

## 👥 Auteurs

**Projet Deep Learning - Prédiction de Mutations Cancéreuses**

---

## 📄 License

Ce projet utilise des données publiques ClinVar (domaine public).
Code sous license MIT.
