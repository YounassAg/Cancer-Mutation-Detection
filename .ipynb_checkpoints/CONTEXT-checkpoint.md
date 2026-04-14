# CONTEXT.md - Projet Deep Learning : Prédiction de Mutations Génétiques Pathogènes

## RÉSUMÉ EXÉCUTIF

**Objectif Principal :** Construire un modèle Deep Learning qui prédit si une mutation génétique cause le cancer (pathogène) ou est inoffensive (bénigne).

**Domaine :** Biologie / Bioinformatique / Apprentissage Automatique  
**Équipe :** Binôme  
**Précision Attendue :** 80-90% ROC-AUC

---

## **PARTIE 1 : CONTEXTE BIOLOGIQUE (Simplifié)**

### **Qu'est-ce que l'ADN ?**

L'ADN est un **manuel d'instructions** pour construire un corps humain. Ce manuel contient ~3 milliards de lettres (A, T, G, C).

```
Exemple :
ATGGCATAGCTACGATAGCTAGCTGATCGATAGCTAGC...

Chaque groupe de 3 lettres = 1 instruction pour fabriquer une protéine
Exemple : ATG = "Commencer ici"
```

### **Qu'est-ce qu'une Mutation ?**

Une mutation = une **ERREUR** dans ce manuel.

```
AVANT (correct)  : ATGGCATAGCTACGATAGC
APRÈS (erreur)   : ATGGCATAGCTACGATXGC
                                   ↑ Une lettre changée
```

**Types de mutations :**

1. **INOFFENSIVE (Benign)**
   - Le corps fonctionne normalement
   - Exemple : cheveux roux au lieu de noirs
   - Label : 0 (Benign)

2. **DANGEREUSE (Pathogenic)**
   - Le corps fonctionne mal
   - Exemple : les cellules se divisent à l'infini → CANCER
   - Label : 1 (Pathogenic)

### **Pourquoi C'est Important ?**

- Les scientifiques découvrent **1000+ nouvelles mutations par jour**
- Tester chacune en laboratoire prend **6 mois à 2 ans** et coûte **10,000€ - 100,000€**
- **Solution :** Utiliser le Deep Learning pour prédire en **< 1 seconde**, gratuitement

---

## **PARTIE 2 : LE DATASET (ClinVar)**

### **Qu'est-ce que ClinVar ?**

ClinVar = base de données **publique et gratuite** des mutations génétiques connues avec leurs annotations cliniques.

- **Source officielle :** https://www.ncbi.nlm.nih.gov/clinvar/
- **Source Kaggle :** https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting
- **Taille :** ~500,000 mutations annotées
- **Format :** CSV avec 43 colonnes
- **Licence :** Données publiques (libre utilisation)

### **Structure du Dataset**

**Une ligne = Une mutation trouvée chez un patient**

Exemple :
```
Mutation du patient Jean :
  Chromosome : 13
  Position : 12,345
  Avant : A
  Après : T
  Fréquence en population : 0.01%
  Score d'experts : 28
  Diagnostic : Pathogenic (cause le cancer du sein)
```

### **Les 43 Colonnes - Expliquées Simplement**

#### **Groupe 1 : Localisation de la Mutation (4 colonnes)**

```
CHROM    → Dans quel "chapitre" du manuel ? (1-22, X, Y)
           Exemple : 13

POS      → À quelle "ligne" du chapitre ? 
           Exemple : 12,345,678

REF      → Quelle lettre était là AVANT ?
           Exemple : A

ALT      → Quelle lettre est MAINTENANT ?
           Exemple : T

→ Résumé : "Chapitre 13, Ligne 12345 : A → T"
```

#### **Groupe 2 : La Réponse (Label) LA PLUS IMPORTANTE (1 colonne)**

```
CLNSIGINCL → "Est-ce Pathogenic ou Benign ?"

Valeurs possibles :
  • "Pathogenic"  → Label = 1 (dangereux, cause le cancer)
  • "Benign"      → Label = 0 (inoffensif, normal)
  • "VUS"         → À IGNORER (Variant of Uncertain Significance)

→ C'EST CE QU'ON VEUT PRÉDIRE !
```

#### **Groupe 3 : Fréquence en Population (3 colonnes)**

```
AF_ESP   → "Sur 1000 personnes saines, combien ont cette mutation ?"
           Population : Europe du Sud (ESP)
           Valeur : 0 à 1 (0.1 = 10%)

AF_EXAC  → Même, mais autre population (EXAC)

AF_TGP   → Même, mais 3e population (1000 Genomes)

LOGIQUE :
  • Mutation RARE (0.01%)    → probablement dangereuse
  • Mutation COMMUNE (30%)   → probablement inoffensive
```

#### **Groupe 4 : Information sur le Diagnostic Clinique (3 colonnes)**

```
CLNDISDB    → "Quelles maladies sont liées ?"
              Exemple : "OMIM:113705"

CLNDN       → "Nom de la maladie"
              Exemple : "Hereditary breast and ovarian cancer"

CLNVC       → "Type de variation"
              Exemple : "Missense" ou "Frameshift"
```

#### **Groupe 5 : Scores de Prédiction d'Experts (3 colonnes) TRÈS IMPORTANT**

```
CADD_PHRED → Score de pathogénicité (0-100)
             Combinaison de plusieurs méthodes d'experts
             > 20 = probablement pathogène
             < 10 = probablement bénin
             
CADD_RAW   → Version "brute" du CADD (rarement utilisé)

SIFT       → Score d'un autre expert (0-1)
             Détermine si le changement affecte la protéine
             < 0.05 = Deleterious (mauvais)
             > 0.05 = Tolerated (bon)

PolyPhen   → Score d'un 3e expert (0-1)
             Prédit les effets d'un changement d'acide aminé
             > 0.85 = Probably Damaging
             < 0.15 = Benign
             0.15-0.85 = Possibly Damaging

BLOSUM62   → Score du changement d'acide aminé
             Montre comment les acides aminés se ressemblent
             Négatif = changement grave
             Positif = changement bénin
```

#### **Groupe 6 : Type de Changement (2 colonnes)**

```
CONSEQUENCE → "Quel type de dégâts cette mutation fait ?"
              Valeurs :
              • "Frameshift" → Décalage complet (très mauvais)
              • "Nonsense" → Grosse cassure (très mauvais)
              • "Missense" → Petit changement (peut être bad)
              • "Synonymous" → Pas de changement (bon)

IMPACT      → Résumé simple de la sévérité
              Valeurs :
              • "HIGH" → Mutation très dangereuse
              • "MODERATE" → Peut être dangereuse
              • "LOW" → Probablement pas dangereuse
              • "MODIFIER" → Impact faible/aucun

BIOTYPE     → "Cette mutation affecte un gène utile ?"
              Valeurs :
              • "protein_coding" → Oui, important
              • "non_coding" → Non, moins important
              • "lncRNA" → ARN long (rarement utile)
```

### **7 Features Sélectionnées pour Notre Modèle**

Nous utiliserons ces 7 colonnes (features) comme **entrée du modèle** :

```
1. CADD_PHRED        → Score expert #1 (très informatif)
2. CADD_RAW          → Score expert #1 (version brute)
3. SIFT              → Score expert #2
4. PolyPhen          → Score expert #3
5. AF_EXAC           → Fréquence en population (rareté)
6. AF_TGP            → Fréquence en population (confirmation)
7. BLOSUM62          → Sévérité du changement d'acide aminé

TARGET (ce qu'on prédit) :
  • CLNSIGINCL convertie en 0 ou 1
```

**Pourquoi ces 7 ?**
- Tous les scores d'experts (utiliser la "sagesse des experts")
- Les fréquences (mutations rares = plus suspectes)
- Le changement d'acide aminé (sévérité)
- Peu de valeurs manquantes
- Directement prédictives

---

## **PARTIE 3 : ARCHITECTURE DU PROJET**

### **Structure des Fichiers**

```
Cancer-Mutation-Detection/
├── data/
│   ├── clinvar_raw.csv          (données brutes ~500MB)
│   ├── clinvar_cleaned.csv      (données nettoyées)
│   └── X_train.csv, X_test.csv, y_train.csv, y_test.csv
├── models/
│   ├── xgboost_model.pkl
│   ├── nn_simple.h5
│   └── nn_advanced.h5
├── notebooks/
│   ├── 01_exploration.ipynb     (EDA)
│   ├── 02_preprocessing.ipynb   (nettoyage)
│   ├── 03_modeling.ipynb        (training)
│   └── 04_evaluation.ipynb      (résultats)
├── results/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── metrics.json
├── src/
│   ├── data_loader.py           (charger données)
│   ├── preprocessing.py         (nettoyer)
│   ├── models.py                (architectures)
│   ├── train.py                 (entraînement)
│   └── evaluate.py              (évaluation)
├── CONTEXT.md                   (ce fichier)
├── README.md                    (pour documenter)
└── requirements.txt             (dépendances)
```

---

## **PARTIE 4 : LES 3 MODÈLES À CONSTRUIRE**

### **MODÈLE 1 : XGBoost (Baseline)**

**Type :** Gradient Boosting (pas du Deep Learning, mais baseline importante)

**Pourquoi ?**
- Excellent sur données tabulaires
- Pas besoin de normalisation
- Fast
- Feature importance automatique

**Architecture :**
```python
XGBClassifier(
    n_estimators=100,        # 100 "experts" qui votent
    max_depth=6,             # Profondeur maximale
    learning_rate=0.1,       # Vitesse d'apprentissage
    class_weight='balanced'  # Gérer déséquilibre Pathogenic/Benign
)
```

**Temps d'entraînement :** ~2-5 minutes  
**Résultats attendus :** 80-85% ROC-AUC

---

### **MODÈLE 2 : Neural Network Simple (Principal)**

**Type :** Dense Neural Network (MLP - Multi-Layer Perceptron)

**Pourquoi ?**
- Démontre le Deep Learning
- Architecture simple mais efficace
- Facile à comprendre et interpréter
- Assez puissant pour ce problème

**Architecture :**
```
Input Layer
  └─ 7 neurones (nos 7 features)
  
Hidden Layer 1
  ├─ 64 neurones
  ├─ Activation : ReLU (Rectified Linear Unit)
  │   f(x) = max(0, x) → laisse passer les valeurs positives
  └─ Dropout 0.3 (oublier 30% random des infos)
  
Hidden Layer 2
  ├─ 32 neurones
  ├─ Activation : ReLU
  └─ Dropout 0.2 (oublier 20% random)
  
Output Layer
  ├─ 1 neurone
  └─ Activation : Sigmoid
      f(x) = 1 / (1 + e^(-x)) → output entre 0 et 1 (probabilité)

Total paramètres : ~2,600 (très léger)
```

**Code :**
```python
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_dim=7),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',              # Optimiseur (ajuste les poids)
    loss='binary_crossentropy',    # Loss pour classification binaire
    metrics=['AUC', 'Precision', 'Recall']
)
```

**Entraînement :**
```python
history = model.fit(
    X_train, y_train,
    epochs=50,                 # 50 passages sur toutes les données
    batch_size=32,             # Par groupe de 32 échantillons
    validation_split=0.2,      # 20% pour valider pendant l'entraînement
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,        # Arrêter si pas d'amélioration 5 epochs
            restore_best_weights=True
        )
    ]
)
```

**Temps d'entraînement :** ~5-10 minutes (CPU) ou ~1-2 minutes (GPU Google Colab)  
**Résultats attendus :** 82-88% ROC-AUC

---

### **MODÈLE 3 : Neural Network Avancé (Optionnel)**

**Type :** Dense Neural Network avec Batch Normalization

**Améliorations :**
- Batch Normalization après chaque couche
- Learning Rate Scheduling
- Regularization (L2)
- Cross-validation

**Architecture :**
```
Input Layer (7)
  ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.3)
  ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.3)
  ↓
Dense(32) + ReLU + BatchNorm + Dropout(0.2)
  ↓
Dense(16) + ReLU + Dropout(0.2)
  ↓
Dense(1) + Sigmoid
```

**Code :**
```python
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_dim=7, 
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(64, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(32, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(1, activation='sigmoid')
])

# Learning Rate Scheduling
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['AUC', 'Precision', 'Recall']
)
```

**Résultats attendus :** 85-90% ROC-AUC

---

## **PARTIE 5 : ÉTAPES D'EXÉCUTION DÉTAILLÉES**

### **ÉTAPE 1 : Exploration des Données (EDA)**

**Objectif :** Comprendre les données

**Actions :**
```python
# 1. Charger
df = pd.read_csv('clinvar.csv')
print(f"Dimensions : {df.shape}")  # Ex : (500000, 43)
print(f"Colonnes : {df.columns.tolist()}")

# 2. Première vue
print(df.head())       # Premières lignes
print(df.info())       # Types et valeurs manquantes
print(df.describe())   # Statistiques

# 3. Cible
print(df['CLNSIGINCL'].value_counts())
# Résultat attendu :
#   Benign        350000  (70%)
#   Pathogenic    150000  (30%)

# 4. Features
for feat in features:
    print(f"\n{feat}:")
    print(f"  Manquantes : {df[feat].isnull().sum()}")
    print(f"  Min : {df[feat].min()}")
    print(f"  Max : {df[feat].max()}")
    print(f"  Dtype : {df[feat].dtype}")

# 5. Corrélations
import seaborn as sns
correlation_matrix = df[features + ['CLNSIGINCL']].corr()
sns.heatmap(correlation_matrix, annot=True)
```

**Résultat attendu :**
- Benign >> Pathogenic (déséquilibre)
- CADD_PHRED, SIFT, PolyPhen fortement corrélés avec le label
- AF_EXAC, AF_TGP corrélés ensemble
- ~10-20% de valeurs manquantes par feature

---

### **ÉTAPE 2 : Nettoyage des Données**

**Objectif :** Préparer les données pour l'entraînement

**Actions :**
```python
# 1. Supprimer VUS (Variant of Uncertain Significance)
df = df[df['CLNSIGINCL'].isin(['Pathogenic', 'Benign'])]

# 2. Créer le label
df['target'] = (df['CLNSIGINCL'] == 'Pathogenic').astype(int)
# Benign = 0, Pathogenic = 1

# 3. Sélectionner les features
features = ['CADD_PHRED', 'CADD_RAW', 'SIFT', 'PolyPhen', 
            'AF_EXAC', 'AF_TGP', 'BLOSUM62']

# 4. Gérer les valeurs manquantes
for feat in features:
    # Remplir avec la médiane (statistique robuste)
    df[feat] = df[feat].fillna(df[feat].median())

# 5. Vérifier
print(f"Valeurs manquantes après : {df[features].isnull().sum().sum()}")
# Résultat : 0

# 6. Créer X et y
X = df[features].values
y = df['target'].values

print(f"X shape : {X.shape}")  # (450000, 7)
print(f"y shape : {y.shape}")  # (450000,)
print(f"Distribution y : {np.unique(y, return_counts=True)}")
# Résultat : (array([0, 1]), array([315000, 135000]))  # 70-30 split
```

---

### **ÉTAPE 3 : Normalisation des Données**

**Objectif :** Mettre toutes les features à la même échelle

**Pourquoi ?**
- CADD_PHRED : 0-100
- SIFT : 0-1
- AF_EXAC : 0-0.5
- Sans normalisation, les grandes valeurs dominent

**Code :**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Avant : mean={X[:, 0].mean():.2f}, std={X[:, 0].std():.2f}")
print(f"Après : mean={X_scaled[:, 0].mean():.2f}, std={X_scaled[:, 0].std():.2f}")
# Résultat : mean ≈ 0.00, std ≈ 1.00
```

**Important :** Normaliser sur X_train, puis appliquer à X_test !

---

### **ÉTAPE 4 : Split Train/Test**

**Objectif :** Diviser les données

**Code :**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,          # 20% test, 80% train
    random_state=42,        # Reproductibilité
    stratify=y              # Garder ratio Benign/Pathogenic
)

print(f"Train : {len(X_train)} samples ({X_train.shape})")
print(f"Test : {len(X_test)} samples ({X_test.shape})")
print(f"Train distribution : {np.unique(y_train, return_counts=True)}")
print(f"Test distribution : {np.unique(y_test, return_counts=True)}")
```

**Résultat attendu :**
```
Train : 360000 samples (360000, 7)
Test : 90000 samples (90000, 7)
Train distribution : [252000, 108000]  # 70-30
Test distribution : [63000, 27000]      # 70-30
```

---

### **ÉTAPE 5 : Entraînement des 3 Modèles**

#### **5A : XGBoost**

```python
from xgboost import XGBClassifier

# Créer
model_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1  # Paralléliser
)

# Entraîner
print("Entraînement XGBoost...")
model_xgb.fit(X_train, y_train)
print("Fait !")

# Prédictions
y_pred_xgb = model_xgb.predict_proba(X_test)[:, 1]
```

#### **5B : Neural Network Simple**

```python
import tensorflow as tf
from tensorflow import keras

# Créer
model_nn = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_dim=7),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compiler
model_nn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['AUC', 'Precision', 'Recall']
)

# Entraîner
print("Entraînement Neural Network Simple...")
history = model_nn.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
print("Fait !")

# Prédictions
y_pred_nn = model_nn.predict(X_test, verbose=0)[:, 0]
```

#### **5C : Neural Network Avancé**

```python
# Créer
model_nn_adv = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_dim=7,
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(64, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(32, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(1, activation='sigmoid')
])

# Compiler avec learning rate schedule
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model_nn_adv.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['AUC', 'Precision', 'Recall']
)

# Entraîner
print("Entraînement Neural Network Avancé...")
history_adv = model_nn_adv.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)
print("Fait !")

# Prédictions
y_pred_nn_adv = model_nn_adv.predict(X_test, verbose=0)[:, 0]
```

---

### **ÉTAPE 6 : Évaluation**

**Objectif :** Mesurer la performance

**Métriques à Calculer :**

```python
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc as pr_auc
)

def evaluate_model(y_true, y_pred, model_name):
    """Évaluer un modèle et afficher tous les métriques"""
    
    # Convertir en binaire (0/1) pour certaines métriques
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # Calculs
    roc_auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    # Courbe Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
    pr_auc_score = pr_auc(recall_vals, precision_vals)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    # Affichage
    print(f"\n{'='*60}")
    print(f"RÉSULTATS : {model_name}")
    print(f"{'='*60}")
    print(f"ROC-AUC              : {roc_auc:.4f}")
    print(f"Precision-Recall AUC : {pr_auc_score:.4f}")
    print(f"Precision            : {precision:.4f}")
    print(f"Recall (Sensibilité) : {recall:.4f}")
    print(f"F1-Score             : {f1:.4f}")
    print(f"\nMatrice de Confusion :")
    print(f"  True Negatives  : {tn:6d}  (bien prédits comme Benign)")
    print(f"  False Positives : {fp:6d}  (mal prédits comme Pathogenic)")
    print(f"  False Negatives : {fn:6d}  (mal prédits comme Benign)")
    print(f"  True Positives  : {tp:6d}  (bien prédits comme Pathogenic)")
    print(f"\nSpécificité : {tn / (tn + fp):.4f}")
    print(f"Sensibilité : {tp / (tp + fn):.4f}")
    print(f"{'='*60}")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc_score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Évaluer les 3 modèles
results_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost")
results_nn = evaluate_model(y_test, y_pred_nn, "Neural Network Simple")
results_nn_adv = evaluate_model(y_test, y_pred_nn_adv, "Neural Network Avancé")
```

**Résultats Attendus :**

```
XGBoost :
  ROC-AUC : 0.820-0.850
  Precision : 0.75-0.80
  Recall : 0.70-0.75

Neural Network Simple :
  ROC-AUC : 0.835-0.880
  Precision : 0.78-0.83
  Recall : 0.72-0.78

Neural Network Avancé :
  ROC-AUC : 0.850-0.900
  Precision : 0.80-0.85
  Recall : 0.75-0.80
```

---

### **ÉTAPE 7 : Visualisations**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Courbes ROC
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (y_pred, model_name, ax) in enumerate([
    (y_pred_xgb, "XGBoost", axes[0]),
    (y_pred_nn, "NN Simple", axes[1]),
    (y_pred_nn_adv, "NN Avancé", axes[2])
]):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    ax.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(model_name)
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=300)
plt.show()

# 2. Matrices de Confusion
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (results, model_name, ax) in enumerate([
    (results_xgb, "XGBoost", axes[0]),
    (results_nn, "NN Simple", axes[1]),
    (results_nn_adv, "NN Avancé", axes[2])
]):
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Pathogenic'],
                yticklabels=['Benign', 'Pathogenic'])
    ax.set_title(model_name)
    ax.set_ylabel('Vrai')
    ax.set_xlabel('Prédit')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=300)
plt.show()

# 3. Feature Importance (XGBoost)
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model_xgb.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (XGBoost)')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300)
plt.show()

# 4. Training History (Neural Networks)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss (Simple)')
axes[0].plot(history.history['val_loss'], label='Val Loss (Simple)')
axes[0].plot(history_adv.history['loss'], label='Train Loss (Avancé)')
axes[0].plot(history_adv.history['val_loss'], label='Val Loss (Avancé)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid()

# AUC
axes[1].plot(history.history['auc'], label='Train AUC (Simple)')
axes[1].plot(history.history['val_auc'], label='Val AUC (Simple)')
axes[1].plot(history_adv.history['auc'], label='Train AUC (Avancé)')
axes[1].plot(history_adv.history['val_auc'], label='Val AUC (Avancé)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC')
axes[1].set_title('Training AUC')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=300)
plt.show()
```

---

### **ÉTAPE 8 : Comparaison Finale**

```python
# Créer un tableau récapitulatif
comparison = pd.DataFrame({
    'Modèle': ['XGBoost', 'NN Simple', 'NN Avancé'],
    'ROC-AUC': [results_xgb['roc_auc'], results_nn['roc_auc'], results_nn_adv['roc_auc']],
    'Precision': [results_xgb['precision'], results_nn['precision'], results_nn_adv['precision']],
    'Recall': [results_xgb['recall'], results_nn['recall'], results_nn_adv['recall']],
    'F1-Score': [results_xgb['f1'], results_nn['f1'], results_nn_adv['f1']]
})

print("\nCOMPARAISON DES MODÈLES :")
print(comparison.to_string(index=False))
print(f"\n Meilleur modèle : {comparison.loc[comparison['ROC-AUC'].idxmax(), 'Modèle']}")

# Sauvegarder
comparison.to_csv('results/model_comparison.csv', index=False)
```

---

### **ÉTAPE 9 : Prédiction sur Cas Réels**

```python
# Exemple : nouvelle mutation inconnue
new_mutation = pd.DataFrame({
    'CADD_PHRED': [28],
    'CADD_RAW': [3.5],
    'SIFT': [0.01],
    'PolyPhen': [0.95],
    'AF_EXAC': [0.0001],
    'AF_TGP': [0.0002],
    'BLOSUM62': [-3]
})

# Normaliser
new_mutation_scaled = scaler.transform(new_mutation)

# Prédictions
pred_xgb = model_xgb.predict_proba(new_mutation_scaled)[:, 1][0]
pred_nn = model_nn.predict(new_mutation_scaled, verbose=0)[0, 0]
pred_nn_adv = model_nn_adv.predict(new_mutation_scaled, verbose=0)[0, 0]

print("\nPRÉDICTION POUR LA NOUVELLE MUTATION :")
print(f"Probabilité Pathogenic :")
print(f"  XGBoost      : {pred_xgb:.1%}")
print(f"  NN Simple    : {pred_nn:.1%}")
print(f"  NN Avancé    : {pred_nn_adv:.1%}")
print(f"  Moyenne      : {(pred_xgb + pred_nn + pred_nn_adv) / 3:.1%}")

if (pred_xgb + pred_nn + pred_nn_adv) / 3 > 0.7:
    print("\nVERDICT : Probablement PATHOGENIC (dangereux)")
    print("→ Recommandation : Faire des tests en laboratoire")
else:
    print("\nVERDICT : Probablement BENIGN (inoffensif)")
    print("→ Recommandation : Pas d'action urgente nécessaire")
```

---

---

## 📊 **PARTIE 7 : MÉTRIQUES DÉTAILLÉES**

### **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**

**Qu'est-ce que c'est ?**
Mesure la capacité du modèle à distinguer les 2 classes sur tout les seuils de décision.

**Interprétation :**
```
0.50 = Lancer une pièce (complètement nul)
0.60-0.70 = Faible
0.70-0.80 = Acceptable
0.80-0.90 = Bon
0.90-0.95 = Très bon
0.95+ = Excellent
```

**Pour ce projet :** Viser ≥ 0.80

---

### **Precision (Précision)**

**Qu'est-ce que c'est ?**
Parmi les mutations classées comme Pathogenic, combien sont réellement Pathogenic ?

```
Precision = TP / (TP + FP)

TP = True Positives (bien prédits comme Pathogenic)
FP = False Positives (mal prédits comme Pathogenic)

Exemple :
  Modèle prédit 100 mutations comme Pathogenic
  80 sont réellement Pathogenic
  20 sont réellement Benign
  → Precision = 80 / 100 = 80%
```

**Importace :** Élevée pour ce projet !
- Faux positif = opérer un patient pour rien
- À minimiser

---

### **Recall (Sensibilité)**

**Qu'est-ce que c'est ?**
Parmi les mutations réellement Pathogenic, combien are correctement identifiées ?

```
Recall = TP / (TP + FN)

TP = True Positives
FN = False Negatives (mal prédits comme Benign)

Exemple :
  Il y a 100 mutations réellement Pathogenic
  Le modèle en détecte 75
  Il en rate 25
  → Recall = 75 / 100 = 75%
```

**Importace :** Très élevée pour ce projet !
- Faux négatif = rater un cancer
- À minimiser absolument

---

### **F1-Score**

**Qu'est-ce que c'est ?**
Moyenne harmonique de Precision et Recall

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Équilibre entre les 2
```

**Utilité :** Bon compromis quand Precision et Recall importants

---

### **Matrice de Confusion**

```
                PRÉDIT
           Benign  Pathogenic
RÉEL Benign      TN         FP
     Pathogenic  FN         TP

TN = True Negatives  (bien prédit comme Benign)
FP = False Positives (mal prédit comme Pathogenic)
FN = False Negatives (mal prédit comme Benign)
TP = True Positives  (bien prédit comme Pathogenic)

Spécificité = TN / (TN + FP)  → "capacité à identifier Benign"
Sensibilité = TP / (TP + FN)  → "capacité à identifier Pathogenic"
```

---

## 🎓 **PARTIE 8 : RÉFÉRENCES & RESSOURCES**

### **Données**

- **ClinVar officiel :** https://www.ncbi.nlm.nih.gov/clinvar/
- **ClinVar sur Kaggle :** https://www.kaggle.com/kevinarvai/clinvar-conflicting
- **Format VCF expliqué :** https://en.wikipedia.org/wiki/Variant_Call_Format

### **Deep Learning**

- **TensorFlow/Keras doc :** https://www.tensorflow.org/api_docs
- **scikit-learn :** https://scikit-learn.org/
- **XGBoost :** https://xgboost.readthedocs.io/

### **Publications Similaires**

- Kircher et al. (2016) "A general framework for estimating the relative pathogenicity of human genetic variants" Nature Genetics
- Chen et al. (2018) "DeepVariant: A universal SNP and small-indel variant caller"
- Sundaram et al. (2021) "Predicting the clinical impact of human mutation with deep neural networks" Nature Medicine

---

## **CHECKLIST FINALE**

### **Code à Produire**

- [ ] `data_loader.py` - Charger et explorer les données
- [ ] `preprocessing.py` - Nettoyer et normaliser
- [ ] `models.py` - Définir architectures (XGBoost, NN simple, NN avancé)
- [ ] `train.py` - Entraîner tous les modèles
- [ ] `evaluate.py` - Évaluer et comparer
- [ ] `main.ipynb` - Notebook Jupyter complet (ou 4 notebooks séparés)

### **Résultats à Produire**

- [ ] Fichier CSV avec données nettoyées
- [ ] 3 modèles sauvegardés (.pkl ou .h5)
- [ ] Fichier JSON avec tous les métriques
- [ ] PNG des courbes ROC
- [ ] PNG des matrices de confusion
- [ ] PNG de feature importance
- [ ] PNG du training history

### **Documentation**

- [ ] README.md expliquant le projet
- [ ] Rapport technique (~15 pages)
- [ ] Code commenté
- [ ] requirements.txt avec dépendances

### **Présentation**

- [ ] Slides (10-15 diapos)
- [ ] Démo live du modèle
- [ ] Exemple de prédiction

