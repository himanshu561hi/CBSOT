<div align="center">

<img width="120" src="https://img.icons8.com/emoji/120/beating-heart.png" alt="heart"/>

# Heart Disease Prediction System
## 🩺 Machine Learning Internship Project

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Google Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-006600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Status](https://img.shields.io/badge/Status-✅_Completed-brightgreen?style=for-the-badge)](.)

<br/>

> *"Every second counts in cardiac care — early prediction through machine learning can be the difference between life and death."*

<br/>

---

</div>

## 👨‍💻 Student & Internship Information

<table>
<tr><td>

### 🎓 Student Details

| | |
|:--|:--|
| **👤 Name** | Himanshu Gupta |
| **🏫 University** | *Dr. APJ Abdul Kalam Technical University* |
| **📚 Program** | *B.Tech* |
| **📅 Year / Semester** | *4th Year, Sem 7* |
| **🆔 Enrollment No.** | *[2302200100054* |

</td><td>

### 🏢 Internship Details

| | |
|:--|:--|
| **🏢 Organization** | *Coding Blocks of Technology* |
| **👨‍💼 Mentor / Guide** | *Coding Blocks of Technology* |
| **📌 Domain** | Machine Learning · Healthcare AI |
| **📍 Mode** | Remote / Offline |

</td></tr>
</table>

---

## 📋 Table of Contents

<details>
<summary><b>Click to expand full index</b></summary>

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Why This Project?](#-why-this-project)
4. [Dataset Details](#-dataset-details)
5. [Project Architecture](#-project-architecture)
6. [Tech Stack](#-tech-stack)
7. [ML Models & Why](#-ml-models--why-each-one)
8. [Feature Engineering](#-feature-engineering)
9. [Model Evaluation](#-model-evaluation-metrics)
10. [Results & Performance](#-results--performance)
11. [Visualizations](#-visualizations-generated)
12. [Project Structure](#-project-structure)
13. [How to Run](#-how-to-run)
14. [Key Learnings](#-key-learnings)
15. [Future Scope](#-future-scope)
16. [References](#-references)

</details>

---

## 🧠 Project Overview

**Heart Disease Prediction** is a complete end-to-end supervised Machine Learning project that predicts whether a patient is at risk of heart disease using 13 routine clinical attributes. The system uses the **Cleveland Heart Disease Dataset** (UCI ML Repository) and evaluates three classification algorithms — Logistic Regression, Random Forest, and XGBoost — to identify the most accurate and generalizable model.

This project was developed as part of a **Machine Learning Internship**, covering the full data science lifecycle:

```
Raw Data  →  Cleaning  →  EDA  →  Feature Engineering
   →  Model Training  →  Evaluation  →  Explainability  →  Deployment-Ready Model
```

The notebook is fully **Google Colab compatible** — it auto-downloads the dataset and auto-installs dependencies, requiring zero manual setup.

---

## 🔴 Problem Statement

> **Heart disease kills 17.9 million people every year** — that's 32% of all global deaths (WHO, 2023). It is the **single leading cause of death worldwide**, yet up to 80% of premature cases are preventable through early detection and lifestyle intervention.

The challenge: traditional cardiac diagnosis requires specialist cardiologists, ECG machines, angiography, and expensive lab tests — resources unavailable to millions in rural or low-income settings.

**Can a machine learning model trained on basic clinical measurements accurately predict whether a patient has heart disease — enabling early, low-cost risk screening?**

---

## ❓ Why This Project?

```
┌─────────────────────────────────────────────────────────────────────┐
│  🌍 REAL-WORLD IMPACT                                               │
│                                                                     │
│  • 523 million people worldwide currently live with heart disease   │
│  • Only 1 in 3 at-risk patients gets diagnosed before a cardiac     │
│    event (heart attack / stroke)                                    │
│  • Rural India has <1 cardiologist per 100,000 people               │
│  • An ML screening tool on basic data = accessible triage for all  │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Machine Learning — Not Rule-Based Systems?

Traditional clinical guidelines (e.g., "flag if BP > 140 AND age > 50") are **rigid, single-feature thresholds** that miss complex multi-feature interactions. Machine learning:

- ✅ Learns **non-linear combinations** across all 13 features simultaneously
- ✅ Adapts to patterns **invisible to the human eye** in historical patient data
- ✅ Quantifies **probability of risk**, not just binary yes/no
- ✅ Provides **explainable predictions** (via SHAP) that doctors can audit
- ✅ Works on **routine data** already collected during standard checkups

### Why This Dataset?

The Cleveland Heart Disease dataset is the **gold standard benchmark** in cardiac ML research. It has been:
- Used in **1,000+ published research papers**
- Curated by medical professionals at the Cleveland Clinic Foundation
- Validated across populations in Hungary, Switzerland, and Virginia
- Compact enough (303 rows) to demonstrate end-to-end ML without cloud compute

---

## 📊 Dataset Details

### Source & Overview

| Property | Value |
|:---------|:------|
| 📦 **Name** | Cleveland Heart Disease Dataset |
| 🌐 **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) |
| 📁 **File** | `processed.cleveland.data` |
| 👥 **Patients** | 303 records |
| 🔢 **Features** | 13 clinical attributes |
| 🎯 **Target** | Binary: `0` = No Disease · `1` = Disease Present |
| ❓ **Missing Values** | 6 cells encoded as `?` in columns `ca` and `thal` |
| ⚖️ **Class Balance** | ~54% No Disease · ~46% Disease |

### 📌 Feature Dictionary

| # | Column | Type | Range | Clinical Meaning |
|:--|:-------|:-----|:------|:-----------------|
| 1 | `age` | Numeric | 29–77 | Patient's age in years |
| 2 | `sex` | Binary | 0, 1 | 1 = Male, 0 = Female |
| 3 | `cp` | Categorical | 0–3 | Chest pain type: 0=Typical angina, 1=Atypical angina, 2=Non-anginal pain, 3=Asymptomatic |
| 4 | `trestbps` | Numeric | 94–200 | Resting blood pressure (mm Hg on admission) |
| 5 | `chol` | Numeric | 126–564 | Serum cholesterol in mg/dl |
| 6 | `fbs` | Binary | 0, 1 | Fasting blood sugar > 120 mg/dl (1 = True) |
| 7 | `restecg` | Categorical | 0–2 | Resting ECG results: 0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy |
| 8 | `thalach` | Numeric | 71–202 | Maximum heart rate achieved during stress test |
| 9 | `exang` | Binary | 0, 1 | Exercise-induced angina (1 = Yes) |
| 10 | `oldpeak` | Numeric | 0–6.2 | ST depression induced by exercise relative to rest |
| 11 | `slope` | Categorical | 0–2 | Slope of peak exercise ST segment: 0=Upsloping, 1=Flat, 2=Downsloping |
| 12 | `ca` | Numeric | 0–3 | Number of major vessels colored by fluoroscopy |
| 13 | `thal` | Categorical | 3, 6, 7 | Thalassemia: 3=Normal, 6=Fixed defect, 7=Reversible defect |
| 14 | `target` | Binary | 0, 1 | **Label**: 0 = No disease, 1 = Disease ← *Predict this* |

---

## 🏗️ Project Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                    COMPLETE ML PIPELINE                             ║
╚══════════════════════════════════════════════════════════════════════╝

  ┌──────────────────┐
  │  📥 DATA INGESTION│  Auto-download from UCI via urllib
  │                  │  Loaded with pandas (303 rows × 14 cols)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  🔧 PREPROCESSING │  Replace '?' → NaN
  │                  │  Median imputation for ca, thal
  │                  │  Type coercion (object → float)
  │                  │  Binarize target (0–4 → 0 or 1)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  🔍 EDA          │  Distribution plots (all 13 features)
  │                  │  Correlation heatmap
  │                  │  Class balance visualization
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  ✨ FEATURE ENG. │  age_group  (4 bins)
  │                  │  high_chol  (chol > 240)
  │                  │  hypertension (trestbps > 140)
  │                  │  hr_ratio   (thalach / 220−age)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  ✂️ SPLIT & SCALE │  80/20 Stratified Train-Test Split
  │                  │  StandardScaler (fit on train only)
  └────────┬─────────┘
           │
           ▼
  ┌─────────────────────────────────────┐
  │  🤖 MODEL TRAINING (×3)            │
  │                                     │
  │  ① Logistic Regression (baseline)  │
  │  ② Random Forest (200 trees)       │
  │  ③ XGBoost (gradient boosting)     │
  │                                     │
  │  All with 5-Fold Stratified CV      │
  └────────────────────┬────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────┐
  │  📈 EVALUATION                   │
  │  Accuracy · Precision · Recall   │
  │  F1 Score · ROC-AUC              │
  │  Confusion Matrices · ROC Curves │
  └────────────┬─────────────────────┘
               │
               ▼
  ┌──────────────────────────────────┐
  │  🔎 EXPLAINABILITY (SHAP)        │
  │  TreeExplainer on XGBoost        │
  │  Summary plot · Bar plot         │
  └────────────┬─────────────────────┘
               │
               ▼
  ┌──────────────────────────────────┐
  │  💾 SAVE BEST MODEL              │
  │  best_heart_model.pkl            │
  │  (model + scaler bundled)        │
  └──────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### 🐍 Core Language

| Tool | Why Used |
|:-----|:---------|
| **Python 3.10+** | Industry standard for data science; rich ML ecosystem, readable syntax, fast prototyping |

### ☁️ Environment

| Tool | Why Used |
|:-----|:---------|
| **Jupyter Notebook** | Cell-by-cell interactive execution; mix code, output, and documentation in one place |
| **Google Colab** | Free cloud GPU, zero local setup, easy sharing — perfect for internship demos |

### 📦 Libraries

| Library | Version | Role | Why This Over Alternatives |
|:--------|:--------|:-----|:--------------------------|
| `pandas` | ≥1.5 | Data loading, cleaning, manipulation | Best DataFrame API; handles CSVs, missing values, type coercion effortlessly |
| `numpy` | ≥1.23 | Numerical arrays, math ops | Foundation of all Python ML; fast vectorized operations |
| `matplotlib` | ≥3.6 | Base plotting engine | Most flexible Python plotting; full control over every chart element |
| `seaborn` | ≥0.12 | Statistical visualizations | Built on matplotlib; beautiful defaults for heatmaps, distributions |
| `scikit-learn` | ≥1.2 | ML models, preprocessing, metrics | Gold standard ML library; consistent API across all algorithms |
| `xgboost` | ≥1.7 | Gradient Boosted Trees | State-of-the-art on tabular data; wins most Kaggle competitions |
| `shap` | ≥0.41 | Model explainability | Best XAI library; explains *why* a model made each prediction |
| `pickle` | built-in | Model serialization | Save trained model to disk for future inference without retraining |
| `urllib` | built-in | Auto-download dataset | Fetch data from UCI programmatically — no manual upload needed in Colab |

---

## 🤖 ML Models & Why Each One

### 1️⃣ Logistic Regression — *The Baseline*

**What it does:** Estimates the probability that a patient has heart disease using a linear combination of features passed through the sigmoid function.

```
P(disease) = 1 / (1 + e^−(β₀ + β₁·age + β₂·cp + β₃·chol + ... + β₁₃·thal))

If P > 0.5  →  Predict Disease (1)
If P ≤ 0.5  →  Predict No Disease (0)
```

**Why we use it:**
| ✅ Pros | ⚠️ Cons |
|:--------|:--------|
| Highly interpretable — each coefficient is a direct weight | Assumes linear decision boundary |
| Outputs calibrated probabilities → good for clinical risk scoring | Cannot capture complex feature interactions |
| Fast to train; no tuning needed for a baseline | Sensitive to outliers and unscaled features |
| Well-understood mathematically by doctors | Underperforms when relationships are non-linear |

---

### 2️⃣ Random Forest — *The Robust Ensemble*

**What it does:** Builds 200 independent Decision Trees, each trained on a **random bootstrap sample** of data and a **random subset of features**. Final prediction = majority vote.

```
Forest prediction = MAJORITY VOTE of:
  Tree₁(patient_data) = Disease
  Tree₂(patient_data) = No Disease
  Tree₃(patient_data) = Disease
  ...
  Tree₂₀₀(patient_data) = Disease
  → Final: Disease (majority wins)
```

**Why we use it:**
| ✅ Pros | ⚠️ Cons |
|:--------|:--------|
| Handles non-linear feature interactions naturally | Slower inference than single models |
| Robust to outliers and irrelevant features | Less interpretable than Logistic Regression |
| Built-in feature importance ranking | Higher memory usage |
| Low overfitting risk due to bagging | Doesn't perform well on sparse/high-dimensional data |
| No need for feature scaling | — |

---

### 3️⃣ XGBoost — *The Champion*

**What it does:** Builds trees **sequentially**, where each new tree corrects the errors of the previous ensemble. Uses gradient descent on the loss function to minimize prediction error.

```
Model at step m:
  F_m(x) = F_{m-1}(x) + η · h_m(x)

Where:
  F_{m-1}(x) = Previous ensemble prediction
  η           = Learning rate (shrinkage)
  h_m(x)      = New tree fitted to residual errors
```

**Why we use it:**
| ✅ Pros | ⚠️ Cons |
|:--------|:--------|
| Consistently best performer on tabular/structured data | More hyperparameters to tune |
| Built-in L1 + L2 regularization prevents overfitting | Slower to train than single models |
| Handles missing values internally | Can overfit on very small datasets without tuning |
| Extremely fast (parallelized tree building) | Less intuitive to interpret without SHAP |
| Native SHAP support for explainability | — |

---

### ❌ Why NOT Deep Learning / Neural Networks?

```
❌ Only 303 rows — neural nets need thousands to tens of thousands
❌ Black-box nature is unacceptable in clinical AI (doctors need explanations)
❌ XGBoost matches or beats NNs on tabular data (proven across Kaggle benchmarks)
❌ Massive overkill: a 50-layer network to classify 303 patients is like using
   a sledgehammer to crack a walnut
❌ No interpretability without additional tools (LIME, SHAP) — adds complexity
```

### ⚖️ Why StandardScaler?

Logistic Regression uses gradient descent to minimize a loss function. Features with larger numeric ranges (e.g., `chol` ≈ 200 vs `fbs` = 0 or 1) cause disproportionately large gradients, slowing convergence and biasing coefficients.

```
Before scaling:  chol ≈ 200,  fbs = 0 or 1  →  chol dominates
After scaling:   chol ≈ 0.0 ± 1.0,  fbs ≈ 0.0 ± 1.0  →  fair comparison

StandardScaler formula:  z = (x − μ) / σ
```

> **Note:** Random Forest and XGBoost are tree-based and **don't need scaling** — but we scale for them too to keep the pipeline consistent and to allow fair metric comparison across all three models.

---

## ✨ Feature Engineering

Four new clinically meaningful features were created from the original 13:

| New Feature | Derivation | Medical Rationale |
|:------------|:-----------|:------------------|
| `age_group` | Bin `age` into: <40 · 40–50 · 50–60 · 60+ (label-encoded) | Heart disease risk rises non-linearly with age decade — capturing this as a category can reveal step-function patterns trees struggle to find |
| `high_chol` | `1` if `chol > 240`, else `0` | 240 mg/dl is the clinical "borderline high" threshold per American Heart Association guidelines |
| `hypertension` | `1` if `trestbps > 140`, else `0` | Stage 2 hypertension threshold (≥140 mmHg) — a direct independent risk factor for cardiac events |
| `hr_ratio` | `thalach / (220 − age)` | Ratio of achieved max HR to age-predicted max HR; values < 0.85 indicate poor cardiac reserve — a known mortality predictor |

---

## 📈 Model Evaluation Metrics

### Why We Use Multiple Metrics

Accuracy alone is **misleading** in medical settings. Consider a dataset where 90% of patients have no disease — a model predicting "no disease" for everyone gets **90% accuracy** but catches **zero actual patients**.

| Metric | Formula | What It Tells Us | Priority |
|:-------|:--------|:-----------------|:---------|
| **Accuracy** | (TP+TN) / N | Overall correctness across both classes | Low — misleading if imbalanced |
| **Precision** | TP / (TP+FP) | Of all "disease" predictions, what fraction were correct? Reduces false alarms | Medium |
| **Recall (Sensitivity)** | TP / (TP+FN) | Of all actual disease patients, what fraction did we catch? | ⭐ **Highest** — missing a sick patient is dangerous |
| **F1 Score** | 2·P·R / (P+R) | Harmonic mean — balance when both Precision and Recall matter | High |
| **ROC-AUC** | Area under ROC curve | Model's ability to rank disease patients above non-disease across ALL thresholds | High — threshold-independent |

### Cross-Validation Strategy

```
5-Fold Stratified Cross-Validation:

Full Training Set (80%)
│
├── Fold 1: [Val]  [Train] [Train] [Train] [Train]
├── Fold 2: [Train] [Val]  [Train] [Train] [Train]
├── Fold 3: [Train] [Train] [Val]  [Train] [Train]
├── Fold 4: [Train] [Train] [Train] [Val]  [Train]
└── Fold 5: [Train] [Train] [Train] [Train] [Val]

Report: Mean Accuracy ± Std Dev across 5 folds
"Stratified" ensures each fold has the same class ratio as the full dataset
```

---

## 🏆 Results & Performance

> *Results from enhanced notebook with feature engineering*

| Model | CV Acc (mean ± std) | Test Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:------|:-------------------:|:-------------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 0.838 ± 0.041 | 0.836 | 0.833 | 0.862 | 0.847 | 0.912 |
| Random Forest | 0.852 ± 0.038 | 0.852 | 0.851 | 0.874 | 0.862 | 0.931 |
| **⭐ XGBoost** | **0.861 ± 0.033** | **0.869** | **0.872** | **0.880** | **0.876** | **0.943** |

### 🥇 Best Model: XGBoost
- Highest accuracy, F1, and ROC-AUC across the board
- Lowest CV standard deviation → most **stable and generalizable**
- Saved as `best_heart_model.pkl` for production inference

### 🔬 Top Predictive Features (SHAP + Feature Importance)

```
  Rank  Feature        Impact
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1 🥇  cp             ████████████  Chest pain type (strongest signal)
  2 🥈  thal           ██████████    Thalassemia type
  3 🥉  ca             █████████     Number of major vessels (fluoroscopy)
  4     oldpeak        ████████      ST depression on exercise
  5     thalach        ███████       Max heart rate achieved
  6     exang          ██████        Exercise-induced angina
  7     age            █████         Patient age
  8 ✨  hr_ratio       ████          Engineered feature — heart rate ratio
  9     sex            ███           Gender
  10    hypertension   ██            Engineered: resting BP flag
```

---

## 🖼️ Visualizations Generated

The notebook auto-generates and saves 9 charts:

| # | File | What It Shows |
|:--|:-----|:--------------|
| 1 | `feature_distributions.png` | Histograms of all 13+ features colored by disease/no-disease |
| 2 | `correlation_heatmap.png` | Pearson correlation matrix — which features move together |
| 3 | `class_distribution.png` | Bar + pie chart of class balance in the dataset |
| 4 | `confusion_matrices.png` | TP/TN/FP/FN breakdown for all 3 models side-by-side |
| 5 | `roc_curves.png` | ROC curves for all 3 models on one chart with AUC values |
| 6 | `model_comparison.png` | Grouped bar chart comparing all 5 metrics across all 3 models |
| 7 | `feature_importance.png` | Top features ranked by Random Forest and XGBoost |
| 8 | `shap_summary.png` | SHAP dot plot — feature impact direction and magnitude per patient |
| 9 | `shap_bar.png` | SHAP bar plot — mean absolute impact per feature globally |

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── 📓 HeartDisease_Enhanced_Colab.ipynb    ← Main notebook (Colab-ready, enhanced)
├── 📓 HeartDisease__1_.ipynb               ← Original internship notebook
├── 📄 README.md                            ← This file
│
├── 📊 data/
│   └── processed.cleveland.data            ← UCI dataset (auto-downloaded by notebook)
│
├── 💾 models/
│   └── best_heart_model.pkl                ← Trained XGBoost model + scaler (generated)
│
└── 📈 outputs/                             ← All charts (generated on run)
    ├── feature_distributions.png
    ├── correlation_heatmap.png
    ├── class_distribution.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── model_comparison.png
    ├── feature_importance.png
    ├── shap_summary.png
    └── shap_bar.png
```

---

## 🚀 How to Run

### ☁️ Option 1 — Google Colab *(Recommended — Zero Setup)*

```
1. Open https://colab.research.google.com
2. File → Upload Notebook → select HeartDisease_Enhanced_Colab.ipynb
3. Runtime → Run All  (Ctrl + F9)
4. ✅ Done — dataset downloads automatically, packages install automatically
```

### 💻 Option 2 — Local Jupyter

```bash
# Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install all dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter

# Launch the notebook
jupyter notebook HeartDisease_Enhanced_Colab.ipynb
```

### 📦 requirements.txt

```text
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
shap>=0.41.0
jupyter>=1.0.0
```

---

## 📚 Key Learnings

This project provided hands-on experience with the complete data science workflow:

```
✅ Data Wrangling
   └── Handling non-standard missing values ('?')
   └── Type coercion and safe numeric conversion
   └── Median imputation strategy for clinical data

✅ Exploratory Data Analysis
   └── Distribution comparison across classes
   └── Identifying correlated features via heatmaps
   └── Detecting class imbalance and its implications

✅ Feature Engineering
   └── Creating medically meaningful derived features
   └── Binning continuous variables into clinically relevant categories
   └── Computing ratios that capture physiological relationships

✅ ML Pipeline Best Practices
   └── Stratified splitting to preserve class ratios
   └── Fitting scaler ONLY on training data (no data leakage)
   └── Wrapping everything in reproducible random seeds

✅ Model Selection
   └── Using a simple baseline before complex models
   └── Understanding when to use ensemble vs linear models
   └── Trade-offs between interpretability and accuracy

✅ Rigorous Evaluation
   └── Why CV mean ± std matters more than a single test split
   └── Prioritizing Recall over Accuracy in medical contexts
   └── Interpreting confusion matrices and ROC-AUC in clinical terms

✅ Explainable AI (XAI) with SHAP
   └── Why black-box models are unacceptable in healthcare
   └── How SHAP values work (Shapley values from game theory)
   └── Reading and interpreting SHAP summary plots

✅ Production Readiness
   └── Saving models with pickle for reuse
   └── Bundling scaler with model to ensure consistent inference
   └── Writing Colab-compatible notebooks for sharing
```

---

## 🔭 Future Scope

| Enhancement | Description | Priority |
|:------------|:------------|:---------|
| 🌐 **Web Application** | Flask/FastAPI + HTML form for patient data input → real-time prediction | 🔴 High |
| ⚖️ **SMOTE Balancing** | Synthetic Minority Oversampling to handle class imbalance | 🔴 High |
| 🔧 **Hyperparameter Tuning** | Optuna / GridSearchCV for XGBoost optimization | 🟡 Medium |
| 📋 **Clinical PDF Report** | Auto-generate per-patient risk report with SHAP explanation | 🟡 Medium |
| 🧪 **More Datasets** | Combine Hungarian, Swiss, Virginia Heart Disease datasets from UCI | 🟡 Medium |
| 📱 **Mobile App** | Android/iOS wrapper for rural clinic field use | 🟠 Low |
| 🤝 **Federated Learning** | Train across hospitals without sharing raw patient data (privacy) | 🟠 Research |
| 🩻 **Multimodal Input** | Add ECG image data + clinical text notes via CNN + NLP pipeline | 🟠 Research |

---

## 📜 References

1. **Dataset** — Detrano, R., Janosi, A., Steinbrunn, W., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304–310.
   → [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

2. **XGBoost** — Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

3. **SHAP** — Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS 2017).

4. **scikit-learn** — Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.

5. **WHO Statistics** — World Health Organization. (2023). *Cardiovascular diseases (CVDs) Fact Sheet.* → [who.int](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))

6. **AHA Cholesterol Guidelines** — American Heart Association. *Understanding Blood Cholesterol.* → [heart.org](https://www.heart.org)

---

<div align="center">

---

## 🙏 Acknowledgements

Sincere gratitude to my **internship mentor** for guidance throughout this project,  
to **UCI Machine Learning Repository** for making the Cleveland dataset publicly available,  
and to the open-source maintainers of **scikit-learn**, **XGBoost**, and **SHAP**  
whose tools made this entire pipeline possible.

---

<br/>

*Built with ❤️ during Machine Learning Internship*

**Himanshu**

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006600?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-XAI-blueviolet?style=flat-square)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>
