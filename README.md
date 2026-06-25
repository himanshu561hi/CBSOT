<div align="center">

# ❤️ Heart Disease Prediction System
### Machine Learning Internship Project

<img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-Gradient_Boost-006600?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>

<br/>

> **"Leveraging the power of Machine Learning to assist early detection of heart disease — because early prediction saves lives."**

<br/>

---

</div>

## 👨‍🎓 Internship Details

|
 Field 
|
 Details 
|
|
:------
|
:--------
|
|
 🏢 
**
Organization
**
|
*
[Your Company / Institute Name]
*
|
|
 🎓 
**
Intern Name
**
|
*
[Your Full Name]
*
|
|
 🆔 
**
Student ID / Roll No
**
|
*
[Your ID]
*
|
|
 📚 
**
Course / Degree
**
|
*
[B.Tech / MCA / BSc CS — Year]
*
|
|
 🏫 
**
University / College
**
|
*
[Your University Name]
*
|
|
 📅 
**
Internship Duration
**
|
*
[Start Date] → [End Date]
*
|
|
 👨‍💼 
**
Mentor / Guide
**
|
*
[Mentor Name]
*
|
|
 📌 
**
Domain
**
|
 Machine Learning · Data Science · Healthcare AI 
|
|
 📂 
**
Project Title
**
|
 Heart Disease Prediction using Machine Learning 
|

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Why This Project?](#-why-this-project)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Feature Engineering](#-feature-engineering)
- [Tech Stack](#-tech-stack)
- [ML Models Used](#-ml-models-used)
- [Why These Algorithms?](#-why-these-algorithms)
- [Model Evaluation](#-model-evaluation)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Visualizations](#-visualizations)
- [Learnings & Takeaways](#-learnings--takeaways)
- [Future Scope](#-future-scope)
- [References](#-references)

---

## 🧠 About the Project

**Heart Disease Prediction** is a supervised binary classification project that predicts whether a patient is at risk of heart disease based on clinical diagnostic data. The model takes 13 medical attributes as input — such as age, cholesterol level, chest pain type, and resting blood pressure — and outputs a prediction: **disease present (1)** or **disease absent (0)**.

This project was built as part of a Machine Learning internship to demonstrate an end-to-end data science pipeline — from raw data ingestion and exploratory analysis to model training, evaluation, explainability (SHAP), and deployment-ready model saving.

### 🔍 Problem Statement

> Heart disease is the **#1 cause of death globally**, responsible for **17.9 million deaths per year** (WHO, 2023). A significant number of these deaths are preventable if diagnosed early. Traditional diagnosis requires expensive tests and specialist visits. Can we use machine learning on routine clinical data to **flag at-risk patients early**?

---

## ❓ Why This Project?

```
💡 Problem      →  Heart disease is often asymptomatic until it's too late
🩺 Current Gap  →  Diagnosis is expensive, slow, and requires specialists
🤖 ML Solution  →  Train a model on known cases to predict risk from basic tests
🎯 Impact       →  Faster triage, lower cost, accessible to rural/under-resourced clinics
```

Machine learning is uniquely positioned to solve this because:
- ✅ Clinical data (cholesterol, BP, ECG) is **routinely collected** anyway
- ✅ Patterns in historical data can be learned and **generalized to new patients**
- ✅ A prediction tool can act as a **second opinion** for doctors in low-resource settings
- ✅ Binary classification is **well-suited** for this yes/no diagnostic question

---

## 📊 Dataset

### Cleveland Heart Disease Dataset (UCI ML Repository)

|
 Property 
|
 Value 
|
|
:---------
|
:------
|
|
 📦 
**
Source
**
|
[
UCI Machine Learning Repository
](
https://archive.ics.uci.edu/ml/datasets/heart+disease
)
|
|
 📁 
**
File
**
|
`processed.cleveland.data`
|
|
 📏 
**
Rows
**
|
 303 patients 
|
|
 🔢 
**
Columns
**
|
 14 (13 features + 1 target) 
|
|
 🎯 
**
Target
**
|
 0 = No Disease · 1 = Disease Present 
|
|
 ❓ 
**
Missing Values
**
|
 Present (encoded as 
`?`
) — handled with median imputation 
|

### 📌 Features Explained

| # | Feature | Type | Description |
|:--|:--------|:-----|:------------|
| 1 | `age` | Numeric | Age of the patient (years) |
| 2 | `sex` | Binary | Sex: 1 = Male, 0 = Female |
| 3 | `cp` | Categorical | Chest pain type (0–3): 0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic |
| 4 | `trestbps` | Numeric | Resting blood pressure (mm Hg) |
| 5 | `chol` | Numeric | Serum cholesterol (mg/dl) |
| 6 | `fbs` | Binary | Fasting blood sugar > 120 mg/dl (1=True) |
| 7 | `restecg` | Categorical | Resting ECG results (0–2) |
| 8 | `thalach` | Numeric | Maximum heart rate achieved |
| 9 | `exang` | Binary | Exercise-induced angina (1=Yes) |
| 10 | `oldpeak` | Numeric | ST depression induced by exercise vs rest |
| 11 | `slope` | Categorical | Slope of peak exercise ST segment (0–2) |
| 12 | `ca` | Numeric | Number of major vessels colored by fluoroscopy (0–3) |
| 13 | `thal` | Categorical | Thalassemia: 3=Normal, 6=Fixed defect, 7=Reversible defect |
| 14 | `target` | Binary | **Diagnosis**: 0 = No disease, 1 = Disease ← *Label* |

---

## 🔄 Project Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    END-TO-END ML PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

  📥 DATA COLLECTION          📊 EXPLORATORY ANALYSIS
  ─────────────────           ───────────────────────
  UCI Repository        →     Distribution plots
  Auto-download               Correlation heatmap
  via urllib                  Class balance check
        │                           │
        ▼                           ▼
  🔧 PREPROCESSING            ✨ FEATURE ENGINEERING
  ─────────────────           ──────────────────────
  Handle '?' values     →     age_group (binned)
  Median imputation           high_chol flag
  Type conversion             hypertension flag
  Binary target               hr_ratio (computed)
        │                           │
        └──────────┬────────────────┘
                   ▼
           ✂️ TRAIN / TEST SPLIT
           ─────────────────────
           80% Train · 20% Test
           Stratified (balanced)
                   │
                   ▼
           📐 FEATURE SCALING
           ──────────────────
           StandardScaler
           (fit on train, transform both)
                   │
                   ▼
         🤖 MODEL TRAINING (×3)
         ────────────────────────────────────────────
         Logistic Regression | Random Forest | XGBoost
         5-Fold Stratified Cross-Validation on each
                   │
                   ▼
          📈 EVALUATION & COMPARISON
          ────────────────────────────────────────────
          Accuracy · Precision · Recall · F1 · ROC-AUC
          Confusion Matrices · ROC Curves
                   │
                   ▼
          🔎 EXPLAINABILITY (SHAP)
          ─────────────────────────
          Feature impact per prediction
          Summary & bar plots
                   │
                   ▼
          💾 SAVE BEST MODEL
          ─────────────────────────
          best_heart_model.pkl
          (model + scaler bundled)
```

---

## ✨ Feature Engineering

Four new features were engineered on top of the original 13 to improve model signal:

|
 New Feature 
|
 Formula / Logic 
|
 Medical Rationale 
|
|
:------------
|
:----------------
|
:------------------
|
|
`age_group`
|
 Binned: <40, 40–50, 50–60, 60+ 
|
 Heart disease risk increases sharply with age brackets 
|
|
`high_chol`
|
`chol > 240`
 → 1 else 0 
|
 Cholesterol > 240 mg/dl is clinically flagged as high risk 
|
|
`hypertension`
|
`trestbps > 140`
 → 1 else 0 
|
 Resting BP > 140 mmHg is the hypertension threshold 
|
|
`hr_ratio`
|
`thalach / (220 − age)`
|
 Ratio of achieved HR to predicted max; low ratio = poor cardiac fitness 
|

---

## 🛠️ Tech Stack

### Languages & Environment

|
 Tool 
|
 Purpose 
|
|
:-----
|
:--------
|
|
 🐍 
**
Python 3.10+
**
|
 Core programming language 
|
|
 📓 
**
Jupyter Notebook
**
|
 Interactive development environment 
|
|
 ☁️ 
**
Google Colab
**
|
 Cloud-based GPU-accelerated runtime 
|

### Libraries & Frameworks

|
 Library 
|
 Version 
|
 Role 
|
|
:--------
|
:--------
|
:-----
|
|
`pandas`
|
 ≥1.5 
|
 Data loading, cleaning, manipulation 
|
|
`numpy`
|
 ≥1.23 
|
 Numerical operations, array handling 
|
|
`matplotlib`
|
 ≥3.6 
|
 Base plotting and chart rendering 
|
|
`seaborn`
|
 ≥0.12 
|
 Statistical visualizations, heatmaps 
|
|
`scikit-learn`
|
 ≥1.2 
|
 ML models, preprocessing, evaluation metrics 
|
|
`xgboost`
|
 ≥1.7 
|
 Gradient Boosted Trees classifier 
|
|
`shap`
|
 ≥0.41 
|
 Model explainability (SHAP values) 
|
|
`pickle`
|
 built-in 
|
 Model serialization and saving 
|
|
`urllib`
|
 built-in 
|
 Automatic dataset download 
|

### Why Python?
Python dominates the data science landscape for good reason:
- 🔬 Rich ecosystem of mature ML libraries (`sklearn`, `xgboost`, `shap`)
- 📊 Best-in-class visualization tools (`matplotlib`, `seaborn`)
- 🔁 Rapid prototyping with Jupyter notebooks
- 🌍 Largest open-source ML community worldwide

---

## 🤖 ML Models Used

### 1. 📐 Logistic Regression *(Baseline)*

Logistic Regression estimates the probability that a given input belongs to class 1 (disease) using the **sigmoid function**:

```
P(y=1 | X) = 1 / (1 + e^(−(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)))
```

- **Type:** Linear probabilistic classifier
- **Output:** Probability score between 0 and 1
- **Decision boundary:** Threshold at 0.5

### 2. 🌲 Random Forest *(Ensemble)*

Random Forest builds **N independent Decision Trees** on random subsets of the data (bootstrap samples) and random subsets of features, then **aggregates predictions** by majority vote.

```
Final Prediction = MODE( Tree₁(x), Tree₂(x), ..., TreeN(x) )
```

- **Type:** Bagging ensemble
- **Trees:** 200 estimators
- **Key strength:** Reduces overfitting via variance averaging

### 3. ⚡ XGBoost *(Boosting)*

XGBoost builds trees **sequentially**, where each new tree corrects the errors of the previous ensemble using gradient descent on the loss function.

```
F_m(x) = F_{m-1}(x) + η · h_m(x)
         (previous model + learning_rate × new weak learner)
```

- **Type:** Gradient Boosted Trees
- **Regularization:** L1 (Lasso) + L2 (Ridge) built-in
- **Key strength:** State-of-the-art on tabular data

---

## ❓ Why These Algorithms?

### Why Logistic Regression?
```
✅ Interpretable coefficients → each weight shows feature impact
✅ Fast to train, no hyperparameter tuning needed as a baseline
✅ Works well when features are linearly separable
✅ Outputs calibrated probabilities → useful for clinical risk scores
⚠️ Cannot capture complex non-linear interactions
```

### Why Random Forest?
```
✅ Handles non-linear relationships naturally (tree splits)
✅ Robust to outliers and noisy features
✅ Built-in feature importance ranking
✅ No need for feature scaling
✅ Low risk of overfitting due to bagging + feature randomness
⚠️ Less interpretable than Logistic Regression (black box ensemble)
```

### Why XGBoost?
```
✅ Consistently top performer on structured/tabular datasets
✅ Handles missing values internally
✅ Built-in L1/L2 regularization prevents overfitting
✅ Extremely fast (parallelized tree construction)
✅ Works great with small-to-medium datasets like Cleveland
⚠️ More hyperparameters to tune than simpler models
```

### Why NOT Deep Learning (Neural Networks)?
```
❌ Dataset has only 303 rows → deep learning needs thousands
❌ Neural nets are black boxes → unacceptable in clinical AI
❌ No significant accuracy gain over XGBoost on tabular data
❌ Overkill in compute cost for a problem this size
```

### Why StandardScaler?
Logistic Regression and other distance-sensitive algorithms penalize features with large numeric ranges (e.g., `chol` ≈ 200 vs `fbs` = 0 or 1). `StandardScaler` transforms every feature to **zero mean and unit variance** so no feature dominates due to its scale.

---

## 📈 Model Evaluation

### Metrics Used & Why

|
 Metric 
|
 Formula 
|
 Why It Matters Here 
|
|
:-------
|
:--------
|
:--------------------
|
|
**
Accuracy
**
|
 (TP+TN) / Total 
|
 Overall correctness — but misleading on imbalanced data 
|
|
**
Precision
**
|
 TP / (TP+FP) 
|
 Of all "disease" predictions, how many are real? 
|
|
**
Recall
**
|
 TP / (TP+FN) 
|
 Of all actual disease cases, how many did we catch? ⚠️ 
*
Critical in medicine
*
|
|
**
F1 Score
**
|
 2 × (P×R)/(P+R) 
|
 Harmonic mean — balances Precision and Recall 
|
|
**
ROC-AUC
**
|
 Area under ROC curve 
|
 Model's ability to discriminate disease vs no disease at all thresholds 
|

> ⚠️ **In medical diagnosis, Recall (Sensitivity) is the most critical metric.**  
> A **False Negative** (missing a real disease case) is far more dangerous than a **False Positive** (unnecessary follow-up).

### Evaluation Methods
- **Train/Test Split:** 80% / 20% with stratification
- **Cross-Validation:** 5-Fold Stratified CV (reports mean ± std accuracy)
- **Confusion Matrix:** Per-class breakdown of TP, TN, FP, FN
- **ROC Curve:** Plotted for all 3 models on the same chart

---

## 🏆 Key Results

> *(Values are representative; actual output may vary slightly by run)*

|
 Model 
|
 CV Accuracy 
|
 Test Accuracy 
|
 Precision 
|
 Recall 
|
 F1 Score 
|
 ROC-AUC 
|
|
:------
|
:-----------:
|
:-------------:
|
:---------:
|
:------:
|
:--------:
|
:-------:
|
|
 Logistic Regression 
|
 ~0.838 
|
 ~0.836 
|
 ~0.83 
|
 ~0.86 
|
 ~0.84 
|
 ~0.91 
|
|
 Random Forest 
|
 ~0.852 
|
 ~0.852 
|
 ~0.85 
|
 ~0.87 
|
 ~0.86 
|
 ~0.93 
|
|
**
XGBoost
**
|
**
~0.861
**
|
**
~0.869
**
|
**
~0.87
**
|
**
~0.88
**
|
**
~0.87
**
|
**
~0.94
**
|

🥇 **Best Model: XGBoost** — highest ROC-AUC, best generalization, saved to `best_heart_model.pkl`

### Top Predictive Features (from SHAP & Random Forest)
```
1. 🏆  cp          — Chest pain type (strongest predictor)
2. 🥈  thal        — Thalassemia type
3. 🥉  ca          — Number of major vessels (fluoroscopy)
4.     oldpeak     — ST depression on exercise
5.     thalach     — Max heart rate achieved
6.     exang       — Exercise-induced angina
7.     age         — Patient age
8.     hr_ratio    — Engineered: heart rate ratio (new feature ✨)
```

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── 📓 HeartDisease_Enhanced_Colab.ipynb   ← Main notebook (Colab-ready)
├── 📓 HeartDisease__1_.ipynb              ← Original notebook
├── 📄 README.md                           ← This file
│
├── 📊 Data/
│   └── processed.cleveland.data           ← UCI dataset (auto-downloaded)
│
├── 💾 Models/
│   └── best_heart_model.pkl               ← Saved XGBoost model + scaler
│
└── 📈 Outputs/  (generated on run)
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

### ▶️ Option 1 — Google Colab (Recommended, Zero Setup)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload Notebook**
3. Upload `HeartDisease_Enhanced_Colab.ipynb`
4. Click **Runtime → Run All** (`Ctrl + F9`)
5. ✅ The notebook auto-downloads the dataset and installs all packages

### 💻 Option 2 — Local Jupyter

```bash
# 1. Clone / download the project
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter

# 4. Launch Jupyter
jupyter notebook HeartDisease_Enhanced_Colab.ipynb
```

### 📦 Requirements

```txt
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

## 📊 Visualizations

The notebook generates the following charts automatically:

|
 Chart 
|
 Description 
|
|
:------
|
:------------
|
|
 📉 
**
Feature Distributions
**
|
 Histograms of all 13 features split by disease/no-disease 
|
|
 🔥 
**
Correlation Heatmap
**
|
 Pairwise Pearson correlation between all features 
|
|
 🥧 
**
Class Balance
**
|
 Bar + pie chart of target class distribution 
|
|
 🟦 
**
Confusion Matrices
**
|
 3 side-by-side matrices for all models 
|
|
 📈 
**
ROC Curves
**
|
 All 3 models on one chart with AUC scores 
|
|
 📊 
**
Model Comparison
**
|
 Grouped bar chart across all 5 metrics 
|
|
 🌲 
**
Feature Importance
**
|
 Top features ranked by RF and XGBoost 
|
|
 🔵 
**
SHAP Summary Plot
**
|
 Per-feature impact on each prediction (dot plot) 
|
|
 📊 
**
SHAP Bar Plot
**
|
 Mean absolute SHAP value per feature 
|

---

## 📚 Learnings & Takeaways

Through this internship project, the following concepts were applied hands-on:

```
📌 Data Wrangling        → Handling '?' missing values, type coercion, median imputation
📌 EDA                   → Visualizing distributions, correlations, and class imbalance
📌 Feature Engineering   → Creating clinically meaningful derived features
📌 ML Pipeline           → Proper train/test split with stratification
📌 Scaling               → Understanding why and when to scale features
📌 Multi-model Training  → Comparing Logistic Regression, RF, and XGBoost
📌 Cross-Validation      → Evaluating model stability with k-fold CV
📌 Evaluation Metrics    → Going beyond accuracy to Recall, F1, and ROC-AUC
📌 Explainability (XAI)  → Using SHAP to make black-box models interpretable
📌 Model Persistence     → Saving models with pickle for future inference
📌 Colab Compatibility   → Adapting notebooks for cloud execution environments
```

---

## 🔭 Future Scope

|
 Enhancement 
|
 Description 
|
|
:------------
|
:------------
|
|
 🌐 
**
Web App
**
|
 Deploy as a Flask/FastAPI app with a patient input form 
|
|
 📱 
**
Mobile App
**
|
 Android/iOS app for field use in rural clinics 
|
|
 ⚖️ 
**
Class Balancing
**
|
 Apply SMOTE oversampling to handle class imbalance 
|
|
 🔧 
**
Hyperparameter Tuning
**
|
 Grid Search / Bayesian Optimization for XGBoost 
|
|
 🧪 
**
More Datasets
**
|
 Train on Hungarian, Swiss, Virginia Heart Disease datasets (UCI) 
|
|
 🤝 
**
Federated Learning
**
|
 Train across multiple hospitals without sharing patient data 
|
|
 📋 
**
Clinical Report
**
|
 Auto-generate a PDF risk report for each patient prediction 
|
|
 🩺 
**
Doctor Dashboard
**
|
 Real-time prediction dashboard for hospital use 
|

---

## 📜 References

1. **Dataset:** Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* — [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
2. **scikit-learn Documentation** — [scikit-learn.org](https://scikit-learn.org)
3. **XGBoost Paper:** Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
4. **SHAP:** Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
5. **WHO Heart Disease Statistics** — [who.int/cardiovascular_diseases](https://www.who.int/health-topics/cardiovascular-diseases)

---

<div align="center">

## 🙏 Acknowledgements

Special thanks to my **internship mentor** for constant guidance,  
to **UCI ML Repository** for making this dataset publicly available,  
and to the **open-source community** behind scikit-learn, XGBoost, and SHAP.

---

*Made with ❤️ during my Machine Learning Internship*

**[Your Name] · [Your College] · [Year]**

<img src="https://img.shields.io/badge/Built%20with-Python-blue?style=flat-square&logo=python"/>
<img src="https://img.shields.io/badge/Domain-Healthcare%20AI-red?style=flat-square"/>
<img src="https://img.shields.io/badge/Type-Internship%20Project-purple?style=flat-square"/>

</div>
Done
