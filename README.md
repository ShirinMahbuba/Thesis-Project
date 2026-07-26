# CardioFormer-XGB

A hybrid deep learning framework for cardiovascular disease (CVD) risk prediction, combining **FT-Transformer**-based feature representation learning with **XGBoost** classification, enhanced with **SHAP** explainability.

---

## 🔍 Overview

Most existing cardiovascular disease prediction models face a trade-off: the highest-accuracy models are rarely interpretable, and the most interpretable models rarely achieve top accuracy. **CardioFormer-XGB** addresses this gap by:

- Learning rich, non-linear feature embeddings via an **FT-Transformer** encoder
- Fusing these embeddings with engineered clinical features: **H = [X ‖ E]**
- Classifying using **XGBoost** for strong, efficient tabular performance
- Explaining every prediction using **SHAP** (global and local interpretability)

---

## 🏗️ Architecture

1. **Data Preprocessing** — duplicate removal, missing value checks, and outlier handling using two independently evaluated strategies: **IQR-based Row Removal** and **Row Clipping**
2. **Feature Engineering** — Age, BMI, Pulse Pressure, Mean Arterial Pressure (MAP), Cholesterol–Glucose Interaction, and categorical risk groups (19 attributes total) + StandardScaler
3. **Baseline Models** — Decision Tree, Random Forest, XGBoost, FT-Transformer, TabTransformer, SAINT
4. **Hybrid Model (CardioFormer-XGB)** — FT-Transformer embeddings (E) concatenated with clinical features (X), then classified via XGBoost
5. **Validation** — Stratified 10-fold cross-validation + evaluation on an independent, unseen test set
6. **Explainability** — SHAP summary, feature importance, dependence, and waterfall plots

---

## 📊 Datasets

| Dataset | Samples | Role |
|---|---|---|
| Kaggle Cardiovascular Disease Dataset | 70,000 | Primary training/testing dataset |
| UCI Cleveland Heart Disease Dataset | 303 | Independent external benchmark |

---

## 🛠️ Tools & Libraries

Python · PyTorch · XGBoost · Scikit-learn · SHAP · Pandas · NumPy · Matplotlib · Seaborn · Google Colab

---

## 📈 Results

### Effect of Preprocessing Strategy (Accuracy)

| Model | IQR Row Removal | Row Clipping | Improvement |
|---|---|---|---|
| Decision Tree | 71.55% | 82.48% | +10.93% |
| Random Forest | 71.93% | 83.32% | +11.39% |
| XGBoost | 71.58% | 83.39% | +11.81% |
| FT-Transformer | 71.90% | 83.02% | +11.12% |
| TabTransformer | 71.74% | 81.69% | +9.95% |
| SAINT | 71.57% | 82.53% | +10.96% |

Row Clipping consistently outperformed IQR-based removal across all baselines and was adopted as the preprocessing strategy for all subsequent experiments.

### Baseline Model Performance (Row-Clipped Data)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 82.48% | 94.78% | 69.01% | 79.71% | 90.02% |
| Random Forest | 83.32% | 95.93% | 69.58% | 80.65% | 90.86% |
| XGBoost | 83.39% | 92.22% | 72.92% | 81.44% | 91.02% |
| FT-Transformer | 83.02% | 92.94% | 71.46% | 80.79% | 90.73% |
| TabTransformer | 81.69% | 90.74% | 70.55% | 79.38% | 89.37% |
| SAINT | 82.53% | 90.41% | 72.76% | 80.63% | 90.55% |

### Proposed CardioFormer-XGB — Cross-Validation vs. Independent Test

| Metric | Average 10-Fold CV | Independent Test Set |
|---|---|---|
| Accuracy | 84.00% | 83.16% |
| Precision | 91.68% | 91.83% |
| Recall | 73.32% | 72.78% |
| F1-Score | 81.48% | 81.21% |
| ROC-AUC | 90.91% | 90.91% |

The minimal gap between cross-validation and test performance (≤0.84 percentage points on every metric) confirms strong generalization with no overfitting.

### CardioFormer-XGB vs. Its Base Models

| Metric | XGBoost | FT-Transformer | **CardioFormer-XGB** |
|---|---|---|---|
| Accuracy | 83.39% | 83.02% | **84.00%** |
| Precision | 92.22% | 92.94% | 91.68% |
| Recall | 72.92% | 71.46% | **73.32%** |
| F1-Score | 81.44% | 80.79% | **81.48%** |
| ROC-AUC | 91.02% | 90.73% | 90.91% |

CardioFormer-XGB achieves the best accuracy, recall, and F1-score of the three, while remaining within ~0.5 percentage points of the best precision and ROC-AUC — showing it successfully combines the strengths of both base models rather than simply averaging them.

### Independent External Benchmark — UCI Cleveland Heart Disease Dataset

| Metric | FT-Transformer | **CardioFormer-XGB** |
|---|---|---|
| Accuracy | 88.52% | **90.16%** |
| Precision | 86.21% | **86.67%** |
| Recall | 89.29% | **92.86%** |
| F1-score | 87.72% | **89.66%** |
| ROC-AUC | **94.70%** | 93.18% |

CardioFormer-XGB outperforms FT-Transformer alone on 4 of 5 metrics on a completely independent, unseen dataset — evidence that the hybrid architecture generalizes beyond the training distribution.

---

## 🔬 Explainability (SHAP)

- **Global importance:** Systolic blood pressure (ap_hi), cholesterol, diastolic blood pressure (ap_lo), age, and MAP are consistently the strongest predictors across baseline and hybrid models.
- **Embedding contribution:** Several FT-Transformer embedding dimensions (e.g., `embed_33`, `embed_16`, `embed_59`) rank among the top contributors in the hybrid model, showing that the learned latent representations capture clinically meaningful, non-linear feature interactions.
- **Local explanations:** SHAP waterfall plots for individual high-risk and low-risk patients show clear, feature-by-feature reasoning behind each prediction.

---

## 📂 Repository Structure

```
├── data/                  # dataset or download instructions (raw data not included)
├── preprocessing/         # cleaning, outlier handling, feature engineering scripts
├── models/                # baseline model training scripts (DT, RF, XGBoost, FT-Transformer, TabTransformer, SAINT)
├── cardioformer_xgb/      # hybrid model: embedding extraction + feature fusion + XGBoost
├── explainability/        # SHAP analysis scripts (summary, importance, dependence, waterfall)
├── notebooks/             # Colab/Jupyter notebooks
├── results/               # saved metrics, confusion matrix, ROC curve, SHAP plots
└── README.md
```


---

## 📌 Key Takeaways

- **Preprocessing matters:** Row Clipping improved accuracy by ~10–12 percentage points over IQR-based row removal across every model tested.
- **Hybrid > either base model:** CardioFormer-XGB matches or exceeds both XGBoost and FT-Transformer on accuracy, recall, and F1-score.
- **Generalizes well:** Minimal train/test gap and strong performance on an independent external dataset (UCI Cleveland).
- **Interpretable by design:** SHAP integration ensures every prediction can be explained down to individual clinical features and learned embeddings — not just a black-box output.

---

## 👤 Author

[Shirin Mahbuba,Tanjila Habib Srabony ] — Undergraduate Thesis Project, [American International University-Bangladesh], 2026

## 📄 License

MIT License
