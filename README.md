# 🔊 Sonar Signal Classification: Rock vs. Mine

A machine learning project that classifies sonar returns as bounced off a **metal cylinder (mine)** or a **rock**, using a Support Vector Machine (SVM).

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

## 📌 Overview

Sonar signals reflect differently depending on the object they hit. Each sample records the signal's energy across 60 frequency bands, and the goal is to tell whether the object was a mine or a rock. It's a classic binary classification problem and a good test for models on small, high-dimensional data.

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository: Connectionist Bench (Sonar, Mines vs. Rocks)](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)
- **Samples:** 208 (no missing values, no duplicates)
- **Features:** 60 numeric values (signal strength per frequency band)
- **Target:** `M` (Mine) or `R` (Rock), with a roughly balanced split of 53.4% / 46.6%

| Label | Encoded as |
|-------|-----------|
| Mine  | `0` |
| Rock  | `1` |

## 🛠️ Workflow

1. **Data inspection:** shape, dtypes, null values, duplicates, class balance
2. **EDA:** histograms with KDE, box plots for all 60 features, and a correlation heatmap
3. **Preprocessing:** label encoding of the target and an 80/20 train/test split (`random_state=42`)
4. **Scaling:** `StandardScaler`, fit on the training set only to avoid leakage
5. **Modeling:** `SVC` (RBF kernel, default hyperparameters)
6. **Evaluation:** accuracy and a classification report
7. **Predictive system:** a function that takes 60 signal values and returns `Rock` or `Mine`

## 📈 Results

| Metric | Score |
|--------|-------|
| Training accuracy | 98.19% |
| Test accuracy | **88.10%** |

**Classification report (test set, 42 samples):**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Mine (0) | 0.96 | 0.85 | 0.90 | 26 |
| Rock (1) | 0.79 | 0.94 | 0.86 | 16 |
| **Accuracy** | | | **0.88** | 42 |

> The gap between training (98%) and test (88%) accuracy suggests mild overfitting, which is expected on a dataset of only 208 samples. Hyperparameter tuning (`C`, `gamma`) with cross-validation is a natural next step.

## 🚀 Getting Started

```bash
git clone https://github.com/<Raghuraj-stack>/<Sonar-Signal-Classification>.git
cd <Sonar-Signal-Classification>
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
jupyter notebook Sonar_signal_classification.ipynb
```

Place `sonar.csv` (no header row) in the project root before running the notebook.

## 🔮 Example Prediction

```python
test_sample = X_test.loc[161].tolist()
predictive_system(test_sample)   # -> Rock / Mine
```

## 📁 Project Structure

```
├── Sonar_signal_classification.ipynb
├── sonar.csv
└── README.md
```

## 🧰 Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn

## 🔭 Future Improvements

- Tune `C`, `gamma` and kernel with `GridSearchCV` and stratified k-fold CV
- Compare against Random Forest, KNN and Logistic Regression
- Try dimensionality reduction (PCA) on the 60 correlated features
- Use stratified splits and cross-validation for a more reliable estimate on a small dataset


