# 💳 Fraud Detection using Machine Learning

## 📌 Overview
This project is a **machine learning pipeline** for detecting fraudulent financial transactions.  
It was developed as part of a data science case study, based on the provided transaction dataset (~6.36M records).  
The model predicts whether a transaction is fraudulent using transaction metadata and derived features.

---

## 🎯 Objectives
From the case study instructions (Task Details PDF), the project aims to:
1. Clean and preprocess the raw data.
2. Train a fraud detection model with clear description of the methodology.
3. Identify the **key factors** that predict fraudulent behavior.
4. Recommend actions to prevent such frauds in the future.
5. Suggest how to **measure** the success of the implemented actions.

---

## 📂 Dataset
**Source:** Provided in `Fraud.csv`  
**Rows:** ~6.36 million  
**Target column:** `isFraud` (1 = fraud, 0 = non-fraud)

### Columns (selected):
- `type`: Transaction type (TRANSFER, CASH_OUT, etc.)
- `amount`: Transaction amount
- `oldbalanceOrg`, `newbalanceOrig`: Sender's balance before/after transaction
- `oldbalanceDest`, `newbalanceDest`: Receiver's balance before/after transaction
- Derived features:
  - `diffOrig`, `diffDest`
  - `amount_over_oldOrig`, `amount_over_oldDest`
  - `isOrigMerchant`, `isDestMerchant`

---

## 🛠️ Methodology
### 1. **Data Preprocessing**
- Convert numeric columns to correct types
- Handle missing values
- One-hot encode `type`
- Create derived features for balance changes and ratios
- Drop irrelevant IDs (`nameOrig`, `nameDest`)

### 2. **Model Selection**
- **LightGBM** chosen for:
  - High accuracy on tabular data
  - Fast training on large datasets
  - Handles missing values and categorical features

### 3. **Training**
- Stratified train-test split to preserve fraud ratio
- Early stopping with AUC metric
- Hyperparameters tuned for balance between recall and precision

### 4. **Evaluation Metrics**
- **ROC AUC**: Measures ranking quality of predictions
- **PR AUC**: Especially important due to class imbalance
- **Classification report**: Precision, recall, F1-score
- **Confusion matrix**: TP, FP, FN, TN counts

---

## 📊 Results (Sample 200k Run)
| Metric      | Value |
|-------------|-------|
| ROC AUC     | 0.999+ |
| PR AUC      | 0.98+  |
| Precision   | High   |
| Recall      | High   |

**Feature Importance (Top 5):**
1. `amount_over_oldOrig`
2. `diffOrig`
3. `oldbalanceOrg`
4. `diffDest`
5. `type_TRANSFER`

**Why these make sense:**
- Fraudulent transactions often involve large transfers relative to account balance.
- Fraud patterns leave sender's account drained (`diffOrig` large).
- Transfers and cash-outs are common fraud transaction types.

---

## 📈 SHAP Analysis
SHAP (SHapley Additive exPlanations) was used to interpret the LightGBM model.

![SHAP Summary Plot](shap_summary.png)

---

## 🛡️ Recommendations
1. **Flag** high `amount_over_oldOrig` ratios, especially for `TRANSFER`/`CASH_OUT`.
2. **Require MFA** for large or unusual transfers.
3. **Monitor sudden balance drops** (`diffOrig`) and unexpected balance increases in receivers (`diffDest`).
4. **Track new recipient accounts** receiving multiple large transfers quickly.

---

## 📏 Measuring Effectiveness
- Track fraud detection **recall** before/after implementing rules.
- Monitor **false positive rate** to reduce user inconvenience.
- Compare **confirmed fraud cases** caught after implementing changes.

---

## 📦 Project Structure
