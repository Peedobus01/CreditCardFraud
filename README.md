# Credit-Card-Fraud-Detection
This project aims to build a robust Credit Card Fraud Detection Model using Machine Learning (Random Forest Classifier), starting from Exploratory Data Analysis (EDA) to deployment via Flask. The workflow is built on the popular **[Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**.

## Problem Statement
Credit card fraud poses significant financial risks globally.  
**Goal** → Accurately classify transactions as **Fraudulent (1)** or **Legitimate (0)**.

**Dataset Summary:**
- **Total Features:** 30  
- **Features:**
  - `Time`
  - `V1` to `V28` (PCA-transformed, anonymized)
  - `Amount`
- **Target Variable:** `Class` (0 = Legit, 1 = Fraud)

## Project Workflow

### 1. 📊 Data Understanding & EDA
- Loaded and explored the **Kaggle Credit Card Fraud dataset**.
- Identified **severe class imbalance** (Very few frauds vs legit transactions).
- Visualized distributions using:
  - **Histograms**
  - **Boxplots**
  - **Correlation Heatmaps**
- Outlier detection in key features: `V3`, `V4`, `V9`, `V10`, `V11`, etc.

### 2. Feature Engineering
- Created new features like:
  - **Transaction Hour** (from `Time`)
  - **Log-transformed Amount** (to reduce skewness)
  - **Outlier Flags** based on boxplot outlier thresholds for key `V` features.
- Dropped raw `Time` and `Amount` columns after transformation.

### 3. Handling Imbalanced Data
- Used two approaches:
  - **Undersampling** (using `RandomUnderSampler` from `imblearn`)
  - **Oversampling** (using `SMOTE`)
- Created separate datasets for both approaches and compared model performance.

### 4. Model Building & Evaluation
Trained four different models on all three data variations (Normal, Undersampled, Oversampled):

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|------ |--------- |--------- |------ |-------- |--- |
| Logistic Regression | ✅ | ✅ | ✅ | ✅ | ✅ |
| Decision Tree | ✅ | ✅ | ✅ | ✅ | ✅ |
| Random Forest | ✅ | ✅ | ✅ | ✅ | ✅ |
| XGBoost | ✅ | ✅ | ✅ | ✅ | ✅ |

 - Evaluation metrics used:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1 Score**
    - **AUC ROC**
    - **Precision-Recall Curve**
    - **Feature Importance (Random Forest & XGBoost)**

**Final Model Chosen**: ✅ **Random Forest on Oversampled Data**
(best balance between **recall**, **F1 score**, and **AUC**)

### 5. Model Deployment (Streamlit)
- Created a preprocessing and prediction system that dynamically checks and structures inputs for the Random Forest pipeline (`rf_credit_fraud_pipeline.pkl`).
- Built an interactive **Streamlit web application** (`app.py`) featuring:
  - **Quick Presets:** Instantly load legitimate or fraudulent transaction details.
  - **Organized Interface:** Features grouped into clean tabs (Basic Details, PCA V1-V14, PCA V15-V28) using sliders for easy adjustments.
  - **Instant Inference:** Predicts transaction security status (Green for Legit, Red for Fraud) and visualizes risk probabilities.

#### Running the App locally:
1. Install Streamlit and required dependencies:
   ```bash
   pip install streamlit joblib pandas numpy scikit-learn
   ```
2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and navigate to the local address displayed (usually `http://localhost:8501`).
