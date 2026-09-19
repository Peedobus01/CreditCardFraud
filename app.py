import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="SecureCard AI - ML Showcase & Predictor",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            color: #1E3A8A;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #4B5563;
            margin-bottom: 2rem;
            text-align: center;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #E5E7EB;
            padding-bottom: 0.3rem;
        }
        .metric-card {
            background-color: #FFFFFF;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #3B82F6;
            margin-bottom: 1rem;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #6B7280;
            text-transform: uppercase;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #111827;
        }
        .highlight-legit {
            border-left: 6px solid #10B981 !important;
            background-color: #ECFDF5;
        }
        .highlight-fraud {
            border-left: 6px solid #EF4444 !important;
            background-color: #FEF2F2;
        }
        .highlight-gold {
            border-left: 6px solid #F59E0B !important;
            background-color: #FEF3C7;
        }
        .confusion-matrix {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            max-width: 400px;
            margin: auto;
            text-align: center;
        }
        .cm-cell {
            padding: 1rem;
            border-radius: 8px;
            background-color: #F3F4F6;
            border: 1px solid #D1D5DB;
        }
        .cm-title {
            font-size: 0.8rem;
            color: #4B5563;
            font-weight: 600;
        }
        .cm-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #111827;
        }
    </style>
""", unsafe_allow_html=True)

# Model Loading Section
@st.cache_resource
def load_model():
    try:
        model = joblib.load("rf_credit_fraud_pipeline.pkl")
        return model, None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

# ==========================================
# Dataset Loading & Caching Section
# ==========================================
def generate_synthetic_data():
    np.random.seed(42)
    n_legit = 10000
    n_fraud = 492
    
    # Legit transaction hours
    hours_legit = np.concatenate([
        np.random.normal(8, 2, int(n_legit * 0.3)),
        np.random.normal(14, 3, int(n_legit * 0.4)),
        np.random.normal(20, 2, int(n_legit * 0.3))
    ])
    hours_legit = np.clip(hours_legit, 0, 23.9)
    time_legit = hours_legit * 3600
    
    # Fraud transaction hours (more uniform or night-shifted)
    hours_fraud = np.random.uniform(0, 24, n_fraud)
    time_fraud = hours_fraud * 3600
    
    # Amounts
    amount_legit = np.random.exponential(80, n_legit)
    n_fraud_small = int(n_fraud * 0.2)
    n_fraud_large = n_fraud - n_fraud_small
    amount_fraud = np.concatenate([
        np.random.uniform(1, 10, n_fraud_small),
        np.random.exponential(300, n_fraud_large)
    ])

    
    # V1-V28 features
    v_legit = np.random.normal(0, 1.0, (n_legit, 28))
    v_fraud = np.random.normal(0, 1.2, (n_fraud, 28))
    
    # Fraud anomalies: V14 is negative, V17 is negative, V4 is positive, V11 is positive
    v_fraud[:, 13] -= 7.0  # V14
    v_fraud[:, 16] -= 6.0  # V17
    v_fraud[:, 11] -= 5.0  # V12
    v_fraud[:, 9] -= 4.0   # V10
    v_fraud[:, 3] += 5.0   # V4
    v_fraud[:, 10] += 4.0  # V11
    
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'] + ['Class']
    
    df_legit = pd.DataFrame(np.hstack([time_legit.reshape(-1, 1), v_legit, amount_legit.reshape(-1, 1), np.zeros((n_legit, 1))]), columns=columns)
    df_fraud = pd.DataFrame(np.hstack([time_fraud.reshape(-1, 1), v_fraud, amount_fraud.reshape(-1, 1), np.ones((n_fraud, 1))]), columns=columns)
    
    df = pd.concat([df_legit, df_fraud]).reset_index(drop=True)
    df['Class'] = df['Class'].astype(int)
    
    stats = {
        "total": 284807,
        "fraud": 492,
        "legit": 284315,
        "fraud_rate": 0.17
    }
    
    return df, df, stats

@st.cache_data
def load_dataset():
    try:
        # Load dataset
        df = pd.read_csv("archive/creditcard.csv")
        
        # Calculate true summary statistics
        total_count = len(df)
        fraud_count = int((df['Class'] == 1).sum())
        legit_count = total_count - fraud_count
        fraud_rate = (fraud_count / total_count) * 100
        
        # Create downsampled version for quick plotting:
        fraud_df = df[df['Class'] == 1].copy()
        legit_df = df[df['Class'] == 0].sample(n=min(10000, legit_count), random_state=42).copy()
        vis_df = pd.concat([fraud_df, legit_df]).reset_index(drop=True)
        
        stats = {
            "total": total_count,
            "fraud": fraud_count,
            "legit": legit_count,
            "fraud_rate": fraud_rate
        }
        
        return vis_df, stats, None
    except Exception as e:
        df_syn, _, stats_syn = generate_synthetic_data()
        return df_syn, stats_syn, str(e)

vis_df, dataset_stats, data_error = load_dataset()

# Sidebar Navigation
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio(
    "Go To Section:",
    [
        "📂 Project Overview",
        "📊 Exploratory Data Analysis (EDA)",
        "🛠️ Feature Engineering & Sampling",
        "🏆 Model Performance & Training",
        "🔐 Live Fraud Predictor"
    ]
)

st.sidebar.markdown("---")
if model_error:
    st.sidebar.error(f"❌ Model load error:\n`{model_error}`")
else:
    st.sidebar.success("✅ Random Forest Model loaded successfully!")

# Define data structures for display
PRESETS = {
    "Custom / Reset": {
        "Time": 0.0,
        "Amount": 0.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    },
    "Sample Legit Transaction": {
        "Time": 36000.0,
        "Amount": 45.50,
        **{f"V{i}": 0.05 * (-1 if i % 3 == 0 else 1) for i in range(1, 29)}
    },
    "Sample Fraudulent Transaction": {
        "Time": 72000.0,
        "Amount": 950.00,
        **{
            "V1": -1.3, "V2": 1.2, "V3": -4.2, "V4": 4.5, "V5": -1.1,
            "V6": -0.8, "V7": -2.5, "V8": 0.9, "V9": -2.8, "V10": -5.9,
            "V11": 3.8, "V12": -7.2, "V13": -0.2, "V14": -8.5, "V15": 0.4,
            "V16": -3.1, "V17": -10.2, "V18": -3.9, "V19": 1.2, "V20": 0.4,
            "V21": 0.8, "V22": -0.3, "V23": -0.2, "V24": -0.4, "V25": 0.3,
            "V26": 0.2, "V27": 0.5, "V28": 0.1
        }
    }
}

MODEL_RESULTS = [
    {"Model": "Logistic Regression (Original)", "Accuracy": 0.9993, "Precision": 0.8391, "Recall": 0.7449, "F1 Score": 0.7892, "AUC": 0.9578, "Data Split": "Original"},
    {"Model": "Decision Tree (Original)", "Accuracy": 0.9991, "Precision": 0.7475, "Recall": 0.7551, "F1 Score": 0.7513, "AUC": 0.8773, "Data Split": "Original"},
    {"Model": "Random Forest (Original)", "Accuracy": 0.9996, "Precision": 0.9405, "Recall": 0.8061, "F1 Score": 0.8681, "AUC": 0.9579, "Data Split": "Original"},
    {"Model": "XGBoost (Original)", "Accuracy": 0.9996, "Precision": 0.9011, "Recall": 0.8367, "F1 Score": 0.8677, "AUC": 0.9788, "Data Split": "Original"},
    
    {"Model": "Logistic Regression (Undersampled)", "Accuracy": 0.9689, "Precision": 0.0481, "Recall": 0.9082, "F1 Score": 0.0913, "AUC": 0.9719, "Data Split": "Undersampled"},
    {"Model": "Decision Tree (Undersampled)", "Accuracy": 0.8915, "Precision": 0.0144, "Recall": 0.9184, "F1 Score": 0.0283, "AUC": 0.9049, "Data Split": "Undersampled"},
    {"Model": "Random Forest (Undersampled)", "Accuracy": 0.9637, "Precision": 0.0415, "Recall": 0.9082, "F1 Score": 0.0793, "AUC": 0.9728, "Data Split": "Undersampled"},
    {"Model": "XGBoost (Undersampled)", "Accuracy": 0.9557, "Precision": 0.0346, "Recall": 0.9184, "F1 Score": 0.0666, "AUC": 0.9750, "Data Split": "Undersampled"},
    
    {"Model": "Logistic Regression (Oversampled)", "Accuracy": 0.9716, "Precision": 0.0520, "Recall": 0.8980, "F1 Score": 0.0983, "AUC": 0.9689, "Data Split": "Oversampled"},
    {"Model": "Decision Tree (Oversampled)", "Accuracy": 0.9974, "Precision": 0.3816, "Recall": 0.8061, "F1 Score": 0.5180, "AUC": 0.9019, "Data Split": "Oversampled"},
    {"Model": "Random Forest (Oversampled)", "Accuracy": 0.9996, "Precision": 0.9022, "Recall": 0.8469, "F1 Score": 0.8737, "AUC": 0.9786, "Data Split": "Oversampled"},
    {"Model": "XGBoost (Oversampled)", "Accuracy": 0.9994, "Precision": 0.8235, "Recall": 0.8571, "F1 Score": 0.8400, "AUC": 0.9861, "Data Split": "Oversampled"}
]
results_df = pd.DataFrame(MODEL_RESULTS)

# ==========================================
# PAGE 1: PROJECT OVERVIEW
# ==========================================
if page == "📂 Project Overview":
    st.markdown('<div class="main-header">🔐 SecureCard AI: Fraud Detection Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ML Project Portfolio & Dynamic Anomaly Classifier</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">📌 Problem Statement</div>', unsafe_allow_html=True)
    st.write("""
    Credit card fraud accounts for billions of dollars in losses annually. Identifying fraudulent transactions in real-time is 
    challenging because fraudulent activities represent a tiny fraction of total transactions. 
    
    **Goal:** Build a robust classifier capable of identifying **Fraudulent (1)** vs. **Legitimate (0)** transactions with high accuracy, recall, and precision.
    """)
    
    st.markdown('<div class="section-header">📊 Dataset Profile</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Total Transactions</div>
                <div class="metric-value">284,807</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card highlight-fraud">
                <div class="metric-label">Fraudulent Cases</div>
                <div class="metric-value">492 (0.17%)</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Anonymized Features</div>
                <div class="metric-value">28 (PCA V1-V28)</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("""
    The dataset used is the popular **Kaggle Credit Card Fraud Detection Dataset**. 
    Due to confidentiality issues, the dataset contains features `V1` to `V28` which are PCA (Principal Component Analysis) transformed dimensions. 
    The only non-transformed features are `Time` (seconds elapsed between each transaction and the first transaction) and `Amount` (transaction amount).
    """)

    st.markdown('<div class="section-header">🔄 Machine Learning Pipeline Workflow</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Data Understanding & EDA:** Analyzed feature correlations, value frequencies, and severe class imbalance.
    2. **Feature Engineering:** Log-transformed high-skewness transaction amounts, extracted transaction hour values, and engineered binary outlier indicators.
    3. **Addressing Class Imbalance:** Tested **SMOTE (Oversampling)** and **Random Undersampling** approaches to ensure the model doesn't overfit to legitimate transactions.
    4. **Model Architecture Comparison:** Evaluated **Logistic Regression, Decision Tree, Random Forest, and XGBoost** across three data splits (Natural, Undersampled, and Oversampled).
    5. **Selection & Serialization:** Saved the best-performing **Random Forest Classifier (SMOTE)** as a pipeline (`rf_credit_fraud_pipeline.pkl`).
    """)

# ==========================================
# PAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
elif page == "📊 Exploratory Data Analysis (EDA)":
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis & Anomaly Detection</div>', unsafe_allow_html=True)
    
    # Error/Graceful degradation handling
    if data_error:
        st.warning(f"⚠️ **Using Synthetic Demo Data:** The dataset `archive/creditcard.csv` could not be found or loaded (Error: `{data_error}`). We are displaying dynamically generated synthetic data to preview the dashboard interactive features. To see real transaction data, please place the Kaggle dataset in `archive/creditcard.csv`.")
    else:
        st.success("✅ **Real Dataset Loaded:** Interactive graphs are rendering live dataset statistics from `archive/creditcard.csv`.")
    
    # Dynamic KPIs
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Transactions</div>
                <div class="metric-value">{dataset_stats['total']:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Legitimate Cases</div>
                <div class="metric-value">{dataset_stats['legit']:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"""
            <div class="metric-card highlight-fraud">
                <div class="metric-label">Fraudulent Cases</div>
                <div class="metric-value">{dataset_stats['fraud']:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_kpi4:
        st.markdown(f"""
            <div class="metric-card highlight-gold">
                <div class="metric-label">Fraud Rate</div>
                <div class="metric-value">{dataset_stats['fraud_rate']:.4f}%</div>
            </div>
        """, unsafe_allow_html=True)

    # Initialize tab layout
    tab_overview, tab_time_amount, tab_pca, tab_corr = st.tabs([
        "⚖️ Class Balance & Overview",
        "🕒 Time & Amount Patterns",
        "🔍 PCA Outlier Boxplots",
        "📊 Correlation Matrix"
    ])

    # Pre-processing visualization data
    vis_df_plot = vis_df.copy()
    vis_df_plot['Hour'] = (vis_df_plot['Time'] // 3600) % 24
    vis_df_plot['Transaction Type'] = vis_df_plot['Class'].map({0: 'Legitimate', 1: 'Fraudulent'})
    vis_df_plot['Log_Amount'] = np.log1p(vis_df_plot['Amount'])

    # ==========================================
    # TAB 1: OVERVIEW & CLASS BALANCE
    # ==========================================
    with tab_overview:
        st.subheader("⚖️ Severe Class Imbalance")
        st.write("""
        A fundamental characteristic of fraud detection datasets is the extreme scarcity of positive classes.
        Out of the entire dataset, only a fraction of a percent are marked as fraudulent.
        """)
        
        # Pie Chart with Plotly Express
        pie_df = pd.DataFrame({
            "Category": ["Legitimate (Class 0)", "Fraudulent (Class 1)"],
            "Count": [dataset_stats['legit'], dataset_stats['fraud']]
        })
        
        fig_pie = px.pie(
            pie_df, 
            values="Count", 
            names="Category",
            color="Category",
            color_discrete_map={
                "Legitimate (Class 0)": "#3B82F6", # Sleek Blue
                "Fraudulent (Class 1)": "#EF4444"  # Alert Red
            },
            hole=0.4,
            title="Distribution of Credit Card Transactions"
        )
        fig_pie.update_layout(
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        col_pie_left, col_pie_right = st.columns([1, 1])
        with col_pie_left:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_pie_right:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Category": ["Legitimate (Class 0)", "Fraudulent (Class 1)"],
                "Count": [dataset_stats['legit'], dataset_stats['fraud']],
                "Percentage": [100 - dataset_stats['fraud_rate'], dataset_stats['fraud_rate']]
            }).style.format({"Count": "{:,}", "Percentage": "{:.4f}%"}), use_container_width=True)
            
        st.warning("""
        ⚠️ **Why Standard Accuracy Fails:** 
        If a naive model predicts 'Legitimate' for every single transaction, it will achieve **99.83% Accuracy**. 
        However, it will catch **0%** of the fraudulent transactions (0% Recall). Therefore, we must focus on **F1 Score**, **Recall**, and **Precision-Recall Curves** rather than raw accuracy.
        """)

    # ==========================================
    # TAB 2: TIME & AMOUNT PATTERNS
    # ==========================================
    with tab_time_amount:
        st.subheader("🕒 Temporal & Value Patterns")
        st.write("""
        Analyzing transaction times and amounts exposes key behavioral differences.
        - **Time Patterns:** Legitimate transactions exhibit clear day-night cycles (dropping in sleep hours), while fraudulent transactions are more evenly spread.
        - **Amount Distributions:** Fraudulent transactions often feature either micro-charges (for card validation) or high-value amounts compared to typical shopping profiles.
        """)
        
        # Time distribution
        fig_time = px.histogram(
            vis_df_plot,
            x="Hour",
            color="Transaction Type",
            barmode="overlay",
            histnorm="probability density",
            nbins=24,
            color_discrete_map={"Legitimate": "#3B82F6", "Fraudulent": "#EF4444"},
            title="Transaction Density over 24-Hour Cycle (Normalized)"
        )
        fig_time.update_layout(
            xaxis_title="Hour of Day (0 = Midnight, 12 = Noon)",
            yaxis_title="Normalized Density",
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        # Amount Log distribution
        fig_amount_dist = px.histogram(
            vis_df_plot,
            x="Log_Amount",
            color="Transaction Type",
            barmode="overlay",
            histnorm="probability density",
            nbins=50,
            color_discrete_map={"Legitimate": "#3B82F6", "Fraudulent": "#EF4444"},
            title="Transaction Amount Distribution (Log Scale)"
        )
        fig_amount_dist.update_layout(
            xaxis_title="Log(Amount + 1)",
            yaxis_title="Normalized Density",
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        col_time, col_amount = st.columns(2)
        with col_time:
            st.plotly_chart(fig_time, use_container_width=True)
        with col_amount:
            st.plotly_chart(fig_amount_dist, use_container_width=True)
            
        st.subheader("💵 Transaction Amount Ranges")
        
        # Box plots for Amount
        fig_amount_box = px.box(
            vis_df_plot,
            x="Transaction Type",
            y="Amount",
            color="Transaction Type",
            color_discrete_map={"Legitimate": "#3B82F6", "Fraudulent": "#EF4444"},
            title="Transaction Amount Comparison (Outliers Trimmed up to $1,000 for Visibility)"
        )
        fig_amount_box.update_layout(
            yaxis=dict(range=[-10, 1000]),
            yaxis_title="Amount ($)",
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_amount_box, use_container_width=True)

    # ==========================================
    # TAB 3: PCA OUTLIER BOXPLOTS
    # ==========================================
    with tab_pca:
        st.subheader("🔍 PCA Feature Distributions")
        st.write("""
        Features `V1` to `V28` are anonymized components obtained through PCA. 
        Select a component below to see how its distribution differs significantly between legitimate and fraudulent transactions.
        Notice how fraudulent transactions (red) are shifted or show wider variance.
        """)
        
        # Selectbox to choose V-feature
        all_pca_features = [f"V{i}" for i in range(1, 29)]
        selected_feature = st.selectbox(
            "Select PCA Component to Inspect:",
            options=all_pca_features,
            index=all_pca_features.index("V14") # Default to V14
        )
        
        # Create boxplot for the selected feature
        fig_pca_box = px.box(
            vis_df_plot,
            x="Transaction Type",
            y=selected_feature,
            color="Transaction Type",
            color_discrete_map={"Legitimate": "#3B82F6", "Fraudulent": "#EF4444"},
            points="outliers",
            title=f"Distribution of {selected_feature} by Transaction Type"
        )
        fig_pca_box.update_layout(
            yaxis_title=f"{selected_feature} Value",
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=False
        )
        
        # Display description based on selected feature
        feature_desc = ""
        if selected_feature == "V14":
            feature_desc = "💡 **Insight on V14:** Deeply negative values (< -4) indicate high risk. It is one of the strongest negative indicators in fraud detection."
        elif selected_feature == "V17":
            feature_desc = "💡 **Insight on V17:** Deeply negative values (< -2) are strongly correlated with fraud. V17 is an essential indicator."
        elif selected_feature == "V12":
            feature_desc = "💡 **Insight on V12:** Shows a severe negative shift in fraud cases. Good indicator of fraudulent transaction patterns."
        elif selected_feature == "V10":
            feature_desc = "💡 **Insight on V10:** Extreme negative outliers below -5 correspond highly with fraud."
        elif selected_feature == "V4":
            feature_desc = "💡 **Insight on V4:** Highly positive values (> 2) are correlated with fraud. Elevated values point to anomalies."
        elif selected_feature == "V11":
            feature_desc = "💡 **Insight on V11:** Show a positive shift, similar to V4. Higher values correspond with elevated risk."
        else:
            feature_desc = f"💡 **Distribution Analysis:** Inspect the median (center line) and range of `{selected_feature}`. A visible separation between legitimate and fraudulent transaction box plots indicates that this feature is a strong discriminator."
            
        col_box_plot, col_box_desc = st.columns([2, 1])
        with col_box_plot:
            st.plotly_chart(fig_pca_box, use_container_width=True)
        with col_box_desc:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: #F3F4F6; padding: 1.5rem; border-radius: 8px; border-left: 5px solid #1E3A8A;">
                <h4>Distribution Summary</h4>
                <p>{feature_desc}</p>
                <p>Use the dropdown above to cycle through other PCA components and analyze their discriminatory power.</p>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # TAB 4: CORRELATION MATRIX
    # ==========================================
    with tab_corr:
        st.subheader("📊 Correlation with Fraud (Class)")
        st.write("""
        Because the PCA components are orthogonal (uncorrelated with each other), their correlation with the target variable `Class` 
        directly indicates their linear discriminatory power.
        """)
        
        # Calculate correlation with Class
        corr_series = vis_df.corr()['Class'].drop('Class').sort_values()
        corr_df = pd.DataFrame({
            "Feature": corr_series.index,
            "Correlation": corr_series.values
        })
        
        # Plot correlation bar chart
        fig_corr_bar = px.bar(
            corr_df,
            x="Feature",
            y="Correlation",
            color="Correlation",
            color_continuous_scale="RdBu",
            title="Correlation of Features with Class (Fraud Target)"
        )
        fig_corr_bar.update_layout(
            yaxis_title="Correlation Coefficient",
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_corr_bar, use_container_width=True)
        
        # Correlation heatmap of top indicators
        st.subheader("🔥 Inter-Feature Correlation Heatmap (Top Indicators)")
        st.write("Below is the correlation matrix of the top 10 features most correlated with fraud, alongside Time and Amount.")
        
        top_neg = corr_series.head(5).index.tolist()
        top_pos = corr_series.tail(5).index.tolist()
        top_features = top_neg + top_pos + ['Amount', 'Time', 'Class']
        
        corr_matrix = vis_df[top_features].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            aspect="auto",
            title="Correlation Heatmap of Top Indicators"
        )
        fig_heatmap.update_layout(
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)


# ==========================================
# PAGE 3: FEATURE ENGINEERING & SAMPLING
# ==========================================
elif page == "🛠️ Feature Engineering & Sampling":
    st.markdown('<div class="section-header">🛠️ Feature Engineering & Class Balancing</div>', unsafe_allow_html=True)
    
    st.subheader("💡 Feature Engineering Steps")
    st.write("To improve model performance, several pre-processing and feature creation steps were introduced:")
    
    st.code("""
# 1. Hour Extraction from Time
data['Hour'] = (data['Time'] // 3600) % 24

# 2. Log Transform skewness reduction
data['Log_Amount'] = np.log1p(data['Amount'])

# 3. Dropping Raw Columns
data.drop(['Time', 'Amount'], axis=1, inplace=True)
    """, language="python")

    st.markdown("""
    - **Transaction Hour:** Transaction times are originally represented as cumulative seconds. Converting them to hours of day ($0$–$23$) helps capture diurnal spend patterns.
    - **Log Transform (`log1p`):** Transaction amounts are highly skewed (mostly small amounts, with rare multi-thousand-dollar transactions). Applying $\log(x+1)$ stabilizes variance.
    - **Outlier Indicators:** Engineered binary outlier flags for critical PCA components (`V3`, `V4`, `V9`, `V10`, `V11`, `V12`, `V14`, `V16`, `V17`, `V18`, `V19`) checking if their values exceeded extreme boundaries ($> 2$ or $< -2$).
    """)

    st.markdown('<div class="section-header">⚖️ Resampling Techniques</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📉 Undersampling (RandomUnderSampler)
        - **Method:** Randomly drop legitimate transaction samples to match the count of fraud cases (492).
        - **Pros:** Fast training, less memory consumption.
        - **Cons:** Discards 99% of available data, leading to high variance and high false-positive rates.
        """)
    with col2:
        st.markdown("""
        ### 📈 Oversampling (SMOTE)
        - **Method:** Synthetic Minority Over-sampling Technique. Synthetically generates new fraud samples along lines joining existing fraud cases.
        - **Pros:** Preserves all legitimate transaction information; robust classifier training.
        - **Cons:** Takes longer to train due to expanded dataset size.
        """)

# ==========================================
# PAGE 4: MODEL TRAINING & PERFORMANCE
# ==========================================
elif page == "🏆 Model Performance & Training":
    st.markdown('<div class="section-header">🏆 Model Evaluation & Comparison</div>', unsafe_allow_html=True)
    st.write("Below is the performance summary of **12 different model combinations** evaluated during the pipeline run:")
    
    # Dynamic table
    st.dataframe(results_df.sort_values(by="F1 Score", ascending=False).reset_index(drop=True), use_container_width=True)

    # Interactive Bar Chart
    st.subheader("📊 Interactive Metric Performance Chart")
    metric_choice = st.selectbox("Select metric to compare:", ["F1 Score", "AUC", "Recall", "Precision"])
    
    chart_df = results_df[["Model", metric_choice]].set_index("Model")
    st.bar_chart(chart_df)

    st.markdown('<div class="section-header">🌟 Chosen Champion Model</div>', unsafe_allow_html=True)
    
    col_champ1, col_champ2 = st.columns([2, 1])
    with col_champ1:
        st.markdown("""
            <div class="metric-card highlight-gold">
                <div class="metric-label">Best-Performing Classifier</div>
                <div class="metric-value">Random Forest Classifier (Oversampled)</div>
                <p style="margin-top: 10px; color: #78350F; font-weight: 500;">
                    While XGBoost on oversampled data has a slightly higher recall, the Random Forest (Oversampled) model provides the best balance with a high F1 Score (0.8737), high Precision (0.9022), and outstanding AUC (0.9786). It minimizes false positives (which cause friction for legitimate users) while catching 85% of actual fraud.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col_champ2:
        st.markdown("""
        ### Final Model Scores:
        - **Accuracy:** 99.96%
        - **Precision:** 90.22%
        - **Recall (Sensitivity):** 84.69%
        - **F1 Score:** 87.37%
        - **AUC ROC:** 0.9786
        """)

    st.subheader("🎯 Final Model Confusion Matrix (Test Set Sample)")
    st.write("This matrix represents how the final Random Forest model behaves on unseen test data:")
    
    # HTML Styled Confusion Matrix
    st.markdown("""
        <div class="confusion-matrix">
            <div class="cm-cell" style="background-color: #E0F2FE;">
                <div class="cm-title">True Legitimate (TN)</div>
                <div class="cm-value">56,851</div>
                <div style="font-size: 0.75rem; color: #0369A1;">Correctly allowed</div>
            </div>
            <div class="cm-cell" style="background-color: #FEE2E2;">
                <div class="cm-title">False Fraudulent (FP)</div>
                <div class="cm-value">13</div>
                <div style="font-size: 0.75rem; color: #B91C1C;">False alarms</div>
            </div>
            <div class="cm-cell" style="background-color: #FEF2F2;">
                <div class="cm-title">False Legitimate (FN)</div>
                <div class="cm-value">15</div>
                <div style="font-size: 0.75rem; color: #B91C1C;">Missed fraud</div>
            </div>
            <div class="cm-cell" style="background-color: #DCFCE7;">
                <div class="cm-title">True Fraudulent (TP)</div>
                <div class="cm-value">83</div>
                <div style="font-size: 0.75rem; color: #15803D;">Correctly blocked</div>
            </div>
        </div>
        <p style="text-align: center; margin-top: 10px; font-size: 0.9rem; color: #4B5563;">
            Total test transactions: 56,962 | Total actual fraud cases: 98 | Caught: 83
        </p>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 5: LIVE FRAUD PREDICTOR (EXISTING PAGE)
# ==========================================
elif page == "🔐 Live Fraud Predictor":
    st.markdown('<div class="main-header">🔐 Live Transaction Fraud Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Test custom transaction profiles or select a preset scenario from the sidebar controls.</div>', unsafe_allow_html=True)
    
    # Quick start instructions for testing
    st.markdown("""
        <div style="background-color: #EFF6FF; border-left: 5px solid #3B82F6; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <h4 style="margin: 0; color: #1E3A8A; display: flex; align-items: center; gap: 8px;">💡 How to Test the Predictor</h4>
            <p style="margin: 5px 0 0 0; color: #1E40AF; font-size: 0.95rem;">
                By default, the simulation starts with <b>Custom / Reset</b>, which sets all PCA components (V1–V28) to 0.0 (representing a standard transaction with no anomalies). 
                To test different classifications:
            </p>
            <ul style="margin: 5px 0 0 0; padding-left: 20px; color: #1E40AF; font-size: 0.95rem; line-height: 1.4;">
                <li>Select <b>Sample Fraudulent Transaction</b> from the preset dropdown below to instantly load deeply anomalous values (extreme outliers under the PCA tabs), then click <b>Analyze Transaction Security</b> to see a <b>Fraud</b> detection.</li>
                <li>Select <b>Sample Legit Transaction</b> to load a normal shopping profile.</li>
                <li>Use <b>Custom / Reset</b> to adjust individual sliders. Note that the model relies heavily on PCA components V1–V28; changing only Time and Amount while keeping V1–V28 at 0.0 will always be classified as safe.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for all inputs if not present
    if "Time" not in st.session_state:
        for key, val in PRESETS["Custom / Reset"].items():
            st.session_state[key] = val

    def load_preset_page():
        preset_name = st.session_state.preset_selection_live
        values = PRESETS[preset_name]
        for key, val in values.items():
            st.session_state[key] = val

    # Sidebar Scenario Selector specifically for Predictor tab
    st.subheader("🎮 Live Simulation Presets")
    st.selectbox(
        "Load Preset Scenario",
        options=list(PRESETS.keys()),
        key="preset_selection_live",
        on_change=load_preset_page
    )

    st.write("Adjust the transaction features below to test the classifier:")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🕒 Basic Details", "📊 PCA Components (V1 - V14)", "📊 PCA Components (V15 - V28)"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Transaction Time (seconds since first transaction)", 0.0, 172800.0, key="Time", step=1.0)
        with col2:
            st.slider("Transaction Amount ($)", 0.0, 25000.0, key="Amount", step=0.01)

    with tab2:
        v_cols1 = st.columns(2)
        for idx in range(1, 15):
            col_idx = 0 if idx <= 7 else 1
            with v_cols1[col_idx]:
                st.slider(f"PCA Component V{idx}", -30.0, 30.0, key=f"V{idx}", step=0.1)

    with tab3:
        v_cols2 = st.columns(2)
        for idx in range(15, 29):
            col_idx = 0 if idx <= 21 else 1
            with v_cols2[col_idx]:
                st.slider(f"PCA Component V{idx}", -30.0, 30.0, key=f"V{idx}", step=0.1)

    # Preprocessing & Inference Logic
    def preprocess_and_predict(model, raw_features):
        n_features = 30
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        elif hasattr(model, 'steps'):
            try:
                n_features = model.steps[0][1].n_features_in_
            except:
                pass

        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        feature_values = [raw_features[col] for col in columns]

        if n_features == 41:
            # Preprocess features to match the 41-feature training configuration
            input_df = pd.DataFrame([feature_values], columns=columns)
            
            # Step 1: Hour and Log_Amount
            input_df['Hour'] = (input_df['Time'] // 3600) % 24
            input_df['Log_Amount'] = np.log1p(input_df['Amount'])
            
            # Step 2: Drop Time and Amount
            input_df.drop(['Time', 'Amount'], axis=1, inplace=True)
            
            # Step 3: Outlier flags
            outlier_cols = ['V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19']
            for col in outlier_cols:
                input_df[f'{col}_outlier'] = ((input_df[col] > 2) | (input_df[col] < -2)).astype(int)
                
            # Ensure correct column order: V1..V28, Hour, Log_Amount, outlier flags
            final_cols = [f'V{i}' for i in range(1, 29)] + ['Hour', 'Log_Amount'] + [f'{col}_outlier' for col in outlier_cols]
            input_df = input_df[final_cols]
            
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
            return pred, prob, input_df
        else:
            # Raw features prediction
            pred = model.predict([feature_values])[0]
            prob = model.predict_proba([feature_values])[0] if hasattr(model, 'predict_proba') else None
            input_df = pd.DataFrame([feature_values], columns=columns)
            return pred, prob, input_df

    st.markdown("---")

    # Prediction Trigger
    if st.button("🔍 Analyze Transaction Security", type="primary", use_container_width=True):
        if model is None:
            st.error("Cannot perform analysis. Model failed to load. Check the sidebar for details.")
        else:
            # Gather inputs
            raw_features = {
                "Time": st.session_state.Time,
                "Amount": st.session_state.Amount,
                **{f"V{i}": st.session_state[f"V{i}"] for i in range(1, 29)}
            }
            
            with st.spinner("Analyzing transaction data..."):
                pred, prob, processed_df = preprocess_and_predict(model, raw_features)
                
            # Display Result
            st.subheader("📊 Analysis Result")
            
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                if pred == 1:
                    st.markdown(f"""
                        <div class="metric-card highlight-fraud">
                            <div class="metric-label">Prediction Status</div>
                            <div class="metric-value" style="color: #EF4444;">🚨 FRAUDULENT TRANSACTION DETECTED</div>
                            <p style="margin-top: 10px; color: #7F1D1D;">
                                The transaction exhibits high anomaly characteristics matching historical fraudulent patterns. Highly recommended to block or hold this transaction.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="metric-card highlight-legit">
                            <div class="metric-label">Prediction Status</div>
                            <div class="metric-value" style="color: #10B981;">✅ LEGITIMATE TRANSACTION</div>
                            <p style="margin-top: 10px; color: #064E3B;">
                                The transaction parameters align with normal purchasing activity patterns. Risk level is normal.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
            with col_res2:
                if prob is not None:
                    fraud_prob = prob[1] * 100
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Fraud Probability Score</div>
                            <div class="metric-value">{fraud_prob:.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(prob[1])
                else:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Prediction Class</div>
                            <div class="metric-value">{pred}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Detailed breakdown of features
            with st.expander("🔍 View Processed Feature Details"):
                st.write("This is the exact feature vector parsed and submitted to the model classifier:")
                st.dataframe(processed_df)
                
                # Show active outlier flags if any
                if "Hour" in processed_df.columns:
                    outlier_cols = [c for c in processed_df.columns if c.endswith("_outlier")]
                    active_outliers = [c.replace("_outlier", "") for c in outlier_cols if processed_df.loc[0, c] == 1]
                    
                    if active_outliers:
                        st.warning(f"⚠️ **Outlier Anomalies Detected in:** {', '.join(active_outliers)}")
                    else:
                        st.success("✅ No extreme feature outliers found in critical PCA variables.")
