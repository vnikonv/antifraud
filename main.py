import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

st.set_page_config(page_title='Antifraud AI', page_icon="ammit_search.png", layout='wide', initial_sidebar_state='auto')

# Custom CSS styling (same as original)
st.markdown("""
<style>
/* Общий стиль сайдбара */
section[data-testid="stSidebar"] {
    background-color: #00B2CA !important;
    color: white !important;
    padding-top: 10px !important;
}

/* Надписи и тексты в сайдбаре */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span {
    color: white !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    border-radius: 12px !important;
}

/* Селектбокс и поля ввода */
section[data-testid="stSidebar"] div[role="combobox"],
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select {
    background-color: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    box-shadow: none !important;
}

/* SVG иконки в сайдбаре */
section[data-testid="stSidebar"] svg {
    color: white !important;
}

/* Выпадающие списки и опции */
ul[role="listbox"],
li[role="option"] {
    background-color: #000000 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
}

li[role="option"]:hover,
li[aria-selected="true"] {
    background-color: #000000 !important;
}

/* Устранение стандартных границ и теней от BaseWeb */
[data-baseweb="select"],
[data-baseweb="select"] * {
    background-color: #000000 !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
}

img {
    border-radius: 32px !important;
}

label div p {
    font-size: 24px !important;
    font-weight: 600 !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 5% !important;
}
</style>
""", unsafe_allow_html=True)

# Model paths
MODEL_PATHS = {
    "XGBoost": "model_xgb.json",
    "LogisticRegression": "model_lr.json",
    "RandomForest": "model_rf.json",
    "GradientBoosting": "model_gb.json",
    "CatBoost": "model_cb.json",
    "LightGBM": "model_lgb.json"
}

# Paths for other components
LABEL_ENCODERS_PATH = "label_encoders.json"
FEATURES_PATH = "features.json"
SCALER_PATH = "scaler.json"

# Sample data generation
def generate_sample_data(num_samples=100):
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 70, size=num_samples),
        'Income': np.random.normal(50000, 15000, num_samples).astype(int),
        'CreditScore': np.random.randint(300, 850, num_samples),
        'LoanAmount': np.random.randint(5000, 50000, num_samples),
        'EmploymentLength': np.random.randint(0, 30, num_samples),
        'DebtToIncome': np.random.uniform(0.1, 0.8, num_samples),
        'HasDefaulted': np.random.choice(['Yes', 'No'], num_samples, p=[0.2, 0.8]),
        'HasBankruptcies': np.random.choice(['Yes', 'No'], num_samples, p=[0.1, 0.9]),
        'GB_flag': np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

# Save sample data
def save_sample_data():
    sample_data = generate_sample_data()
    sample_data.to_csv("sample_data.csv", index=False)
    return sample_data

# Model saving/loading functions (similar to original)
def save_model(model, feature_names, model_path):
    if isinstance(model, xgb.XGBClassifier):
        model.save_model(model_path)
    elif hasattr(model, 'save_model'):  # For CatBoost
        model.save_model(model_path)
    else:
        with open(model_path, "w") as f:
            json.dump({
                "coef": model.coef_.tolist() if hasattr(model, 'coef_') else None,
                "intercept": model.intercept_.tolist() if hasattr(model, 'intercept_') else None,
                "features": feature_names
            }, f)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f)

def load_model(model_path, model_type):
    if model_type == "XGBoost":
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    elif model_type == "CatBoost":
        model = CatBoostClassifier()
        model.load_model(model_path)
    elif model_type == "LogisticRegression":
        with open(model_path, "r") as f:
            data = json.load(f)
        model = LogisticRegression()
        model.coef_ = np.array(data["coef"])
        model.intercept_ = np.array(data["intercept"])
        model.classes_ = np.array([0, 1])
    else:  # For other models that can be pickled
        import joblib
        model = joblib.load(model_path)
    return model

# Other utility functions (same as original)
def save_label_encoders(label_encoders):
    with open(LABEL_ENCODERS_PATH, "w") as f:
        json.dump({col: le.classes_.tolist() for col, le in label_encoders.items()}, f)

def load_label_encoders():
    with open(LABEL_ENCODERS_PATH, "r") as f:
        return {col: LabelEncoder().fit(classes) for col, classes in json.load(f).items()}

def save_scaler(scaler):
    with open(SCALER_PATH, "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

def load_scaler():
    with open(SCALER_PATH, "r") as f:
        data = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(data["mean"])
    scaler.scale_ = np.array(data["scale"])
    return scaler

def load_feature_names():
    with open(FEATURES_PATH, "r") as f:
        return json.load(f)

def encode_categorical(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

def align_features(df, feature_names):
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

def ks_statistic(y_true, y_proba):
    from scipy.stats import ks_2samp
    return ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic

# App header (same as original)
icon = "nku_icon.png"
left_co, cent_co, last_co = st.columns([0.35, 0.3, 0.35])
with cent_co:
    st.image("ammit.png")
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image(icon, width=100)
with col2:
    st.title("Антифрод ML")

# Sidebar navigation - simplified to two pages
menu = st.sidebar.selectbox(
    "Выберите вкладку:",
    ["🏋️‍♂️ Обучение модели", "🔍 Проверка модели"]
)

if menu == "🏋️‍♂️ Обучение модели":
    st.header("📌 Обучение модели")
    
    # Model selection
    selected_model = st.selectbox(
        "Выберите модель для обучения:",
        list(MODEL_PATHS.keys())
    )
    
    # Data upload
    uploaded_file = st.file_uploader("Загрузите CSV или Excel-файл", type=["csv", "xlsx", "xls"])
    use_sample_data = st.checkbox("Использовать пример данных для демонстрации")
    
    if uploaded_file or use_sample_data:
        if use_sample_data:
            df = save_sample_data()
            st.success("Сгенерирован пример данных для демонстрации")
        else:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        
        st.dataframe(df.head())
        
        if "IsFraud" not in df.columns:
            st.error("❌ Ошибка: В файле нет столбца 'IsFraud'")
        else:
            X, y = df.drop(columns=["IsFraud"]), df["IsFraud"]
            X_raw = X.copy()
            X, label_encoders = encode_categorical(X)
            imputer = SimpleImputer(strategy="mean")
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_original_for_export = X.copy()
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            _, X_test_raw = train_test_split(X_raw, test_size=0.3, random_state=42)
            feature_names = list(X.columns)
            
            # Initialize selected model
            if selected_model == "XGBoost":
                model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, 
                                        random_state=42, eval_metric="logloss", n_jobs=-1)
            elif selected_model == "LogisticRegression":
                model = LogisticRegression()
            elif selected_model == "RandomForest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif selected_model == "GradientBoosting":
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif selected_model == "CatBoost":
                model = CatBoostClassifier(verbose=0, random_state=42)
            elif selected_model == "LightGBM":
                model = LGBMClassifier(random_state=42)
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Predictions
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, y_test_proba)
            gini = 2 * roc_auc - 1
            ks = ks_statistic(y_test.to_numpy(), y_test_proba)

            # Save everything
            save_model(model, feature_names, MODEL_PATHS[selected_model])
            save_label_encoders(label_encoders)
            save_scaler(scaler)

            # Display results (same format as original)
            st.success(f"✅ {selected_model} модель обучена и сохранена")

            st.subheader("📈 Результаты обучения (тест)")
            st.markdown(f"⏱ **Время обучения:** {train_time:.2f} сек")
            st.markdown(f"📊 **Gini индекс:** {gini:.4f}")
            st.markdown(f"📊 **KS индекс:** {ks:.4f}")
            st.markdown(f"✅ **Accuracy:** {accuracy:.4f}")
            st.markdown(f"🎯 **Precision:** {precision:.4f}")
            st.markdown(f"🔁 **Recall:** {recall:.4f}")

            # Metrics table
            metrics_df = pd.DataFrame({
                "Метрика": ["Время обучения (сек)", "Gini", "KS", "Accuracy", "Precision", "Recall", "ROC AUC"],
                "Значение": [round(train_time, 2), round(gini, 4), round(ks, 4), 
                            round(accuracy, 4), round(precision, 4), round(recall, 4), round(roc_auc, 4)]
            })
            st.subheader("📊 Сводная таблица метрик (тест)")
            st.table(metrics_df)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            fig_roc = plt.figure()
            plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC-кривая")
            plt.legend()
            st.subheader("📈 ROC-кривая")
            st.pyplot(fig_roc)

            # Probability distribution
            fig_hist = plt.figure()
            plt.hist(y_test_proba[y_test == 0], bins=30, alpha=0.6, label="Класс 0 (не мошенник)")
            plt.hist(y_test_proba[y_test == 1], bins=30, alpha=0.6, label="Класс 1 (мошенник)")
            plt.title("📊 Распределение вероятностей по классам")
            plt.xlabel("Предсказанная вероятность")
            plt.ylabel("Количество")
            plt.legend()
            st.subheader("🔍 Распределение вероятностей")
            st.pyplot(fig_hist)

            # Features used
            st.subheader("📌 Использованные переменные")
            st.code(", ".join(feature_names))

            # Download test results
            results_df = X_test_raw.copy()
            results_df["Предсказанный класс"] = y_test_pred
            results_df["Вероятность"] = y_test_proba
            results_df["Истинный GB_flag"] = y_test.values

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Скачать результаты на тестовой выборке",
                data=csv,
                file_name="test_predictions.csv",
                mime="text/csv"
            )

elif menu == "🔍 Проверка модели":
    st.header("🔍 Проверка модели")
    
    # Check for trained models
    available_models = [m for m in MODEL_PATHS.keys() if os.path.exists(MODEL_PATHS[m])]
    
    if not available_models:
        st.error("❌ Нет обученных моделей. Сначала обучите модель на вкладке 'Обучение'.")
    else:
        # Model selection
        selected_model = st.selectbox(
            "Выберите модель для проверки:",
            available_models
        )
        
        # Data upload
        uploaded_file = st.file_uploader("Загрузите тестовый файл", type=["csv", "xlsx", "xls"])
        use_sample_data = st.checkbox("Использовать пример данных для проверки")
        
        if uploaded_file or use_sample_data:
            if use_sample_data:
                df = generate_sample_data()
                st.success("Используется пример данных для проверки")
            else:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            
            # Load model and components
            model = load_model(MODEL_PATHS[selected_model], selected_model)
            scaler = load_scaler()
            feature_names = load_feature_names()
            
            # Prepare data
            df_aligned = align_features(df, feature_names)
            imputer = SimpleImputer(strategy="mean")
            df_imputed = pd.DataFrame(imputer.fit_transform(df_aligned), columns=df_aligned.columns)
            df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)
            
            # Make predictions
            predictions = model.predict(df_scaled)
            probabilities = model.predict_proba(df_scaled)[:, 1]
            
            # Create results table
            results = pd.DataFrame({
                'ID': df.index,
                'IsFraud': predictions,
                'Probability': probabilities
            })
            
            # Display results
            st.subheader("🔢 Результаты предсказаний")
            st.dataframe(results)
            
            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать результаты",
                data=csv,
                file_name=f"predictions_{selected_model}.csv",
                mime='text/csv'
            )
            
            # Fraud distribution
            st.subheader("📊 Распределение предсказаний")
            fraud_counts = results['IsFraud'].value_counts()
            st.bar_chart(fraud_counts)