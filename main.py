import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler

# Loading data
df = pd.read_csv("account_data.csv")

# GB Flag = "IsFraud" column, binary classification task
y = df["IsFraud"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["IsFraud"])

# Detect categorical columns and encode them using LabelEncoder as str
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Balancing samples using RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# Classifier models selection
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.1, max_depth=6, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42, scale_pos_weight=scale_pos_weight),
    "LightGBM": LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                             scale_pos_weight=scale_pos_weight, random_state=42)
}

# Training and evaluation of models, training results output
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = None

    print(f"\n==== {name} ====")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")

# Model inference. Takes a csv file as input and outputs the predicted class of fraud for each model, i.e. 0 or 1.
def predict_fraud(input_file):
    df_input = pd.read_csv(input_file)
    
    for col in df_input.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_input[col] = le.fit_transform(df_input[col].astype(str))
    
    df_input_scaled = scaler.transform(df_input)
    
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(df_input_scaled)
    
    return pd.DataFrame(predictions)

predict_fraud("new_account_data.csv").to_csv("predictions.csv", index=True)