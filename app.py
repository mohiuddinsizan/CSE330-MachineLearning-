import warnings
warnings.filterwarnings("ignore")

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


st.set_page_config(
    page_title="COVID-19 Clinical Decision Support Presentation App",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 COVID-19 Clinical Decision Support Presentation App")
st.caption("Presentation-ready frontend with EDA, balancing, per-task/per-model training, comparison, and live prediction.")

DEFAULT_DATA_PATH = "COVID-19_RECOVERY_DATASET.csv"

IRRELEVANT_COLS = [
    "patient_id",
    "hospital_name",
    "doctor_assigned",
    "source_url",
    "date_of_recovery",
    "date_of_death",
    "date_reported",
    "recovered",
    "days_to_recovery",
]

SEV_ORDER = {"Mild": 0, "Moderate": 1, "Severe": 2, "Critical": 3}
HIGH_RISK_COMORBIDITIES = {"Diabetes", "Heart Disease", "COPD", "Hypertension"}
RESP_SYMPTOMS = {"Fever", "Cough", "Shortness of Breath", "Loss of Smell"}

MODEL_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"]
TASK_ORDER = ["ICU Escalation", "Severity Progression", "Mortality", "Severity"]

TASK_META = {
    "ICU Escalation": {
        "target_col": "icu_admission",
        "display_classes": ["No ICU", "ICU"],
        "binary": True,
    },
    "Severity Progression": {
        "target_col": "severity_progressed",
        "display_classes": ["Stable", "Progressed"],
        "binary": True,
    },
    "Mortality": {
        "target_col": "death",
        "display_classes": ["Survived", "Died"],
        "binary": True,
    },
    "Severity": {
        "target_col": "severity",
        "display_classes": ["Mild", "Moderate", "Severe", "Critical"],
        "binary": False,
    },
}


@dataclass
class TaskArtifacts:
    task_name: str
    target_col: str
    is_binary: bool
    preprocessor: ColumnTransformer
    feature_names: List[str]
    label_encoder: Optional[LabelEncoder]
    X_train_processed: np.ndarray
    X_test_processed: np.ndarray
    X_train_df: pd.DataFrame
    X_test_df: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    feature_columns_raw: List[str]
    balanced_df: pd.DataFrame
    model_results: pd.DataFrame
    models: Dict[str, object]
    feature_importance_df: pd.DataFrame


def hash_dataframe(df: pd.DataFrame) -> str:
    return hashlib.md5(df.to_csv(index=False).encode("utf-8")).hexdigest()


def safe_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(path: str, uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(path)

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["nan", "None", "NaN", ""]), c] = np.nan

    bool_cols = ["hospitalized", "ventilator_support", "icu_admission", "death", "recovered"]
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = (
                    df[col].str.lower().map(
                        {
                            "yes": 1, "true": 1, "1": 1,
                            "no": 0, "false": 0, "0": 0,
                        }
                    )
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "tests_conducted" in df.columns:
        df["tests_conducted"] = pd.to_numeric(df["tests_conducted"], errors="coerce")

    return df


def add_post_admission_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "severity" in df.columns:
        df["severity_score"] = df["severity"].map(SEV_ORDER).fillna(0).astype(int)
    else:
        df["severity_score"] = 0

    if "comorbidities" in df.columns:
        df["comorbidity_risk"] = df["comorbidities"].isin(HIGH_RISK_COMORBIDITIES).astype(int)
    else:
        df["comorbidity_risk"] = 0

    symptom_cols = [c for c in ["symptoms_1", "symptoms_2", "symptoms_3"] if c in df.columns]
    if symptom_cols:
        df["symptom_count"] = df[symptom_cols].notna().sum(axis=1).astype(int)

        def respiratory_flag(row) -> int:
            vals = {str(row.get(c, "")) for c in symptom_cols}
            return int(any(v in RESP_SYMPTOMS for v in vals))

        df["has_respiratory"] = df.apply(respiratory_flag, axis=1)
    else:
        df["symptom_count"] = 0
        df["has_respiratory"] = 0

    if "age" in df.columns:
        df["age_x_comorbidity"] = pd.to_numeric(df["age"], errors="coerce").fillna(0) * df["comorbidity_risk"]
        age_group = pd.cut(
            pd.to_numeric(df["age"], errors="coerce").fillna(0),
            bins=[0, 40, 60, 80, 120],
            labels=["Young", "Middle", "Senior", "Elderly"],
            include_lowest=True,
        )
        df["age_group"] = age_group.astype(object).where(age_group.notna(), "Middle")
    else:
        df["age_x_comorbidity"] = 0
        df["age_group"] = "Middle"

    if "icu_admission" in df.columns:
        df["sev_x_icu"] = df["severity_score"] * pd.to_numeric(df["icu_admission"], errors="coerce").fillna(0)
    else:
        df["sev_x_icu"] = 0

    if "ventilator_support" in df.columns:
        df["sev_x_vent"] = df["severity_score"] * pd.to_numeric(df["ventilator_support"], errors="coerce").fillna(0)
    else:
        df["sev_x_vent"] = 0

    if "severity" in df.columns:
        df["severity_progressed"] = df["severity"].isin(["Severe", "Critical"]).astype(int)

    return df


def plot_counts(series: pd.Series, title: str, rotation: int = 0):
    counts = series.value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25, axis="y")
    plt.xticks(rotation=rotation, ha="right")
    st.pyplot(fig)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", safe_ohe()),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ], remainder="drop")

    return preprocessor, cat_cols, num_cols


def get_feature_cols(df_data: pd.DataFrame, target_col: str) -> List[str]:
    exclude = set(IRRELEVANT_COLS + [target_col])

    if target_col == "icu_admission":
        exclude.update(["death", "severity_progressed", "severity", "severity_score", "sev_x_icu", "sev_x_vent"])
    elif target_col == "severity_progressed":
        exclude.update(["death", "icu_admission", "severity", "severity_score", "sev_x_icu", "sev_x_vent"])
    elif target_col == "death":
        exclude.update(["severity_progressed"])
    elif target_col == "severity":
        exclude.update(["severity_progressed", "severity_score"])

    return [c for c in df_data.columns if c not in exclude]


def balance_binary_dataset(df: pd.DataFrame, target_col: str, fraction_of_total: float = 0.30, seed: int = 42) -> pd.DataFrame:
    work = df.dropna(subset=[target_col]).copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col])
    work[target_col] = work[target_col].astype(int)

    class0 = work[work[target_col] == 0]
    class1 = work[work[target_col] == 1]

    if len(class0) == 0 or len(class1) == 0:
        return work.copy()

    target_per_class = int(len(work) * fraction_of_total / 2)
    target_per_class = max(50, target_per_class)

    s0 = class0.sample(n=min(target_per_class, len(class0)), random_state=seed)
    s1 = class1.sample(n=min(target_per_class, len(class1)), random_state=seed)

    return pd.concat([s0, s1]).sample(frac=1, random_state=seed).reset_index(drop=True)


def balance_multiclass_dataset(df: pd.DataFrame, target_col: str, per_class: int = 300, seed: int = 42) -> pd.DataFrame:
    work = df.dropna(subset=[target_col]).copy()
    pieces = []
    for klass in work[target_col].dropna().unique():
        sub = work[work[target_col] == klass]
        pieces.append(sub.sample(n=min(per_class, len(sub)), random_state=seed))
    if not pieces:
        return work
    return pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)


def get_model_factories(is_binary: bool) -> Dict[str, object]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            C=0.5,
            class_weight="balanced" if is_binary else None,
            random_state=42,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10 if is_binary else 12,
            min_samples_split=10 if is_binary else 2,
            min_samples_leaf=4 if is_binary else 1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            random_state=42,
        ),
    }

    if HAS_XGBOOST:
        xgb_params = dict(
            n_estimators=300,
            max_depth=5 if is_binary else 6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss" if is_binary else "mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        models["XGBoost"] = XGBClassifier(**xgb_params)

    ordered = {}
    for name in MODEL_ORDER:
        if name in models:
            ordered[name] = models[name]
    return ordered


def evaluate_model(
    model,
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    X_test_proc: np.ndarray,
    y_test: np.ndarray,
    is_binary: bool,
) -> Dict:
    mdl = clone(model)
    mdl.fit(X_train_proc, y_train)

    preds = mdl.predict(X_test_proc)
    probs = mdl.predict_proba(X_test_proc)

    acc = accuracy_score(y_test, preds)
    if is_binary:
        auc = roc_auc_score(y_test, probs[:, 1])
    else:
        auc = roc_auc_score(y_test, probs, multi_class="ovr", average="weighted")

    return {
        "model": mdl,
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "preds": preds,
        "probs": probs,
        "report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }


def feature_importance_from_model(model, feature_names: List[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values("importance", ascending=False)

    if hasattr(model, "coef_"):
        coef = model.coef_
        if np.ndim(coef) == 1:
            vals = np.abs(coef)
        else:
            vals = np.abs(coef).mean(axis=0)
        return pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values("importance", ascending=False)

    return pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})


def build_task_base(df_task: pd.DataFrame, task_name: str) -> TaskArtifacts:
    meta = TASK_META[task_name]
    target_col = meta["target_col"]
    is_binary = meta["binary"]

    data = df_task.dropna(subset=[target_col]).copy()

    label_encoder = None
    if is_binary:
        y = pd.to_numeric(data[target_col], errors="coerce")
        valid_mask = y.notna()
        data = data.loc[valid_mask].copy()
        y = y.loc[valid_mask].astype(int).values
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data[target_col].astype(str))

    feature_cols_raw = get_feature_cols(data, target_col)
    X_raw = data[feature_cols_raw].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor, cat_cols, num_cols = build_preprocessor(X_train_raw)
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    ohe_features = []
    if len(cat_cols) > 0:
        ohe_features = list(preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(cat_cols))
    feature_names = ohe_features + num_cols

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test_raw.index)

    return TaskArtifacts(
        task_name=task_name,
        target_col=target_col,
        is_binary=is_binary,
        preprocessor=preprocessor,
        feature_names=feature_names,
        label_encoder=label_encoder,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        y_train=np.array(y_train),
        y_test=np.array(y_test),
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        feature_columns_raw=feature_cols_raw,
        balanced_df=data,
        model_results=pd.DataFrame(columns=["Model", "Accuracy", "ROC-AUC", "Predictions", "Probabilities", "Report"]),
        models={},
        feature_importance_df=pd.DataFrame(columns=["feature", "importance", "Model"]),
    )


def train_one_model_for_task(task_art: TaskArtifacts, model_name: str) -> TaskArtifacts:
    model_factories = get_model_factories(task_art.is_binary)
    if model_name not in model_factories:
        raise ValueError(f"Model {model_name} is not available.")

    outcome = evaluate_model(
        model_factories[model_name],
        task_art.X_train_processed,
        task_art.y_train,
        task_art.X_test_processed,
        task_art.y_test,
        task_art.is_binary,
    )

    task_art.models[model_name] = outcome["model"]

    new_row = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": outcome["accuracy"],
        "ROC-AUC": outcome["roc_auc"],
        "Predictions": outcome["preds"],
        "Probabilities": outcome["probs"],
        "Report": outcome["report"],
    }])

    existing = task_art.model_results[task_art.model_results["Model"] != model_name].copy()
    task_art.model_results = pd.concat([existing, new_row], ignore_index=True).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    fi = feature_importance_from_model(outcome["model"], task_art.feature_names)
    fi["Model"] = model_name
    existing_fi = task_art.feature_importance_df[task_art.feature_importance_df["Model"] != model_name].copy()
    task_art.feature_importance_df = pd.concat([existing_fi, fi], ignore_index=True)

    return task_art


def prepare_all_task_datasets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df_work = add_post_admission_features(df.copy())

    if "severity" in df_work.columns:
        df_work["severity_progressed"] = df_work["severity"].isin(["Severe", "Critical"]).astype(int)

    datasets = {}
    if "icu_admission" in df_work.columns:
        datasets["ICU Escalation"] = balance_binary_dataset(df_work, "icu_admission", fraction_of_total=0.30)

    if "severity" in df_work.columns:
        sev_bal = balance_multiclass_dataset(df_work, "severity", per_class=300)
        sev_bal = add_post_admission_features(sev_bal)
        sev_bal["severity_progressed"] = sev_bal["severity"].isin(["Severe", "Critical"]).astype(int)
        datasets["Severity"] = sev_bal
        datasets["Severity Progression"] = sev_bal.copy()

    if "death" in df_work.columns:
        datasets["Mortality"] = balance_binary_dataset(df_work, "death", fraction_of_total=0.30)

    return datasets


def overall_summary_from_artifacts(artifacts: Dict[str, TaskArtifacts]) -> pd.DataFrame:
    rows = []
    for task_name in TASK_ORDER:
        if task_name not in artifacts:
            continue
        for _, row in artifacts[task_name].model_results.iterrows():
            rows.append({
                "Task": task_name,
                "Model": row["Model"],
                "Accuracy": row["Accuracy"],
                "ROC-AUC": row["ROC-AUC"],
            })
    return pd.DataFrame(rows)


def best_model_name(task_art: TaskArtifacts) -> Optional[str]:
    if task_art.model_results.empty:
        return None
    return task_art.model_results.iloc[0]["Model"]


def create_patient_input(df: pd.DataFrame) -> pd.DataFrame:
    def safe_choices(col: str, fallback: List[str]) -> List[str]:
        if col in df.columns:
            vals = [str(v) for v in df[col].dropna().unique() if str(v) not in ("nan", "None", "")]
            vals = sorted(vals)
            return vals[:50] if vals else fallback
        return fallback

    gender_opts = safe_choices("gender", ["Male", "Female", "Other"])
    vacc_opts = safe_choices("vaccination_status", ["Vaccinated", "Unvaccinated", "Partially Vaccinated"])
    variant_opts = safe_choices("variant", ["Delta", "Omicron", "Alpha", "Beta", "Other"])
    country_opts = safe_choices("country", ["Bangladesh", "India", "USA", "UK"])
    region_opts = safe_choices("region/state", ["Dhaka", "Chattogram", "Sylhet"])
    comor_opts = safe_choices("comorbidities", ["None", "Diabetes", "Hypertension", "Heart Disease"])
    sym1_opts = safe_choices("symptoms_1", ["Fever", "Cough", "Shortness of Breath", "Fatigue"])
    sym2_opts = safe_choices("symptoms_2", ["None", "Headache", "Loss of Smell", "Chest Pain"])
    sym3_opts = safe_choices("symptoms_3", ["None", "Myalgia", "Nausea", "Diarrhoea"])
    trt1_opts = safe_choices("treatment_given_1", ["Oxygen", "Remdesivir", "Dexamethasone", "Supportive"])
    trt2_opts = safe_choices("treatment_given_2", ["None", "Tocilizumab", "Plasma", "Antiviral"])
    test_opts = safe_choices("test_type", ["PCR", "Rapid Antigen", "Antibody"])
    severity_opts = safe_choices("severity", ["Mild", "Moderate", "Severe", "Critical"])

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", 1, 100, 55)
        gender = st.selectbox("Gender", gender_opts)
        vaccination_status = st.selectbox("Vaccination Status", vacc_opts)
        variant = st.selectbox("Variant", variant_opts)
        country = st.selectbox("Country", country_opts)
        region_state = st.selectbox("Region / State", region_opts)

    with c2:
        comorbidities = st.selectbox("Comorbidity", comor_opts)
        symptoms_1 = st.selectbox("Symptom 1", sym1_opts)
        symptoms_2 = st.selectbox("Symptom 2", sym2_opts)
        symptoms_3 = st.selectbox("Symptom 3", sym3_opts)
        severity = st.selectbox("Current Severity", severity_opts)
        tests_conducted = st.slider("Tests Conducted", 1, 15, 2)

    with c3:
        test_type = st.selectbox("Test Type", test_opts)
        treatment_given_1 = st.selectbox("Treatment 1", trt1_opts)
        treatment_given_2 = st.selectbox("Treatment 2", trt2_opts)
        hospitalized = st.checkbox("Hospitalized", value=True)
        ventilator_support = st.checkbox("Ventilator Support", value=False)
        icu_admission = st.checkbox("Already in ICU", value=False)

    patient = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "vaccination_status": vaccination_status,
        "variant": variant,
        "country": country,
        "region/state": region_state,
        "comorbidities": comorbidities,
        "symptoms_1": symptoms_1,
        "symptoms_2": symptoms_2,
        "symptoms_3": symptoms_3,
        "severity": severity,
        "tests_conducted": tests_conducted,
        "test_type": test_type,
        "treatment_given_1": treatment_given_1,
        "treatment_given_2": treatment_given_2,
        "hospitalized": int(hospitalized),
        "ventilator_support": int(ventilator_support),
        "icu_admission": int(icu_admission),
        "death": 0,
    }])

    return add_post_admission_features(patient)


def align_patient_to_task(patient_df: pd.DataFrame, task_art: TaskArtifacts) -> pd.DataFrame:
    temp = patient_df.copy()
    for col in task_art.feature_columns_raw:
        if col not in temp.columns:
            temp[col] = np.nan
    return temp[task_art.feature_columns_raw]


def predict_for_task(task_art: TaskArtifacts, patient_df: pd.DataFrame, model_name: str) -> Dict:
    if model_name not in task_art.models:
        raise ValueError(f"{model_name} is not trained yet for {task_art.task_name}.")

    model = task_art.models[model_name]
    patient_task = align_patient_to_task(patient_df, task_art)
    patient_proc = task_art.preprocessor.transform(patient_task)
    probs = model.predict_proba(patient_proc)
    pred_idx = int(model.predict(patient_proc)[0])

    if task_art.is_binary:
        label = TASK_META[task_art.task_name]["display_classes"][pred_idx]
        prob = float(probs[0, 1])
    else:
        label = task_art.label_encoder.inverse_transform([pred_idx])[0]
        prob = float(np.max(probs[0]))

    return {
        "label": label,
        "probability": prob,
        "all_probabilities": probs[0],
    }


def triage_text(icu_prob: float, mort_prob: float, severity_label: str) -> str:
    sev = str(severity_label).lower()
    if mort_prob >= 0.70 or icu_prob >= 0.60 or sev == "critical":
        return "🔴 HIGH RISK — Immediate ICU consideration / urgent escalation"
    if mort_prob >= 0.40 or icu_prob >= 0.35 or sev == "severe":
        return "🟡 MEDIUM RISK — Close monitoring / step-down care"
    return "🟢 LOWER RISK — Standard care with routine reassessment"


def plot_model_comparison(task_art: TaskArtifacts):
    if task_art.model_results.empty:
        st.info("No model trained yet for this task.")
        return

    df = task_art.model_results.copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["Accuracy"], w, label="Accuracy")
    ax.bar(x + w / 2, df["ROC-AUC"], w, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"{task_art.task_name} — Model Comparison")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    st.pyplot(fig)


def plot_roc_for_task(task_art: TaskArtifacts):
    if task_art.model_results.empty:
        st.info("No model trained yet for this task.")
        return

    if not task_art.is_binary:
        st.info("ROC curves are shown only for binary tasks. Severity uses weighted multiclass ROC-AUC in the table.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in task_art.model_results.iterrows():
        probs = row["Probabilities"][:, 1]
        fpr, tpr, _ = roc_curve(task_art.y_test, probs)
        ax.plot(fpr, tpr, label=f"{row['Model']} (AUC={row['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"{task_art.task_name} — ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


def plot_confusion_for_model(task_art: TaskArtifacts, model_name: str):
    if model_name not in task_art.models:
        st.info("Train this model first.")
        return

    row = task_art.model_results.loc[task_art.model_results["Model"] == model_name].iloc[0]
    preds = row["Predictions"]
    cm = confusion_matrix(task_art.y_test, preds)

    labels = TASK_META[task_art.task_name]["display_classes"] if task_art.is_binary else list(task_art.label_encoder.classes_)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)


def plot_feature_importance(task_art: TaskArtifacts, model_name: str, top_n: int = 15):
    if task_art.feature_importance_df.empty:
        st.info("No feature importance available yet.")
        return

    fi = task_art.feature_importance_df
    fi = fi[fi["Model"] == model_name].copy().sort_values("importance", ascending=False).head(top_n)
    if fi.empty or float(fi["importance"].sum()) == 0:
        st.info(f"{model_name} does not expose straightforward feature importances for display here.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1])
    ax.set_title(f"Top {top_n} Features — {model_name}")
    ax.set_xlabel("Importance")
    st.pyplot(fig)


def plot_overall_comparison(summary_df: pd.DataFrame):
    if summary_df.empty:
        st.info("No trained models yet.")
        return

    tasks = [t for t in TASK_ORDER if t in summary_df["Task"].unique()]
    models = [m for m in MODEL_ORDER if m in summary_df["Model"].unique()]
    x = np.arange(len(tasks))
    width = 0.18 if len(models) >= 4 else 0.22

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, model_name in enumerate(models):
        aucs = []
        for task in tasks:
            row = summary_df[(summary_df["Task"] == task) & (summary_df["Model"] == model_name)]
            aucs.append(float(row["ROC-AUC"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + i * width - (len(models) - 1) * width / 2, aucs, width, label=model_name)

    ax.set_title("Overall ROC-AUC Comparison Across Tasks")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=10)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=9)
    st.pyplot(fig)


def optional_shap_section(task_art: TaskArtifacts, selected_model_name: str):
    if selected_model_name != "XGBoost":
        st.info("SHAP section is enabled only for XGBoost.")
        return
    if not HAS_XGBOOST or not HAS_SHAP:
        st.warning("Install both `xgboost` and `shap` to show SHAP explainability.")
        return

    with st.expander("Show SHAP explainability for this task"):
        try:
            model = task_art.models[selected_model_name]
            explainer = shap.TreeExplainer(model)
            sample_df = task_art.X_test_df.iloc[: min(120, len(task_art.X_test_df))]
            shap_values = explainer(sample_df)

            st.write("Global feature importance (mean absolute SHAP)")
            fig1 = plt.figure()
            shap.plots.bar(shap_values, max_display=15, show=False)
            st.pyplot(fig1)

            st.write("Beeswarm summary")
            fig2 = plt.figure()
            shap.plots.beeswarm(shap_values, max_display=15, show=False)
            st.pyplot(fig2)
        except Exception as exc:
            st.warning(f"SHAP could not be rendered here: {exc}")


st.sidebar.header("Setup")
data_mode = st.sidebar.radio("Dataset source", ["Use local CSV file", "Upload CSV manually"])
uploaded_file = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"]) if data_mode == "Upload CSV manually" else None

try:
    raw_df = load_data(DEFAULT_DATA_PATH, uploaded_file)
    st.sidebar.success(f"Dataset loaded: {len(raw_df):,} rows")
except Exception as exc:
    st.sidebar.error(f"Dataset could not be loaded: {exc}")
    st.info("Put `COVID-19_RECOVERY_DATASET.csv` in the same folder as `app.py`, or upload it from the sidebar.")
    st.stop()

df_hash = hash_dataframe(raw_df)

if "training_state" not in st.session_state or st.session_state.training_state.get("df_hash") != df_hash:
    st.session_state.training_state = {
        "df_hash": df_hash,
        "task_datasets": prepare_all_task_datasets(raw_df),
        "artifacts": {},
    }

task_datasets = st.session_state.training_state["task_datasets"]
artifacts = st.session_state.training_state["artifacts"]

for task_name in TASK_ORDER:
    if task_name in task_datasets and task_name not in artifacts:
        artifacts[task_name] = build_task_base(task_datasets[task_name], task_name)

overview_tab, eda_tab, balanced_tab, training_tab, prediction_tab = st.tabs(
    ["Overview", "EDA", "Balanced Datasets", "Training & Comparison", "Live Prediction"]
)

with overview_tab:
    st.markdown("""

    """)
    st.dataframe(raw_df.head(10), use_container_width=True)

with eda_tab:
    st.subheader("Exploratory Data Analysis")
    c1, c2 = st.columns(2)

    with c1:
        if "severity" in raw_df.columns:
            plot_counts(raw_df["severity"], "Severity Distribution")
        if "icu_admission" in raw_df.columns:
            plot_counts(raw_df["icu_admission"], "ICU Admission Distribution")
    with c2:
        if "death" in raw_df.columns:
            plot_counts(raw_df["death"], "Mortality Distribution")
        if "variant" in raw_df.columns:
            top_variant = raw_df["variant"].fillna("Missing")
            top_variant = top_variant.where(top_variant.isin(top_variant.value_counts().head(10).index), "Other")
            plot_counts(top_variant, "Variant Distribution (Top 10)", rotation=20)

with balanced_tab:
    st.subheader("Balanced Dataset Creation")
    rows = []
    for task_name, df_task in task_datasets.items():
        rows.append({
            "Task": task_name,
            "Rows in balanced set": len(df_task),
            "Target column": TASK_META[task_name]["target_col"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    for task_name in TASK_ORDER:
        if task_name not in task_datasets:
            continue
        st.markdown(f"#### {task_name}")
        target_col = TASK_META[task_name]["target_col"]
        c1, c2 = st.columns(2)
        with c1:
            st.write("Raw distribution")
            raw_work = add_post_admission_features(raw_df.copy())
            if target_col in raw_work.columns:
                plot_counts(raw_work[target_col], f"Raw — {target_col}")
        with c2:
            st.write("Balanced distribution")
            plot_counts(task_datasets[task_name][target_col], f"Balanced — {target_col}")

with training_tab:
    st.subheader("Training & Comparison")

    st.markdown("### Quick training")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Train all tasks + all models", type="primary"):
            progress = st.progress(0, text="Starting training...")
            available_tasks = [t for t in TASK_ORDER if t in artifacts]
            total_steps = sum(len(get_model_factories(artifacts[t].is_binary)) for t in available_tasks)
            step = 0
            for task_name in available_tasks:
                for model_name in get_model_factories(artifacts[task_name].is_binary).keys():
                    artifacts[task_name] = train_one_model_for_task(artifacts[task_name], model_name)
                    step += 1
                    progress.progress(int(step / total_steps * 100), text=f"Training {task_name} — {model_name}")
            st.success("All tasks and all models trained.")

    with col_b:
        st.write("Status")
        status_rows = []
        for task_name in TASK_ORDER:
            if task_name not in artifacts:
                continue
            status_rows.append({
                "Task": task_name,
                "Trained models": len(artifacts[task_name].models),
                "Available models": len(get_model_factories(artifacts[task_name].is_binary)),
            })
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True)

    st.markdown("### Individual training")
    train_task = st.selectbox("Choose task", [t for t in TASK_ORDER if t in artifacts], key="ind_train_task")
    available_model_names = list(get_model_factories(artifacts[train_task].is_binary).keys())
    train_model_name = st.selectbox("Choose model", available_model_names, key="ind_train_model")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Train selected model only"):
            artifacts[train_task] = train_one_model_for_task(artifacts[train_task], train_model_name)
            st.success(f"Trained {train_model_name} for {train_task}.")

    with c2:
        if st.button("Train all models for selected task"):
            for model_name in available_model_names:
                artifacts[train_task] = train_one_model_for_task(artifacts[train_task], model_name)
            st.success(f"Trained all models for {train_task}.")

    summary_df = overall_summary_from_artifacts(artifacts)

    st.markdown("### Best trained model per task")
    best_rows = []
    for task_name in TASK_ORDER:
        if task_name not in artifacts:
            continue
        best_name = best_model_name(artifacts[task_name])
        if best_name is None:
            best_rows.append({"Task": task_name, "Best trained model": "None yet", "Accuracy": np.nan, "ROC-AUC": np.nan})
        else:
            row = artifacts[task_name].model_results.iloc[0]
            best_rows.append({
                "Task": task_name,
                "Best trained model": best_name,
                "Accuracy": row["Accuracy"],
                "ROC-AUC": row["ROC-AUC"],
            })
    st.dataframe(pd.DataFrame(best_rows).round(4), use_container_width=True)

    st.markdown("### Overall trained-model comparison")
    if summary_df.empty:
        st.info("No model has been trained yet.")
    else:
        st.dataframe(summary_df.round(4), use_container_width=True)
        plot_overall_comparison(summary_df)

    st.markdown("### Deep inspection")
    inspect_task = st.selectbox("Choose a task to inspect", [t for t in TASK_ORDER if t in artifacts], key="inspect_task")
    task_art = artifacts[inspect_task]

    if task_art.model_results.empty:
        st.info("Train at least one model for this task first.")
    else:
        st.dataframe(task_art.model_results[["Model", "Accuracy", "ROC-AUC"]].round(4), use_container_width=True)
        plot_model_comparison(task_art)
        plot_roc_for_task(task_art)

        inspect_model = st.selectbox("Choose trained model to inspect", task_art.model_results["Model"].tolist(), key="inspect_model")
        c1, c2 = st.columns(2)
        with c1:
            plot_confusion_for_model(task_art, inspect_model)
        with c2:
            plot_feature_importance(task_art, inspect_model, top_n=15)

        optional_shap_section(task_art, inspect_model)

with prediction_tab:
    st.subheader("Live Prediction")
    st.write("Only already-trained models appear here.")

    patient_df = create_patient_input(raw_df)

    selectable_tasks = [t for t in TASK_ORDER if t in artifacts]
    chosen_models = {}

    cols = st.columns(4)
    col_idx = 0
    for task_name in selectable_tasks:
        trained_models = artifacts[task_name].model_results["Model"].tolist()
        with cols[col_idx % 4]:
            if trained_models:
                default_model = best_model_name(artifacts[task_name])
                chosen_models[task_name] = st.selectbox(
                    task_name,
                    trained_models,
                    index=trained_models.index(default_model) if default_model in trained_models else 0,
                    key=f"pred_{task_name}",
                )
                st.caption(f"Using trained model list only")
            else:
                st.warning(f"No trained model yet for {task_name}")
        col_idx += 1

    if st.button("Run live prediction", type="primary"):
        if not chosen_models:
            st.error("Train at least one model first.")
        else:
            results = {}
            for task_name, model_name in chosen_models.items():
                results[task_name] = predict_for_task(artifacts[task_name], patient_df, model_name)

            icu_prob = results.get("ICU Escalation", {}).get("probability", 0.0)
            mort_prob = results.get("Mortality", {}).get("probability", 0.0)
            severity_label = results.get("Severity", {}).get("label", "Unknown")

            triage = triage_text(icu_prob, mort_prob, severity_label)

            st.markdown("### Prediction outputs")
            metric_cols = st.columns(max(1, len(results)))
            for i, task_name in enumerate(results.keys()):
                out = results[task_name]
                with metric_cols[i]:
                    st.metric(task_name, out["label"], f"{out['probability'] * 100:.1f}%")
                    st.caption(f"Model used: {chosen_models[task_name]}")

            st.markdown("### Triage recommendation")
            st.success(triage)

            st.markdown("### Detailed probability tables")
            for task_name, out in results.items():
                st.markdown(f"#### {task_name}")
                if TASK_META[task_name]["binary"]:
                    prob_df = pd.DataFrame({
                        "Class": TASK_META[task_name]["display_classes"],
                        "Probability": out["all_probabilities"],
                    })
                else:
                    prob_df = pd.DataFrame({
                        "Class": list(artifacts[task_name].label_encoder.classes_),
                        "Probability": out["all_probabilities"],
                    })
                st.dataframe(prob_df.round(4), use_container_width=True)

                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.bar(prob_df["Class"], prob_df["Probability"])
                ax.set_ylim(0, 1)
                ax.set_title(f"{task_name} probabilities")
                ax.grid(True, alpha=0.25, axis="y")
                st.pyplot(fig)

    st.markdown("### Patient row used for prediction")
    st.dataframe(patient_df, use_container_width=True)

st.markdown("---")
st.markdown(
    ""
)