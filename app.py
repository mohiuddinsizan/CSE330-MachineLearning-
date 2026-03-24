import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="COVID Clinical Decision Support Demo",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 COVID-19 Clinical Decision Support Demo")
st.caption("Notebook-inspired UI demo for ICU risk, severity progression, and mortality prediction")


# ============================================================
# CONSTANTS
# ============================================================
DATA_PATH = "COVID-19_RECOVERY_DATASET.csv"

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


# ============================================================
# HELPERS
# ============================================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Clean object columns
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["nan", "None", "NaN", ""]), c] = np.nan

    # Standardize boolean-ish columns if needed
    bool_cols = ["hospitalized", "ventilator_support", "icu_admission", "death"]
    for col in bool_cols:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.lower().map(
                    {
                        "yes": 1, "true": 1, "1": 1,
                        "no": 0, "false": 0, "0": 0
                    }
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_post_admission_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "severity" in df.columns:
        df["severity_score"] = df["severity"].map(SEV_ORDER).fillna(0).astype(int)
    else:
        df["severity_score"] = 0

    high_risk = {"Diabetes", "Heart Disease", "COPD", "Hypertension"}

    if "comorbidities" in df.columns:
        df["comorbidity_risk"] = df["comorbidities"].isin(high_risk).astype(int)
    else:
        df["comorbidity_risk"] = 0

    symptom_cols = [c for c in ["symptoms_1", "symptoms_2", "symptoms_3"] if c in df.columns]
    if symptom_cols:
        df["symptom_count"] = df[symptom_cols].notna().sum(axis=1).astype(int)
    else:
        df["symptom_count"] = 0

    resp = {"Fever", "Cough", "Shortness of Breath", "Loss of Smell"}
    has_resp = []
    for _, row in df.iterrows():
        vals = {str(row.get("symptoms_1", "")), str(row.get("symptoms_2", "")), str(row.get("symptoms_3", ""))}
        has_resp.append(int(any(v in resp for v in vals)))
    df["has_respiratory"] = has_resp

    if "age" in df.columns:
        df["age_x_comorbidity"] = pd.to_numeric(df["age"], errors="coerce").fillna(0) * df["comorbidity_risk"]
        df["age_group"] = pd.cut(
            pd.to_numeric(df["age"], errors="coerce").fillna(0),
            bins=[0, 18, 35, 50, 65, 120],
            labels=["Child", "Young Adult", "Adult", "Older Adult", "Senior"]
        )
    else:
        df["age_x_comorbidity"] = 0
        df["age_group"] = "Adult"

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
    else:
        df["severity_progressed"] = 0

    return df


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )

    return preprocessor, cat_cols, num_cols


def safe_feature_cols(df: pd.DataFrame, target_col: str, extra_drop=None):
    extra_drop = extra_drop or []
    drop_cols = set(IRRELEVANT_COLS + [target_col] + extra_drop)
    return [c for c in df.columns if c not in drop_cols]


def balance_binary_dataset(df: pd.DataFrame, target_col: str, fraction_of_total=0.30, seed=42):
    work = df.dropna(subset=[target_col]).copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col])
    work[target_col] = work[target_col].astype(int)

    class0 = work[work[target_col] == 0]
    class1 = work[work[target_col] == 1]

    if len(class0) == 0 or len(class1) == 0:
        return work

    target_per_class = int(len(work) * fraction_of_total / 2)
    target_per_class = max(50, target_per_class)

    s0 = class0.sample(n=min(target_per_class, len(class0)), random_state=seed)
    s1 = class1.sample(n=min(target_per_class, len(class1)), random_state=seed)

    out = pd.concat([s0, s1]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def balance_multiclass(df: pd.DataFrame, target_col: str, per_class=300, seed=42):
    work = df.dropna(subset=[target_col]).copy()
    parts = []
    for klass in work[target_col].dropna().unique():
        sub = work[work[target_col] == klass]
        parts.append(sub.sample(n=min(per_class, len(sub)), random_state=seed))
    if not parts:
        return work
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def train_binary_model(df: pd.DataFrame, target_col: str, extra_drop=None):
    data = df.dropna(subset=[target_col]).copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[target_col])
    data[target_col] = data[target_col].astype(int)

    features = safe_feature_cols(data, target_col, extra_drop=extra_drop)
    X = data[features].copy()
    y = data[target_col].copy()

    preprocessor, _, _ = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                random_state=42,
                class_weight="balanced"
            )),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "report": classification_report(y_test, preds, output_dict=True),
        "features": features,
    }

    return model, metrics


def train_multiclass_model(df: pd.DataFrame, target_col: str, extra_drop=None):
    data = df.dropna(subset=[target_col]).copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[target_col].astype(str))

    features = safe_feature_cols(data, target_col, extra_drop=extra_drop)
    X = data[features].copy()

    preprocessor, _, _ = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                random_state=42,
                class_weight="balanced"
            )),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "features": features,
        "classes": list(label_encoder.classes_),
    }

    return model, metrics, label_encoder


def build_patient_input_form(df: pd.DataFrame):
    st.subheader("Patient Input Form")

    def options_for(col, fallback):
        if col in df.columns:
            vals = sorted([v for v in df[col].dropna().astype(str).unique().tolist() if v != "nan"])
            return vals[:50] if vals else fallback
        return fallback

    gender_opts = options_for("gender", ["Male", "Female", "Other"])
    vacc_opts = options_for("vaccination_status", ["Vaccinated", "Unvaccinated", "Partially Vaccinated"])
    variant_opts = options_for("variant", ["Delta", "Omicron", "Alpha", "Beta", "Other"])
    country_opts = options_for("country", ["Bangladesh", "India", "USA", "UK"])
    region_opts = options_for("region/state", ["Dhaka", "Chattogram", "Sylhet"])
    comor_opts = options_for("comorbidities", ["None", "Diabetes", "Hypertension", "Heart Disease"])
    sym1_opts = options_for("symptoms_1", ["Fever", "Cough", "Shortness of Breath", "Fatigue"])
    sym2_opts = options_for("symptoms_2", ["None", "Headache", "Loss of Smell", "Chest Pain"])
    sym3_opts = options_for("symptoms_3", ["None", "Myalgia", "Nausea", "Diarrhoea"])
    trt1_opts = options_for("treatment_given_1", ["Oxygen", "Remdesivir", "Dexamethasone", "Supportive"])
    trt2_opts = options_for("treatment_given_2", ["None", "Tocilizumab", "Plasma", "Antiviral"])
    test_opts = options_for("test_type", ["PCR", "Rapid Antigen", "Antibody"])
    severity_opts = options_for("severity", ["Mild", "Moderate", "Severe", "Critical"])

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", 1, 100, 45)
        gender = st.selectbox("Gender", gender_opts)
        vaccination_status = st.selectbox("Vaccination Status", vacc_opts)
        variant = st.selectbox("Variant", variant_opts)
        country = st.selectbox("Country", country_opts)
        region_state = st.selectbox("Region/State", region_opts)

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

    patient = {
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
        "icu_admission": 0,
        "death": 0,
    }

    return pd.DataFrame([patient])


def get_prob_binary(model, patient_df: pd.DataFrame):
    prob = float(model.predict_proba(patient_df)[0][1])
    pred = int(model.predict(patient_df)[0])
    return pred, prob


def get_pred_multiclass(model, label_encoder, patient_df: pd.DataFrame):
    pred_idx = int(model.predict(patient_df)[0])
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(patient_df)[0]
    return pred_label, probs


def triage_text(icu_prob: float, mort_prob: float, severity_label: str):
    sev = str(severity_label).lower()

    if mort_prob >= 0.70 or icu_prob >= 0.70 or sev in ["critical"]:
        return "🚨 High Risk — ICU escalation / urgent specialist review recommended"
    if mort_prob >= 0.40 or icu_prob >= 0.40 or sev in ["severe"]:
        return "⚠️ Medium-High Risk — close hospital monitoring recommended"
    if sev in ["moderate"]:
        return "🟡 Moderate Risk — ward observation / frequent reassessment"
    return "🟢 Lower Risk — supportive care and routine monitoring"


def plot_bar(values: dict, title: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(list(values.keys()), list(values.values()))
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    st.pyplot(fig)


# ============================================================
# LOAD DATA
# ============================================================
st.sidebar.header("Setup")

if "models_ready" not in st.session_state:
    st.session_state.models_ready = False

data_source = st.sidebar.radio(
    "Choose dataset source",
    ["Use local CSV file", "Upload CSV manually"]
)

df = None

if data_source == "Use local CSV file":
    try:
        df = load_data(DATA_PATH)
        st.sidebar.success(f"Loaded local dataset: {DATA_PATH}")
    except Exception as e:
        st.sidebar.error(f"Could not load local file: {e}")
else:
    uploaded = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])
    if uploaded is not None:
        df = load_data(uploaded)
        st.sidebar.success("Uploaded dataset loaded successfully")

if df is None:
    st.info("Add the dataset CSV first, then reload the app.")
    st.stop()

df = add_post_admission_features(df)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Project Overview",
    "Train Models",
    "Demo Prediction",
    "Dataset Explorer",
])

with tab1:
    st.subheader("Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("ICU Column Exists", "Yes" if "icu_admission" in df.columns else "No")
    c4.metric("Mortality Column Exists", "Yes" if "death" in df.columns else "No")

    st.markdown("""
    ### What this demo does
    This app demonstrates a simple clinical decision support UI inspired by your notebook.

    It trains and predicts:
    - **ICU Escalation**
    - **Severity Progression**
    - **Mortality**
    - and gives a simple **triage recommendation**
    """)

    st.subheader("Columns found")
    st.write(df.columns.tolist())

    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

with tab2:
    st.subheader("Train Models")

    st.markdown("Click the button below once. It will train three models from your dataset.")

    if st.button("Train All Models", type="primary"):
        try:
            with st.spinner("Training models..."):

                # Balanced datasets
                df_icu_bal = balance_binary_dataset(df, "icu_admission", fraction_of_total=0.30)
                df_prog_bal = balance_binary_dataset(df, "severity_progressed", fraction_of_total=0.30)
                df_mort_bal = balance_binary_dataset(df, "death", fraction_of_total=0.30)
                df_sev_bal = balance_multiclass(df, "severity", per_class=300)

                # ICU
                icu_extra_drop = ["death", "severity_progressed"]
                icu_model, icu_metrics = train_binary_model(
                    df_icu_bal, "icu_admission", extra_drop=icu_extra_drop
                )

                # Progression
                prog_extra_drop = ["death", "icu_admission"]
                prog_model, prog_metrics = train_binary_model(
                    df_prog_bal, "severity_progressed", extra_drop=prog_extra_drop
                )

                # Mortality
                mort_extra_drop = ["severity_progressed"]
                mort_model, mort_metrics = train_binary_model(
                    df_mort_bal, "death", extra_drop=mort_extra_drop
                )

                # Severity multiclass
                sev_extra_drop = [
                    "icu_admission",
                    "death",
                    "severity_progressed",
                    "severity_score",
                    "sev_x_icu",
                    "sev_x_vent",
                ]
                sev_model, sev_metrics, sev_label_encoder = train_multiclass_model(
                    df_sev_bal, "severity", extra_drop=sev_extra_drop
                )

                st.session_state.icu_model = icu_model
                st.session_state.prog_model = prog_model
                st.session_state.mort_model = mort_model
                st.session_state.sev_model = sev_model
                st.session_state.sev_label_encoder = sev_label_encoder

                st.session_state.icu_metrics = icu_metrics
                st.session_state.prog_metrics = prog_metrics
                st.session_state.mort_metrics = mort_metrics
                st.session_state.sev_metrics = sev_metrics

                st.session_state.models_ready = True

            st.success("All models trained successfully.")

        except Exception as e:
            st.error(f"Training failed: {e}")

    if st.session_state.models_ready:
        st.subheader("Model Performance")

        perf_df = pd.DataFrame([
            {
                "Task": "ICU Escalation",
                "Accuracy": round(st.session_state.icu_metrics["accuracy"], 4),
                "ROC-AUC": round(st.session_state.icu_metrics["roc_auc"], 4),
            },
            {
                "Task": "Severity Progression",
                "Accuracy": round(st.session_state.prog_metrics["accuracy"], 4),
                "ROC-AUC": round(st.session_state.prog_metrics["roc_auc"], 4),
            },
            {
                "Task": "Mortality",
                "Accuracy": round(st.session_state.mort_metrics["accuracy"], 4),
                "ROC-AUC": round(st.session_state.mort_metrics["roc_auc"], 4),
            },
            {
                "Task": "Severity (Multiclass)",
                "Accuracy": round(st.session_state.sev_metrics["accuracy"], 4),
                "ROC-AUC": np.nan,
            },
        ])

        st.dataframe(perf_df, use_container_width=True)

        chart_values = {
            "ICU": st.session_state.icu_metrics["accuracy"],
            "Progression": st.session_state.prog_metrics["accuracy"],
            "Mortality": st.session_state.mort_metrics["accuracy"],
            "Severity": st.session_state.sev_metrics["accuracy"],
        }
        plot_bar(chart_values, "Accuracy by Task")

with tab3:
    st.subheader("Demo Prediction")

    if not st.session_state.models_ready:
        st.warning("Train the models first from the 'Train Models' tab.")
    else:
        patient_df = build_patient_input_form(df)
        patient_df = add_post_admission_features(patient_df)

        if st.button("Run Prediction", type="primary"):
            try:
                # Align patient columns for each task
                icu_features = st.session_state.icu_metrics["features"]
                prog_features = st.session_state.prog_metrics["features"]
                mort_features = st.session_state.mort_metrics["features"]
                sev_features = st.session_state.sev_metrics["features"]

                for col in set(icu_features + prog_features + mort_features + sev_features):
                    if col not in patient_df.columns:
                        patient_df[col] = np.nan

                patient_icu = patient_df[icu_features]
                patient_prog = patient_df[prog_features]
                patient_mort = patient_df[mort_features]
                patient_sev = patient_df[sev_features]

                icu_pred, icu_prob = get_prob_binary(st.session_state.icu_model, patient_icu)
                prog_pred, prog_prob = get_prob_binary(st.session_state.prog_model, patient_prog)
                mort_pred, mort_prob = get_prob_binary(st.session_state.mort_model, patient_mort)
                sev_label, sev_probs = get_pred_multiclass(
                    st.session_state.sev_model,
                    st.session_state.sev_label_encoder,
                    patient_sev
                )

                triage = triage_text(icu_prob, mort_prob, sev_label)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ICU Risk", f"{icu_prob * 100:.1f}%")
                c2.metric("Progression Risk", f"{prog_prob * 100:.1f}%")
                c3.metric("Mortality Risk", f"{mort_prob * 100:.1f}%")
                c4.metric("Predicted Severity", str(sev_label))

                st.subheader("Triage Recommendation")
                st.success(triage)

                st.subheader("Risk Summary")
                risk_df = pd.DataFrame({
                    "Prediction Task": ["ICU Escalation", "Severity Progression", "Mortality"],
                    "Probability": [icu_prob, prog_prob, mort_prob]
                })
                st.dataframe(risk_df, use_container_width=True)

                st.subheader("Severity Class Probabilities")
                sev_class_df = pd.DataFrame({
                    "Severity Class": st.session_state.sev_metrics["classes"],
                    "Probability": sev_probs
                })
                st.dataframe(sev_class_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.bar(sev_class_df["Severity Class"], sev_class_df["Probability"])
                ax.set_title("Severity Probabilities")
                ax.set_ylim(0, 1)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab4:
    st.subheader("Dataset Explorer")

    st.write("Basic statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    if "severity" in df.columns:
        st.subheader("Severity Distribution")
        sev_counts = df["severity"].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(sev_counts.index.astype(str), sev_counts.values)
        ax.set_title("Severity Distribution")
        st.pyplot(fig)

    if "icu_admission" in df.columns:
        st.subheader("ICU Admission Distribution")
        icu_counts = df["icu_admission"].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(icu_counts.index.astype(str), icu_counts.values)
        ax.set_title("ICU Admission Distribution")
        st.pyplot(fig)

    if "death" in df.columns:
        st.subheader("Mortality Distribution")
        death_counts = df["death"].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(death_counts.index.astype(str), death_counts.values)
        ax.set_title("Mortality Distribution")
        st.pyplot(fig)