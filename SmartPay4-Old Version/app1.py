
# =============================================================================
# SMARTPAY - EMPLOYEE SALARY INTELLIGENCE
# =============================================================================
#
# Required files in the same directory:
#
#   best_salary_prediction_model.pkl
#   label_encoders.pkl
#   feature_scaler.pkl
#   model_metadata.pkl
#
# Optional dataset files:
#
#   cleaned_data.csv
#   DATASET.csv
#
# Evaluation images:
#
#   best_model_actual_vs_predicted.png
#   best_model_feature_importance.png
#   best_model_residuals.png
#   correlation_heatmap.png
#   feature_importance.png
#   model_comprehensive_comparison.png
#   model_cv_comparison.png
#   model_error_comparison.png
#   model_r2_comparison.png
#   r2_rmse_scores.png
#   residuals_plot.png
#
# Run:
#
#   streamlit run app.py
#
# =============================================================================


import pickle
import pathlib
import io
import traceback

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SmartPay | Salary Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# BASE DIRECTORY
# =============================================================================

BASE_DIR = pathlib.Path(__file__).parent.resolve()


# =============================================================================
# FILE PATHS
# =============================================================================

MODEL_FILE = BASE_DIR / "best_salary_prediction_model.pkl"
ENCODER_FILE = BASE_DIR / "label_encoders.pkl"
SCALER_FILE = BASE_DIR / "feature_scaler.pkl"
METADATA_FILE = BASE_DIR / "model_metadata.pkl"


DATASET_FILES = [
    BASE_DIR / "cleaned_data.csv",
    BASE_DIR / "DATASET.csv",
]


# =============================================================================
# REQUIRED MODEL FEATURES
# =============================================================================
#
# These are the RAW columns expected from the user.
#
# The trained model, according to the error you reported, expects:
#
#   Age
#   Experience
#   Gender_encoded
#   Education_encoded
#   Occupation_encoded
#
# We therefore encode the categorical variables before prediction.
# =============================================================================

RAW_FEATURES = [
    "Age",
    "Experience",
    "Gender",
    "Education",
    "Occupation",
]


MODEL_FEATURES = [
    "Age",
    "Experience",
    "Gender_encoded",
    "Education_encoded",
    "Occupation_encoded",
]


# =============================================================================
# EVALUATION IMAGE FILES
# =============================================================================

EVALUATION_IMAGES = {
    "Best Model — Actual vs Predicted":
        "best_model_actual_vs_predicted.png",

    "Best Model — Feature Importance":
        "best_model_feature_importance.png",

    "Best Model — Residuals":
        "best_model_residuals.png",

    "Correlation Heatmap":
        "correlation_heatmap.png",

    "Feature Importance":
        "feature_importance.png",

    "Model Comprehensive Comparison":
        "model_comprehensive_comparison.png",

    "Model CV Comparison":
        "model_cv_comparison.png",

    "Model Error Comparison":
        "model_error_comparison.png",

    "Model R² Comparison":
        "model_r2_comparison.png",

    "R² / RMSE Scores":
        "r2_rmse_scores.png",

    "Residuals Plot":
        "residuals_plot.png",
}


# =============================================================================
# DARK THEME
# =============================================================================

st.markdown(
    """
    <style>

    /* =========================================================
       GLOBAL
       ========================================================= */

    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0b1120;
    }

    [data-testid="stHeader"] {
        background-color: #0b1120;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =========================================================
       TEXT
       ========================================================= */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #f8fafc !important;
    }

    p {
        color: #cbd5e1;
    }

    label {
        color: #e2e8f0 !important;
    }


    /* =========================================================
       SIDEBAR
       ========================================================= */

    [data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] > div {
        background-color: #020617 !important;
    }


    /* =========================================================
       METRICS
       ========================================================= */

    [data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid #263244 !important;
        border-radius: 14px !important;
        padding: 16px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    [data-testid="stMetricDelta"] {
        color: #4ade80 !important;
    }


    /* =========================================================
       INPUTS
       ========================================================= */

    input,
    textarea {
        background-color: #111827 !important;
        color: #f8fafc !important;
        border: 1px solid #374151 !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #64748b !important;
    }


    /* =========================================================
       SELECTBOX
       ========================================================= */

    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        border-color: #374151 !important;
    }

    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }

    div[role="listbox"] {
        background-color: #111827 !important;
        border-color: #374151 !important;
    }

    div[role="option"] {
        background-color: #111827 !important;
        color: #f8fafc !important;
    }

    div[role="option"]:hover {
        background-color: #1f2937 !important;
    }


    /* =========================================================
       BUTTONS
       ========================================================= */

    .stButton > button {
        background-color: #16a34a !important;
        color: #ffffff !important;
        border: 1px solid #22c55e !important;
        border-radius: 10px !important;
        min-height: 45px;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        background-color: #22c55e !important;
        color: #052e16 !important;
        border-color: #4ade80 !important;
    }


    /* =========================================================
       DOWNLOAD BUTTON
       ========================================================= */

    [data-testid="stDownloadButton"] button {
        background-color: #172033 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        background-color: #1e293b !important;
        border-color: #22c55e !important;
        color: #4ade80 !important;
    }


    /* =========================================================
       FILE UPLOADER
       ========================================================= */

    [data-testid="stFileUploader"] {
        background-color: #111827 !important;
        border: 1px dashed #475569 !important;
        border-radius: 14px !important;
        padding: 10px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #111827 !important;
    }


    /* =========================================================
       DATAFRAME
       ========================================================= */

    [data-testid="stDataFrame"] {
        border: 1px solid #263244 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }


    /* =========================================================
       EXPANDERS
       ========================================================= */

    [data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #263244 !important;
        border-radius: 12px !important;
    }


    /* =========================================================
       ALERTS
       ========================================================= */

    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }


    /* =========================================================
       TABS
       ========================================================= */

    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4ade80 !important;
    }


    /* =========================================================
       IMAGES
       ========================================================= */

    [data-testid="stImage"] {
        background-color: #111827 !important;
        border: 1px solid #263244 !important;
        border-radius: 12px !important;
        padding: 8px !important;
    }


    /* =========================================================
       DIVIDERS
       ========================================================= */

    hr {
        border-color: #1e293b !important;
    }


    /* =========================================================
       CHECKBOX
       ========================================================= */

    [data-testid="stCheckbox"] label {
        color: #e2e8f0 !important;
    }


    /* =========================================================
       RADIO
       ========================================================= */

    [data-testid="stRadio"] label {
        color: #e2e8f0 !important;
    }


    /* =========================================================
       SCROLLBAR
       ========================================================= */

    ::-webkit-scrollbar {
        width: 9px;
        height: 9px;
    }

    ::-webkit-scrollbar-track {
        background: #020617;
    }

    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def safe_float(value):
    """
    Safely convert a value to float.
    """
    try:
        result = float(value)

        if not np.isfinite(result):
            return None

        return result

    except Exception:
        return None


def money(value):
    """
    Format numeric value as Indian Rupees.
    """
    number = safe_float(value)

    if number is None:
        return "N/A"

    return f"₹{number:,.2f}"


def metadata_value(key, default=None):
    """
    Safely read metadata.
    """
    if not isinstance(metadata, dict):
        return default

    return metadata.get(key, default)


def normalize_string(value):
    """
    Normalize categorical input to a clean string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


# =============================================================================
# LOAD MODEL
# =============================================================================


@st.cache_resource
def load_model_and_objects():

    required_files = [
        MODEL_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        METADATA_FILE,
    ]

    missing_files = [
        file.name
        for file in required_files
        if not file.exists()
    ]

    if missing_files:

        raise FileNotFoundError(
            "The following required files are missing:\n\n"
            + "\n".join(
                f"• {name}"
                for name in missing_files
            )
        )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    with open(MODEL_FILE, "rb") as file:
        loaded_model = pickle.load(file)

    # ---------------------------------------------------------
    # Label encoders
    # ---------------------------------------------------------

    with open(ENCODER_FILE, "rb") as file:
        loaded_encoders = pickle.load(file)

    # ---------------------------------------------------------
    # Scaler
    # ---------------------------------------------------------

    with open(SCALER_FILE, "rb") as file:
        loaded_scaler = pickle.load(file)

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    with open(METADATA_FILE, "rb") as file:
        loaded_metadata = pickle.load(file)

    return (
        loaded_model,
        loaded_encoders,
        loaded_scaler,
        loaded_metadata,
    )


# =============================================================================
# INITIALIZE MODEL
# =============================================================================


try:

    (
        model,
        label_encoders,
        scaler,
        metadata,
    ) = load_model_and_objects()

except Exception as error:

    st.error(
        "❌ SmartPay could not load the trained model."
    )

    st.exception(error)

    st.stop()


# =============================================================================
# VALIDATE LOADED OBJECTS
# =============================================================================


missing_encoders = [
    name
    for name in [
        "Gender",
        "Education",
        "Occupation",
    ]
    if name not in label_encoders
]


if missing_encoders:

    st.error(
        "❌ Missing label encoders: "
        + ", ".join(missing_encoders)
    )

    st.stop()


# =============================================================================
# ENCODING FUNCTION
# =============================================================================


def encode_features(df):
    """
    Convert raw employee columns into exactly the columns used
    when the trained model was fitted.

    RAW:
        Age
        Experience
        Gender
        Education
        Occupation

    MODEL:
        Age
        Experience
        Gender_encoded
        Education_encoded
        Occupation_encoded
    """

    data = df.copy()

    # ---------------------------------------------------------
    # Verify raw columns
    # ---------------------------------------------------------

    missing = [
        column
        for column in RAW_FEATURES
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------
    # Numeric columns
    # ---------------------------------------------------------

    data["Age"] = pd.to_numeric(
        data["Age"],
        errors="coerce",
    )

    data["Experience"] = pd.to_numeric(
        data["Experience"],
        errors="coerce",
    )

    if data["Age"].isna().any():

        raise ValueError(
            "Age contains missing or non-numeric values."
        )

    if data["Experience"].isna().any():

        raise ValueError(
            "Experience contains missing or non-numeric values."
        )

    # ---------------------------------------------------------
    # Encode categorical variables
    # ---------------------------------------------------------

    for column in [
        "Gender",
        "Education",
        "Occupation",
    ]:

        encoder = label_encoders[column]

        values = data[column].map(
            normalize_string
        )

        if values.eq("").any():

            raise ValueError(
                f"{column} contains empty values."
            )

        # Convert encoder classes to strings for robust comparison.
        known_classes = {
            str(value)
            for value in encoder.classes_
        }

        unknown_values = sorted(
            set(values.unique())
            - known_classes
        )

        if unknown_values:

            raise ValueError(
                f"Unknown {column} value(s): "
                + ", ".join(
                    map(
                        str,
                        unknown_values,
                    )
                )
                + ". "
                + f"Valid values include: "
                + ", ".join(
                    map(
                        str,
                        encoder.classes_,
                    )
                )
            )

        # IMPORTANT:
        # LabelEncoder.transform expects values in the same
        # representation as the encoder was trained with.
        #
        # We find the exact original class corresponding to
        # the cleaned string.

        class_lookup = {
            str(value): value
            for value in encoder.classes_
        }

        exact_values = values.map(
            lambda value:
                class_lookup[value]
        )

        data[f"{column}_encoded"] = (
            encoder.transform(
                exact_values
            )
        )

    # ---------------------------------------------------------
    # Exact model feature order
    # ---------------------------------------------------------

    X = data[
        MODEL_FEATURES
    ].copy()

    X = X.astype(float)

    # ---------------------------------------------------------
    # Scaling
    # ---------------------------------------------------------

    needs_scaling = bool(
        metadata_value(
            "needs_scaling",
            False,
        )
    )

    if needs_scaling:

        transformed = scaler.transform(X)

        # If scaler remembers feature names,
        # preserve those names.

        if hasattr(
            scaler,
            "feature_names_in_",
        ):

            X = pd.DataFrame(
                transformed,
                columns=list(
                    scaler.feature_names_in_
                ),
                index=data.index,
            )

        else:

            X = pd.DataFrame(
                transformed,
                columns=MODEL_FEATURES,
                index=data.index,
            )

    return X


# =============================================================================
# MODEL PREDICTION
# =============================================================================


def predict_dataframe(df):
    """
    Predict salary for one or many employees.
    """

    try:

        X = encode_features(df)

        predictions = model.predict(X)

        predictions = np.asarray(
            predictions,
            dtype=float,
        ).reshape(-1)

        if not np.all(
            np.isfinite(predictions)
        ):

            raise ValueError(
                "The model returned invalid numeric predictions."
            )

        return predictions

    except Exception as error:

        raise RuntimeError(
            f"Prediction failed: {error}"
        ) from error


# =============================================================================
# EMPLOYEE VALIDATION
# =============================================================================


def validate_employee(
    age,
    experience,
    gender,
    education,
    occupation,
):
    """
    Business/logical validation rules.
    """

    errors = []
    warnings = []

    # ---------------------------------------------------------
    # Age
    # ---------------------------------------------------------

    age_value = safe_float(age)

    if age_value is None:

        errors.append(
            "Age must be numeric."
        )

    else:

        if age_value < 18:

            errors.append(
                "Age cannot be below 18."
            )

        if age_value > 65:

            errors.append(
                "Age cannot exceed 65."
            )

    # ---------------------------------------------------------
    # Experience
    # ---------------------------------------------------------

    experience_value = safe_float(
        experience
    )

    if experience_value is None:

        errors.append(
            "Experience must be numeric."
        )

    else:

        if experience_value < 0:

            errors.append(
                "Experience cannot be negative."
            )

    # ---------------------------------------------------------
    # Age / Experience relationship
    # ---------------------------------------------------------

    if (
        age_value is not None
        and experience_value is not None
    ):

        maximum_reasonable_experience = (
            age_value - 18
        )

        if (
            experience_value
            > maximum_reasonable_experience
        ):

            errors.append(
                f"{experience_value:.1f} years of experience "
                f"is inconsistent with age {age_value:.0f}. "
                f"Maximum reasonable experience is approximately "
                f"{maximum_reasonable_experience:.0f} years."
            )

        if (
            age_value <= 22
            and experience_value >= 5
        ):

            warnings.append(
                "The employee is relatively young but has "
                "high reported experience. Please verify the input."
            )

    # ---------------------------------------------------------
    # Category validation
    # ---------------------------------------------------------

    category_values = {
        "Gender": gender,
        "Education": education,
        "Occupation": occupation,
    }

    for column, value in category_values.items():

        cleaned = normalize_string(
            value
        )

        if not cleaned:

            errors.append(
                f"{column} cannot be empty."
            )

            continue

        encoder = label_encoders.get(
            column
        )

        if encoder is None:

            errors.append(
                f"No encoder was found for {column}."
            )

            continue

        known = {
            str(item)
            for item in encoder.classes_
        }

        if cleaned not in known:

            errors.append(
                f"{column} '{cleaned}' was not "
                "present in the training data."
            )

    # ---------------------------------------------------------
    # Return
    # ---------------------------------------------------------

    return errors, warnings


# =============================================================================
# SINGLE PREDICTION
# =============================================================================


def predict_single(
    age,
    experience,
    gender,
    education,
    occupation,
):

    errors, warnings = validate_employee(
        age,
        experience,
        gender,
        education,
        occupation,
    )

    if errors:

        return (
            None,
            errors,
            warnings,
        )

    row = pd.DataFrame(
        [
            {
                "Age": age,
                "Experience": experience,
                "Gender": gender,
                "Education": education,
                "Occupation": occupation,
            }
        ]
    )

    prediction = predict_dataframe(
        row
    )[0]

    return (
        float(prediction),
        errors,
        warnings,
    )


# =============================================================================
# TARGET COLUMN DETECTION
# =============================================================================


def find_target_column(df):

    possible_targets = [
        "Salary",
        "salary",
        "SALARY",
        "Target",
        "target",
        "Annual Salary",
        "Annual_Salary",
    ]

    for column in possible_targets:

        if column in df.columns:

            return column

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace("_", " ")
        )

        if normalized in [
            "salary",
            "annual salary",
            "target",
        ]:

            return column

    return None


# =============================================================================
# METRIC CALCULATION
# =============================================================================


def calculate_metrics(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        &
        np.isfinite(predicted)
    )

    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:

        return {}

    error = (
        actual - predicted
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    ss_res = float(
        np.sum(
            error ** 2
        )
    )

    ss_tot = float(
        np.sum(
            (
                actual
                - np.mean(actual)
            ) ** 2
        )
    )

    if ss_tot == 0:

        r2 = np.nan

    else:

        r2 = float(
            1
            - (
                ss_res
                / ss_tot
            )
        )

    return {
        "R²": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Samples": len(actual),
    }


# =============================================================================
# LOAD DATASET
# =============================================================================


def load_available_dataset():

    for path in DATASET_FILES:

        if path.exists():

            try:

                dataset = pd.read_csv(
                    path
                )

                return (
                    dataset,
                    path.name,
                )

            except Exception:
                continue

    return (
        None,
        None,
    )


# =============================================================================
# GET MODEL NAME
# =============================================================================


model_name = str(
    metadata_value(
        "model_name",
        "Best Salary Prediction Model",
    )
)


# =============================================================================
# OFFICIAL MODEL METRICS
# =============================================================================


test_r2 = safe_float(
    metadata_value(
        "test_r2"
    )
)

test_rmse = safe_float(
    metadata_value(
        "test_rmse"
    )
)

test_mae = safe_float(
    metadata_value(
        "test_mae"
    )
)

cv_score = safe_float(
    metadata_value(
        "cv_score"
    )
)


# =============================================================================
# SIDEBAR
# =============================================================================


with st.sidebar:

    st.title("💼 SmartPay")

    st.caption(
        "Employee Salary Intelligence"
    )

    st.divider()

    st.subheader("🧭 Navigation")

    page = st.radio(
        "Choose a module",
        [
            "🔮 Single Prediction",
            "📂 Batch Prediction",
            "📊 Model Evaluation",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("🤖 Active Model")

    st.info(
        model_name
    )

    st.subheader("📦 Model Status")

    model_files = {
        "Model": MODEL_FILE,
        "Encoders": ENCODER_FILE,
        "Scaler": SCALER_FILE,
        "Metadata": METADATA_FILE,
    }

    for name, path in model_files.items():

        if path.exists():

            st.success(
                f"✓ {name}"
            )

        else:

            st.error(
                f"✗ {name}"
            )

    st.divider()

    st.caption(
        "SmartPay v1.0"
    )

    st.caption(
        "Salary prediction and model evaluation"
    )


# =============================================================================
# MAIN HEADER
# =============================================================================


st.title(
    "💼 SmartPay Salary Intelligence"
)

st.write(
    "Machine-learning powered employee salary estimation "
    "with logical validation, batch prediction and model evidence."
)

st.info(
    f"🤖 Active model: **{model_name}**"
)


# =============================================================================
# GLOBAL MODEL PERFORMANCE
# =============================================================================


st.header("📈 Model Performance")

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "Test R²",
        (
            f"{test_r2:.4f}"
            if test_r2 is not None
            else "N/A"
        ),
    )

with m2:

    st.metric(
        "Test RMSE",
        money(test_rmse),
    )

with m3:

    st.metric(
        "Test MAE",
        money(test_mae),
    )

with m4:

    st.metric(
        "Cross Validation",
        (
            f"{cv_score:.4f}"
            if cv_score is not None
            else "N/A"
        ),
    )


st.divider()


# =============================================================================
# PAGE 1 — SINGLE PREDICTION
# =============================================================================


if page == "🔮 Single Prediction":

    st.header(
        "🔮 Individual Salary Prediction"
    )

    st.write(
        "Enter employee information. SmartPay checks "
        "the values against logical business rules before "
        "running the trained model."
    )

    left, right = st.columns(2)

    # -------------------------------------------------------------------------
    # Employee profile
    # -------------------------------------------------------------------------

    with left:

        st.subheader(
            "👤 Employee Profile"
        )

        age = st.slider(
            "Age",
            min_value=18,
            max_value=65,
            value=30,
            step=1,
        )

        max_experience = max(
            0,
            age - 18,
        )

        experience = st.slider(
            "Years of Experience",
            min_value=0,
            max_value=47,
            value=min(
                5,
                max_experience,
            ),
            step=1,
        )

        if experience <= max_experience:

            st.success(
                "✓ Age and experience relationship looks reasonable."
            )

        else:

            st.error(
                "⚠️ Experience cannot be greater than "
                f"approximately {max_experience} years for this age."
            )

    # -------------------------------------------------------------------------
    # Career
    # -------------------------------------------------------------------------

    with right:

        st.subheader(
            "🎓 Education & Career"
        )

        gender_options = sorted(
            [
                str(value)
                for value in
                label_encoders[
                    "Gender"
                ].classes_
            ]
        )

        education_options = sorted(
            [
                str(value)
                for value in
                label_encoders[
                    "Education"
                ].classes_
            ]
        )

        occupation_options = sorted(
            [
                str(value)
                for value in
                label_encoders[
                    "Occupation"
                ].classes_
            ]
        )

        gender = st.selectbox(
            "Gender",
            gender_options,
        )

        education = st.selectbox(
            "Education Level",
            education_options,
        )

        occupation = st.selectbox(
            "Occupation",
            occupation_options,
        )

    # -------------------------------------------------------------------------
    # Input summary
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader(
        "🔍 Input Verification"
    )

    input_summary = pd.DataFrame(
        {
            "Field": [
                "Age",
                "Experience",
                "Gender",
                "Education",
                "Occupation",
            ],
            "Value": [
                age,
                experience,
                gender,
                education,
                occupation,
            ],
        }
    )

    st.dataframe(
        input_summary,
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------

    if st.button(
        "🚀 Predict Salary",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Validating employee information..."
        ):

            prediction = None
            errors = []
            warnings = []

            try:

                (
                    prediction,
                    errors,
                    warnings,
                ) = predict_single(
                    age,
                    experience,
                    gender,
                    education,
                    occupation,
                )

            except Exception as error:

                errors.append(
                    str(error)
                )

        if errors:

            st.error(
                "❌ Prediction was blocked."
            )

            for error in errors:

                st.error(
                    error
                )

        else:

            for warning in warnings:

                st.warning(
                    warning
                )

            st.success(
                "✓ Prediction completed successfully."
            )

            st.subheader(
                "💰 Predicted Salary"
            )

            result_col1, result_col2 = st.columns(
                [2, 1]
            )

            with result_col1:

                st.metric(
                    "Estimated Annual Salary",
                    money(prediction),
                )

            with result_col2:

                if test_mae is not None:

                    st.metric(
                        "Typical Absolute Error",
                        money(test_mae),
                    )

                else:

                    st.metric(
                        "Typical Absolute Error",
                        "N/A",
                    )

            # -------------------------------------------------------------
            # Reference range
            # -------------------------------------------------------------

            if test_mae is not None:

                lower = max(
                    0,
                    prediction - test_mae,
                )

                upper = (
                    prediction
                    + test_mae
                )

                st.info(
                    f"Reference range using the model's MAE: "
                    f"**{money(lower)} – {money(upper)}**"
                )

                st.caption(
                    "This is an error reference based on MAE, "
                    "not a statistical confidence interval."
                )

            # -------------------------------------------------------------
            # Prediction details
            # -------------------------------------------------------------

            st.subheader(
                "📋 Prediction Summary"
            )

            summary = pd.DataFrame(
                {
                    "Employee Attribute": [
                        "Age",
                        "Experience",
                        "Gender",
                        "Education",
                        "Occupation",
                        "Predicted Salary",
                    ],
                    "Value": [
                        age,
                        experience,
                        gender,
                        education,
                        occupation,
                        money(prediction),
                    ],
                }
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# PAGE 2 — BATCH PREDICTION
# =============================================================================


elif page == "📂 Batch Prediction":

    st.header(
        "📂 Batch Salary Prediction"
    )

    st.write(
        "Upload a CSV file containing multiple employees. "
        "SmartPay validates every row before prediction."
    )

    # -------------------------------------------------------------------------
    # CSV format
    # -------------------------------------------------------------------------

    with st.expander(
        "📋 Required CSV Format",
        expanded=True,
    ):

        st.write(
            "Your CSV must contain these columns:"
        )

        st.code(
            ", ".join(RAW_FEATURES)
        )

        st.write(
            "Example:"
        )

        example_csv = (
            "Age,Experience,Gender,Education,Occupation\n"
            "30,5,Male,Bachelor's,Software Engineer\n"
            "42,15,Female,Master's,Manager\n"
            "27,3,Male,Bachelor's,Data Scientist"
        )

        st.code(
            example_csv,
            language="csv",
        )

    uploaded_file = st.file_uploader(
        "Upload employee CSV",
        type=["csv"],
        help="CSV must contain Age, Experience, Gender, Education and Occupation.",
    )

    if uploaded_file is not None:

        try:

            batch_data = pd.read_csv(
                uploaded_file
            )

        except Exception as error:

            st.error(
                f"❌ Could not read the CSV: {error}"
            )

            st.stop()

        # ---------------------------------------------------------------------
        # Dataset summary
        # ---------------------------------------------------------------------

        st.subheader(
            "📊 Uploaded Dataset"
        )

        s1, s2, s3 = st.columns(3)

        with s1:

            st.metric(
                "Rows",
                f"{len(batch_data):,}",
            )

        with s2:

            st.metric(
                "Columns",
                f"{len(batch_data.columns):,}",
            )

        with s3:

            missing_cells = int(
                batch_data.isna()
                .sum()
                .sum()
            )

            st.metric(
                "Missing Cells",
                f"{missing_cells:,}",
            )

        # ---------------------------------------------------------------------
        # Required columns
        # ---------------------------------------------------------------------

        missing_columns = [
            column
            for column in RAW_FEATURES
            if column not in batch_data.columns
        ]

        if missing_columns:

            st.error(
                "❌ Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

            st.stop()

        st.dataframe(
            batch_data.head(100),
            use_container_width=True,
            hide_index=True,
        )

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        st.subheader(
            "🔎 Row Validation"
        )

        valid_indices = []
        invalid_rows = []

        for index, row in batch_data.iterrows():

            row_errors = []
            row_warnings = []

            # -------------------------------------------------------------
            # Missing fields
            # -------------------------------------------------------------

            for column in RAW_FEATURES:

                if (
                    pd.isna(
                        row[column]
                    )
                    or
                    normalize_string(
                        row[column]
                    ) == ""
                ):

                    row_errors.append(
                        f"{column} is missing"
                    )

            if row_errors:

                invalid_rows.append(
                    {
                        "CSV Row": index + 2,
                        "Reason": "; ".join(
                            row_errors
                        ),
                    }
                )

                continue

            # -------------------------------------------------------------
            # Numeric values
            # -------------------------------------------------------------

            row_age = safe_float(
                row["Age"]
            )

            row_experience = safe_float(
                row["Experience"]
            )

            if row_age is None:

                row_errors.append(
                    "Age must be numeric"
                )

            if row_experience is None:

                row_errors.append(
                    "Experience must be numeric"
                )

            if row_errors:

                invalid_rows.append(
                    {
                        "CSV Row": index + 2,
                        "Reason": "; ".join(
                            row_errors
                        ),
                    }
                )

                continue

            # -------------------------------------------------------------
            # Logical rules
            # -------------------------------------------------------------

            (
                row_errors,
                row_warnings,
            ) = validate_employee(
                row_age,
                row_experience,
                row["Gender"],
                row["Education"],
                row["Occupation"],
            )

            if row_errors:

                invalid_rows.append(
                    {
                        "CSV Row": index + 2,
                        "Reason": "; ".join(
                            row_errors
                        ),
                    }
                )

            else:

                valid_indices.append(
                    index
                )

        # ---------------------------------------------------------------------
        # Validation statistics
        # ---------------------------------------------------------------------

        v1, v2, v3 = st.columns(3)

        with v1:

            st.metric(
                "Total Rows",
                f"{len(batch_data):,}",
            )

        with v2:

            st.metric(
                "Valid Rows",
                f"{len(valid_indices):,}",
            )

        with v3:

            st.metric(
                "Invalid Rows",
                f"{len(invalid_rows):,}",
            )

        if invalid_rows:

            st.warning(
                f"{len(invalid_rows):,} row(s) failed validation."
            )

            with st.expander(
                "⚠️ View Invalid Rows"
            ):

                invalid_df = pd.DataFrame(
                    invalid_rows
                )

                st.dataframe(
                    invalid_df,
                    use_container_width=True,
                    hide_index=True,
                )

        else:

            st.success(
                "✓ All uploaded rows passed validation."
            )

        # ---------------------------------------------------------------------
        # Run prediction
        # ---------------------------------------------------------------------

        if valid_indices:

            st.divider()

            if st.button(
                "🚀 Run Batch Prediction",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    f"Predicting {len(valid_indices):,} employees..."
                ):

                    try:

                        valid_data = batch_data.loc[
                            valid_indices,
                            RAW_FEATURES,
                        ].copy()

                        predictions = predict_dataframe(
                            valid_data
                        )

                        result_data = batch_data.copy()

                        result_data[
                            "Predicted_Salary"
                        ] = np.nan

                        result_data.loc[
                            valid_indices,
                            "Predicted_Salary",
                        ] = predictions

                        result_data[
                            "Prediction_Status"
                        ] = "Invalid"

                        result_data.loc[
                            valid_indices,
                            "Prediction_Status",
                        ] = "Predicted"

                        # Store invalid reason
                        result_data[
                            "Validation_Status"
                        ] = "Invalid"

                        result_data.loc[
                            valid_indices,
                            "Validation_Status",
                        ] = "Valid"

                        st.session_state[
                            "batch_result"
                        ] = result_data

                        st.success(
                            "✓ Batch prediction completed successfully."
                        )

                    except Exception as error:

                        st.error(
                            "❌ Batch prediction failed."
                        )

                        st.exception(
                            error
                        )

        # ---------------------------------------------------------------------
        # Display results
        # ---------------------------------------------------------------------

        if (
            "batch_result"
            in st.session_state
        ):

            result_data = st.session_state[
                "batch_result"
            ]

            st.divider()

            st.subheader(
                "💰 Batch Prediction Results"
            )

            predicted_values = (
                pd.to_numeric(
                    result_data[
                        "Predicted_Salary"
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            if len(predicted_values) > 0:

                r1, r2, r3, r4 = st.columns(4)

                with r1:

                    st.metric(
                        "Predictions",
                        f"{len(predicted_values):,}",
                    )

                with r2:

                    st.metric(
                        "Average Salary",
                        money(
                            predicted_values.mean()
                        ),
                    )

                with r3:

                    st.metric(
                        "Minimum",
                        money(
                            predicted_values.min()
                        ),
                    )

                with r4:

                    st.metric(
                        "Maximum",
                        money(
                            predicted_values.max()
                        ),
                    )

                # ---------------------------------------------------------
                # Distribution
                # ---------------------------------------------------------

                st.subheader(
                    "📊 Predicted Salary Distribution"
                )

                chart_data = pd.DataFrame(
                    {
                        "Predicted Salary":
                            predicted_values.values
                    }
                )

                st.bar_chart(
                    chart_data,
                    use_container_width=True,
                )

            # -----------------------------------------------------------------
            # Result table
            # -----------------------------------------------------------------

            st.subheader(
                "📋 Detailed Results"
            )

            st.dataframe(
                result_data,
                use_container_width=True,
                hide_index=True,
            )

            # -----------------------------------------------------------------
            # Download
            # -----------------------------------------------------------------

            csv_output = result_data.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            st.download_button(
                "📥 Download Prediction Results",
                data=csv_output,
                file_name="smartpay_salary_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# =============================================================================
# PAGE 3 — MODEL EVALUATION
# =============================================================================


elif page == "📊 Model Evaluation":

    st.header(
        "📊 Model Evaluation & Evidence"
    )

    st.write(
        "This section presents the official model metrics, "
        "dataset-based verification and every available "
        "visual artifact produced during model development."
    )

    # =========================================================================
    # OFFICIAL RESULTS
    # =========================================================================

    st.subheader(
        "🏆 Official Model Results"
    )

    st.info(
        "These values come directly from "
        "`model_metadata.pkl` and represent the model's stored evaluation results."
    )

    o1, o2, o3, o4 = st.columns(4)

    with o1:

        st.metric(
            "Test R²",
            (
                f"{test_r2:.4f}"
                if test_r2 is not None
                else "N/A"
            ),
        )

    with o2:

        st.metric(
            "Test RMSE",
            money(test_rmse),
        )

    with o3:

        st.metric(
            "Test MAE",
            money(test_mae),
        )

    with o4:

        st.metric(
            "CV Score",
            (
                f"{cv_score:.4f}"
                if cv_score is not None
                else "N/A"
            ),
        )

    # =========================================================================
    # MODEL VERDICT
    # =========================================================================

    st.subheader(
        "🧠 Model Verdict"
    )

    if test_r2 is None:

        st.warning(
            "Test R² is not available."
        )

    elif test_r2 >= 0.90:

        st.success(
            f"🟢 EXCELLENT — Test R² = {test_r2:.4f}. "
            "The model demonstrates very strong explanatory performance."
        )

    elif test_r2 >= 0.80:

        st.success(
            f"🔵 STRONG — Test R² = {test_r2:.4f}. "
            "The model demonstrates strong predictive performance."
        )

    elif test_r2 >= 0.70:

        st.warning(
            f"🟡 MODERATE — Test R² = {test_r2:.4f}. "
            "The model has useful predictive performance but should be interpreted carefully."
        )

    else:

        st.error(
            f"🔴 LIMITED — Test R² = {test_r2:.4f}. "
            "The model may require further improvement."
        )

    if test_mae is not None:

        st.info(
            f"Typical absolute error according to the stored test MAE: "
            f"**{money(test_mae)}**."
        )

    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================

    with st.expander(
        "🔧 Model Configuration"
    ):

        config_data = {
            "Property": [
                "Model",
                "Scaling Required",
                "Test R²",
                "Test RMSE",
                "Test MAE",
                "Cross Validation",
            ],
            "Value": [
                model_name,
                str(
                    metadata_value(
                        "needs_scaling",
                        False,
                    )
                ),
                (
                    f"{test_r2:.6f}"
                    if test_r2 is not None
                    else "N/A"
                ),
                money(test_rmse),
                money(test_mae),
                (
                    f"{cv_score:.6f}"
                    if cv_score is not None
                    else "N/A"
                ),
            ],
        }

        st.dataframe(
            pd.DataFrame(config_data),
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================================
    # DATASET VERIFICATION
    # =========================================================================

    st.divider()

    st.subheader(
        "🔬 Dataset-Based Verification"
    )

    dataset, dataset_name = (
        load_available_dataset()
    )

    if dataset is None:

        st.warning(
            "Neither `cleaned_data.csv` nor `DATASET.csv` "
            "was found in the application directory."
        )

    else:

        st.success(
            f"Dataset loaded: **{dataset_name}** "
            f"with **{len(dataset):,} rows**."
        )

        target_column = find_target_column(
            dataset
        )

        if target_column is None:

            st.warning(
                "No salary target column could be detected."
            )

            st.write(
                "Available columns:"
            )

            st.write(
                list(
                    dataset.columns
                )
            )

        else:

            st.info(
                f"Detected target column: **{target_column}**"
            )

            missing_dataset_columns = [
                column
                for column in RAW_FEATURES
                if column not in dataset.columns
            ]

            if missing_dataset_columns:

                st.warning(
                    "The dataset is missing the following "
                    "prediction features: "
                    + ", ".join(
                        missing_dataset_columns
                    )
                )

            else:

                evaluation_data = dataset[
                    RAW_FEATURES + [target_column]
                ].copy()

                # -------------------------------------------------------------
                # Numeric target
                # -------------------------------------------------------------

                evaluation_data[
                    target_column
                ] = pd.to_numeric(
                    evaluation_data[
                        target_column
                    ],
                    errors="coerce",
                )

                # -------------------------------------------------------------
                # Drop missing rows
                # -------------------------------------------------------------

                evaluation_data = (
                    evaluation_data
                    .dropna(
                        subset=RAW_FEATURES
                        + [target_column]
                    )
                    .copy()
                )

                st.write(
                    f"Rows available for evaluation: "
                    f"**{len(evaluation_data):,}**"
                )

                if len(evaluation_data) == 0:

                    st.error(
                        "No complete rows are available for dataset evaluation."
                    )

                else:

                    # ---------------------------------------------------------
                    # Validate dataset rows
                    # ---------------------------------------------------------

                    dataset_valid_indices = []
                    dataset_invalid_count = 0

                    for index, row in evaluation_data.iterrows():

                        row_age = safe_float(
                            row["Age"]
                        )

                        row_experience = safe_float(
                            row["Experience"]
                        )

                        if (
                            row_age is None
                            or row_experience is None
                        ):

                            dataset_invalid_count += 1
                            continue

                        row_errors, _ = validate_employee(
                            row_age,
                            row_experience,
                            row["Gender"],
                            row["Education"],
                            row["Occupation"],
                        )

                        if row_errors:

                            dataset_invalid_count += 1

                        else:

                            dataset_valid_indices.append(
                                index
                            )

                    if dataset_invalid_count:

                        st.warning(
                            f"{dataset_invalid_count:,} dataset row(s) "
                            "failed logical validation and were excluded."
                        )

                    if dataset_valid_indices:

                        clean_evaluation = (
                            evaluation_data.loc[
                                dataset_valid_indices
                            ].copy()
                        )

                        try:

                            # -------------------------------------------------
                            # Generate predictions
                            # -------------------------------------------------

                            predictions = predict_dataframe(
                                clean_evaluation[
                                    RAW_FEATURES
                                ]
                            )

                            actual_values = (
                                clean_evaluation[
                                    target_column
                                ]
                                .astype(float)
                                .to_numpy()
                            )

                            metrics = calculate_metrics(
                                actual_values,
                                predictions,
                            )

                            # -------------------------------------------------
                            # Display independent calculation
                            # -------------------------------------------------

                            st.subheader(
                                "📊 Calculated Dataset Metrics"
                            )

                            d1, d2, d3, d4 = st.columns(4)

                            with d1:

                                st.metric(
                                    "R²",
                                    (
                                        f"{metrics['R²']:.4f}"
                                        if np.isfinite(
                                            metrics["R²"]
                                        )
                                        else "N/A"
                                    ),
                                )

                            with d2:

                                st.metric(
                                    "RMSE",
                                    money(
                                        metrics["RMSE"]
                                    ),
                                )

                            with d3:

                                st.metric(
                                    "MAE",
                                    money(
                                        metrics["MAE"]
                                    ),
                                )

                            with d4:

                                st.metric(
                                    "Rows",
                                    f"{metrics['Samples']:,}",
                                )

                            # -------------------------------------------------
                            # Important interpretation
                            # -------------------------------------------------

                            st.warning(
                                "⚠️ Dataset verification is not automatically "
                                "independent test evidence. If this CSV was "
                                "used during training, preprocessing or model "
                                "selection, its metrics can be optimistic. "
                                "The official Test R² / RMSE / MAE above should "
                                "be used as the primary generalization evidence."
                            )

                            # -------------------------------------------------
                            # Actual vs predicted
                            # -------------------------------------------------

                            verification = pd.DataFrame(
                                {
                                    "Actual Salary":
                                        actual_values,

                                    "Predicted Salary":
                                        predictions,

                                    "Absolute Error":
                                        np.abs(
                                            actual_values
                                            - predictions
                                        ),

                                    "Percentage Error":
                                        np.where(
                                            actual_values != 0,
                                            (
                                                np.abs(
                                                    actual_values
                                                    - predictions
                                                )
                                                /
                                                np.abs(
                                                    actual_values
                                                )
                                                * 100
                                            ),
                                            np.nan,
                                        ),
                                }
                            )

                            st.subheader(
                                "🎯 Actual vs Predicted Verification"
                            )

                            st.dataframe(
                                verification.head(100),
                                use_container_width=True,
                                hide_index=True,
                            )

                            # -------------------------------------------------
                            # Error statistics
                            # -------------------------------------------------

                            st.subheader(
                                "📉 Error Analysis"
                            )

                            abs_errors = (
                                verification[
                                    "Absolute Error"
                                ]
                            )

                            e1, e2, e3 = st.columns(3)

                            with e1:

                                st.metric(
                                    "Average Absolute Error",
                                    money(
                                        abs_errors.mean()
                                    ),
                                )

                            with e2:

                                st.metric(
                                    "Median Absolute Error",
                                    money(
                                        abs_errors.median()
                                    ),
                                )

                            with e3:

                                st.metric(
                                    "Maximum Absolute Error",
                                    money(
                                        abs_errors.max()
                                    ),
                                )

                            # -------------------------------------------------
                            # Error chart
                            # -------------------------------------------------

                            error_chart = pd.DataFrame(
                                {
                                    "Absolute Error":
                                        abs_errors
                                        .head(100)
                                        .values
                                }
                            )

                            st.bar_chart(
                                error_chart,
                                use_container_width=True,
                            )

                            # -------------------------------------------------
                            # Download verification
                            # -------------------------------------------------

                            verification_csv = (
                                verification
                                .to_csv(
                                    index=False
                                )
                                .encode(
                                    "utf-8"
                                )
                            )

                            st.download_button(
                                "📥 Download Dataset Verification",
                                verification_csv,
                                "smartpay_model_verification.csv",
                                "text/csv",
                                use_container_width=True,
                            )

                        except Exception as error:

                            st.error(
                                "❌ Dataset evaluation failed."
                            )

                            st.exception(
                                error
                            )

                    else:

                        st.error(
                            "No valid dataset rows were available "
                            "for model evaluation."
                        )

    # =========================================================================
    # VISUAL EVIDENCE
    # =========================================================================

    st.divider()

    st.subheader(
        "🖼️ Complete Visual Evidence"
    )

    st.write(
        "All available evaluation images generated by the project "
        "are displayed below. The images themselves are not color-inverted, "
        "so the charts remain scientifically accurate."
    )

    available_images = []
    missing_images = []

    for title, filename in EVALUATION_IMAGES.items():

        image_path = (
            BASE_DIR
            / filename
        )

        if image_path.exists():

            available_images.append(
                (
                    title,
                    filename,
                    image_path,
                )
            )

        else:

            missing_images.append(
                filename
            )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.metric(
            "Evidence Found",
            f"{len(available_images)}/{len(EVALUATION_IMAGES)}",
        )

    with a2:

        st.metric(
            "Missing",
            str(
                len(missing_images)
            ),
        )

    with a3:

        st.metric(
            "Total Artifacts",
            str(
                len(EVALUATION_IMAGES)
            ),
        )

    if missing_images:

        with st.expander(
            "⚠️ Missing Evaluation Images"
        ):

            for filename in missing_images:

                st.write(
                    f"• `{filename}`"
                )

    # =========================================================================
    # EVIDENCE TABS
    # =========================================================================

    evidence_tabs = st.tabs(
        [
            "🎯 Accuracy",
            "📉 Errors",
            "🧠 Features",
            "⚖️ Comparisons",
            "🔗 Correlation",
            "📚 All",
        ]
    )

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------

    def display_evidence_image(
        title,
        filename,
        path,
    ):

        st.subheader(
            title
        )

        st.caption(
            f"Source: `{filename}`"
        )

        st.image(
            str(path),
            use_container_width=True,
        )

        st.divider()

    # -------------------------------------------------------------------------
    # Accuracy
    # -------------------------------------------------------------------------

    with evidence_tabs[0]:

        accuracy_titles = [
            "Best Model — Actual vs Predicted",
            "R² / RMSE Scores",
            "Model R² Comparison",
        ]

        found = False

        for title in accuracy_titles:

            item = next(
                (
                    x
                    for x in available_images
                    if x[0] == title
                ),
                None,
            )

            if item:

                found = True

                display_evidence_image(
                    item[0],
                    item[1],
                    item[2],
                )

        if not found:

            st.info(
                "No accuracy images were found."
            )

    # -------------------------------------------------------------------------
    # Errors
    # -------------------------------------------------------------------------

    with evidence_tabs[1]:

        error_titles = [
            "Best Model — Residuals",
            "Residuals Plot",
            "Model Error Comparison",
        ]

        found = False

        for title in error_titles:

            item = next(
                (
                    x
                    for x in available_images
                    if x[0] == title
                ),
                None,
            )

            if item:

                found = True

                display_evidence_image(
                    item[0],
                    item[1],
                    item[2],
                )

        if not found:

            st.info(
                "No error-analysis images were found."
            )

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------

    with evidence_tabs[2]:

        feature_titles = [
            "Best Model — Feature Importance",
            "Feature Importance",
        ]

        found = False

        for title in feature_titles:

            item = next(
                (
                    x
                    for x in available_images
                    if x[0] == title
                ),
                None,
            )

            if item:

                found = True

                display_evidence_image(
                    item[0],
                    item[1],
                    item[2],
                )

        if not found:

            st.info(
                "No feature-importance images were found."
            )

    # -------------------------------------------------------------------------
    # Comparisons
    # -------------------------------------------------------------------------

    with evidence_tabs[3]:

        comparison_titles = [
            "Model Comprehensive Comparison",
            "Model CV Comparison",
            "Model Error Comparison",
            "Model R² Comparison",
        ]

        found = False

        for title in comparison_titles:

            item = next(
                (
                    x
                    for x in available_images
                    if x[0] == title
                ),
                None,
            )

            if item:

                found = True

                display_evidence_image(
                    item[0],
                    item[1],
                    item[2],
                )

        if not found:

            st.info(
                "No model-comparison images were found."
            )

    # -------------------------------------------------------------------------
    # Correlation
    # -------------------------------------------------------------------------

    with evidence_tabs[4]:

        item = next(
            (
                x
                for x in available_images
                if x[0] == "Correlation Heatmap"
            ),
            None,
        )

        if item:

            display_evidence_image(
                item[0],
                item[1],
                item[2],
            )

        else:

            st.info(
                "Correlation heatmap was not found."
            )

    # -------------------------------------------------------------------------
    # All evidence
    # -------------------------------------------------------------------------

    with evidence_tabs[5]:

        if not available_images:

            st.info(
                "No evaluation images were found."
            )

        else:

            for (
                title,
                filename,
                path,
            ) in available_images:

                with st.expander(
                    f"📊 {title}"
                ):

                    st.caption(
                        f"Source: `{filename}`"
                    )

                    st.image(
                        str(path),
                        use_container_width=True,
                    )


# =============================================================================
# FOOTER
# =============================================================================


st.divider()

st.caption(
    "💼 SmartPay Salary Intelligence • "
    "Machine-learning based salary estimation system"
)

st.caption(
    "Predictions should support, not replace, professional "
    "compensation decisions."
)