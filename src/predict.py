from pathlib import Path
from typing import Any
import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"
DEFAULT_THRESHOLD = 0.30


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
    return joblib.load(model_path)


def _get_expected_columns(pipeline) -> list[str] | None:
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    preprocessor = pipeline.named_steps.get("preprocessor") if hasattr(pipeline, "named_steps") else None
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    return None


def predict_churn_proba(
    customer_data: dict[str, Any],
    model=None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    if not isinstance(customer_data, dict) or not customer_data:
        raise ValueError("customer_data must be a non-empty dictionary.")

    model = model or load_model()
    expected_cols = _get_expected_columns(model)

    if expected_cols is not None:
        missing_cols = [col for col in expected_cols if col not in customer_data]
        if missing_cols:
            raise ValueError(f"Missing required customer feature(s): {missing_cols}")
        row = {col: customer_data[col] for col in expected_cols}
    else:
        row = dict(customer_data)

    df = pd.DataFrame([row])

    # Coerce TotalCharges if provided as string/number, matching training prep
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    proba = float(model.predict_proba(df)[0, 1])
    return {
        "churn_probability": proba,
        "churn_prediction": int(proba >= threshold),
        "threshold": threshold,
    }


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 179.7,
    }
    result = predict_churn_proba(sample_customer)
    print(result)