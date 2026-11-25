import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from src.predict import load_model, predict_churn_proba


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"


def _sample_customer() -> dict:
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.5,
        "TotalCharges": 377.5,
    }


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run src/train.py first to generate churn_model.joblib")
def test_model_loads():
    model = load_model(MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict_proba")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run src/train.py first to generate churn_model.joblib")
def test_prediction_probability_between_zero_and_one():
    model = load_model(MODEL_PATH)
    result = predict_churn_proba(_sample_customer(), model=model)
    prob = result["churn_probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run src/train.py first to generate churn_model.joblib")
def test_missing_column_raises_error():
    model = load_model(MODEL_PATH)
    incomplete_customer = _sample_customer()
    del incomplete_customer["Contract"]
    with pytest.raises((ValueError, KeyError)):
        predict_churn_proba(incomplete_customer, model=model)