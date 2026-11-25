from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_param_grid(model_name: str) -> dict:
    grids = {
        "Logistic Regression": {
            "model__C": np.logspace(-3, 2, 20),
            "model__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "model__n_estimators": [150, 250, 400],
            "model__max_depth": [4, 6, 8, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__class_weight": [None, "balanced"],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 150, 250],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4, 5],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [3, 4, 6, 8],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        },
    }
    return grids[model_name]


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "customer_churn.csv"
    models_dir = project_root / "models"
    assets_dir = project_root / "assets"
    models_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare data
    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    drop_cols = [c for c in ["Churn", "Churn_Num", "customerID"] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_features = X_train.select_dtypes(include="number").columns
    categorical_features = X_train.select_dtypes(exclude="number").columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    # 2. Compare candidate models with 5-fold StratifiedKFold CV
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42)
    except ImportError:
        pass

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    summary_rows = []

    for name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={"f1": "f1", "roc_auc": "roc_auc"},
            n_jobs=-1,
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        summary_rows.append(
            {
                "Model": name,
                "F1 Mean": cv_scores["test_f1"].mean(),
                "F1 Std": cv_scores["test_f1"].std(),
                "ROC-AUC Mean": cv_scores["test_roc_auc"].mean(),
                "ROC-AUC Std": cv_scores["test_roc_auc"].std(),
                "Test Churn Recall": recall_score(y_test, y_pred, pos_label=1),
            }
        )

    results_df = pd.DataFrame(summary_rows).sort_values(by="ROC-AUC Mean", ascending=False)
    print("\n=== 5-Fold Cross-Validation Comparison ===")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 3. Tune best model
    best_name = results_df.iloc[0]["Model"]
    best_estimator = models[best_name]
    search_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_estimator)])

    search = RandomizedSearchCV(
        estimator=search_pipe,
        param_distributions=get_param_grid(best_name),
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    final_model = search.best_estimator_

    # 4. Evaluate tuned model on test set
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    print(f"\n=== Tuned {best_name} Test Evaluation ===")
    print(f"Best Params: {search.best_params_}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f"Test F1 (Churn=1): {f1_score(y_test, y_test_pred, pos_label=1):.4f}")
    print(f"Test Churn Recall (Class 1): {recall_score(y_test, y_test_pred, pos_label=1):.4f}\n")
    print(classification_report(y_test, y_test_pred, target_names=["Stayed (0)", "Churned (1)"]))

    # Save one chart for README
    fig, ax = plt.subplots(figsize=(6, 4.5))
    RocCurveDisplay.from_estimator(final_model, X_test, y_test, name=f"Tuned {best_name}", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random Baseline (AUC = 0.50)")
    ax.set_title(f"ROC Curve — Tuned {best_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    chart_path = assets_dir / "roc_curve.png"
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)

    # 5. Save pipeline artifact with joblib
    model_path = models_dir / "churn_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved ROC chart to: {chart_path}")


if __name__ == "__main__":
    main()