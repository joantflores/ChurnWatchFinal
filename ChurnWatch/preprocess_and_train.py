"""Preprocesamiento y entrenamiento de modelos de churn.

Entrena Logistic Regression, Random Forest y XGBoost sobre
`shopping_trends_churn.csv` y guarda el mejor pipeline completo en
`best_model.pkl`.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    import xgboost as xgb
except Exception:
    xgb = None

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "train"
OUT.mkdir(parents=True, exist_ok=True)

SHOP = ROOT / "shopping_trends_churn.csv"

RANDOM_STATE = 42


def canonical_name(column: str) -> str:
    return (
        column.lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding_errors="ignore")
    df.columns = [canonical_name(c) for c in df.columns]
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()
    if "churn" not in df.columns:
        raise ValueError("El dataset no tiene columna 'churn'. Ejecuta build_churn.py primero.")

    # target
    y = df["churn"].astype(int)

    if y.nunique() < 2:
        counts = y.value_counts().to_dict()
        raise ValueError(
            "La variable churn tiene una sola clase. "
            f"Distribucion encontrada: {counts}. Ejecuta build_churn.py para regenerarla."
        )

    # candidate features
    num_cols = [c for c in ["previous_purchases", "purchase_amount_usd"] if c in df.columns]
    drop_cols = {"churn", "customer_id"}
    cat_cols = [c for c in df.columns if c not in set(num_cols) | drop_cols]

    # simple imputation
    X_num = df[num_cols].fillna(0)
    X_cat = df[cat_cols].fillna("missing").astype(str)

    # column transformer
    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], remainder="drop")

    X = pd.concat([X_num, X_cat], axis=1)
    return X, y, preproc, num_cols, cat_cols


def fit_and_eval(X, y, preproc, num_cols, cat_cols):
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # If training set has only one class, use a DummyClassifier fallback
    if y_train.nunique() < 2:
        print("[WARN] Solo una clase presente en los datos de entrenamiento. Usando DummyClassifier como fallback.")
        dummy_pipe = Pipeline([
            ("pre", preproc),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ])
        dummy_pipe.fit(X_train, y_train)
        preds = dummy_pipe.predict(X_test)
        try:
            probs = dummy_pipe.predict_proba(X_test)[:, 1]
        except Exception:
            probs = None

        results = {
            "ConstantBaseline": {
                "roc_auc": None if probs is None else float(roc_auc_score(y_test, probs)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                "f1": float(f1_score(y_test, preds, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
            }
        }

        payload = {
            "modelo": dummy_pipe,
            "scaler": None,
            "feature_columns": num_cols,
            "nombre": "ConstantBaseline",
        }
        joblib.dump(payload, ROOT / "best_model.pkl")
        (OUT / "evaluation.json").write_text(json.dumps(results, indent=2))
        print(f"Fallback guardado como ConstantBaseline en {OUT}")
        return

    # build pipelines
    def make_pipeline(model):
        return Pipeline([
            ("pre", preproc),
            ("clf", model),
        ])

    models = {}
    models["LogisticRegression"] = make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    )
    models["RandomForest"] = make_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    if xgb is not None:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = neg / pos if pos else 1.0
        models["XGBoost"] = make_pipeline(
            xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        )

    results = {}

    for name, pipe in models.items():
        print(f"Entrenando {name}...")
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = pipe.predict(X_test)
        results[name] = {
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
        }

    # Mejor modelo: ROC-AUC, desempate por F1 (favorece XGBoost si empata con RF)
    def _rank(name: str) -> tuple:
        r = results[name]
        return (r["roc_auc"], r["f1"], r["recall"])

    best_name = max(results.keys(), key=_rank)
    best_model = models[best_name]

    # save model and meta
    # need fitted preproc/scaler; extract feature columns by transforming a sample
    fitted_preproc = best_model.named_steps["pre"]
    # create a transformer to get feature names
    num_features = num_cols
    # get cat feature names
    cat_encoder = fitted_preproc.named_transformers_["cat"]
    try:
        cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    except Exception:
        cat_names = []
    feature_columns = list(num_features) + cat_names

    from churn_engine import BAJA_FRECUENCIA_CANON, CHURN_UMBRAL_ENTRENAMIENTO

    payload = {
        "modelo": best_model,
        "scaler": None,
        "feature_columns": feature_columns,
        "input_columns": list(num_cols) + list(cat_cols),
        "numeric_features": list(num_cols),
        "categorical_features": list(cat_cols),
        "nombre": best_name,
        "churn_config": {
            "umbral_previous_purchases": CHURN_UMBRAL_ENTRENAMIENTO,
            "baja_frecuencia": sorted(BAJA_FRECUENCIA_CANON),
        },
    }

    joblib.dump(payload, ROOT / "best_model.pkl")
    (OUT / "evaluation.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame.from_dict(results, orient="index").drop(columns=["confusion_matrix"]).to_csv(
        OUT / "model_comparison.csv"
    )
    target_summary = {
        "rows": int(len(y)),
        "churn_0": int((y == 0).sum()),
        "churn_1": int((y == 1).sum()),
        "churn_rate": float(y.mean()),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "best_model": best_name,
    }
    (OUT / "target_summary.json").write_text(json.dumps(target_summary, indent=2))
    print(f"Modelos entrenados. Mejor: {best_name}. Resultados guardados en {OUT}")


def main():
    if not SHOP.exists():
        print("Ejecuta build_churn.py primero para generar shopping_trends_churn.csv")
        return
    df = load_data(SHOP)
    X, y, preproc, num_cols, cat_cols = preprocess(df)
    fit_and_eval(X, y, preproc, num_cols, cat_cols)


if __name__ == "__main__":
    main()
