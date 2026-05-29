from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from churn_engine import construir_churn, preparar_features


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "benchmark_shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_FOLDS = 5
SCORING = ["roc_auc", "precision", "recall", "f1"]


MODELOS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

# Preparar dataset Telco
def preparar_telco(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    df["Churn"] = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    y = df.pop("Churn")

    X = pd.get_dummies(
        df,
        drop_first=True,
    )

    num_cols = X.select_dtypes(include=["number"]).columns

    scaler = StandardScaler()

    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y

# Evaluación con validación cruzada
def evaluar_dataset(
    nombre_dataset: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:

    cv = StratifiedKFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows: list[dict] = []

    for nombre_modelo, modelo in MODELOS.items():

        print(f"Evaluando {nombre_modelo}...")

        scores = cross_validate(
            modelo,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
        )

        rows.append(
            {
                "Dataset": nombre_dataset,
                "Modelo": nombre_modelo,
                "ROC-AUC": scores["test_roc_auc"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "F1": scores["test_f1"].mean(),
            }
        )

    return pd.DataFrame(rows)

def benchmark_telco() -> pd.DataFrame:

    telco_path = ROOT / "TelcoCustomerChurn.csv"

    if not telco_path.exists():
        raise FileNotFoundError(
            "No se encontro TelcoCustomerChurn.csv en la raiz del proyecto."
        )

    X_telco, y_telco = preparar_telco(telco_path)

    telco_res = evaluar_dataset(
        "Telco",
        X_telco,
        y_telco,
    )

    telco_res.to_csv(
        OUTPUT_DIR / "telco_metrics.csv",
        index=False,
    )

    # Grafica comparativa de métricas

    fig, ax = plt.subplots(figsize=(10, 4.5))

    metricas = [
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1",
    ]

    x = np.arange(len(metricas))
    width = 0.22

    colores = [
        "#B4B2A9",
        "#5DCAA5",
        "#185FA5",
    ]

    for i, (_, row) in enumerate(telco_res.iterrows()):

        vals = [row[m] for m in metricas]

        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=row["Modelo"],
            color=colores[i],
            alpha=0.9,
        )

        for b, v in zip(bars, vals):

            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.004,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x + width)

    ax.set_xticklabels(metricas)

    ax.set_ylim(0, 1.12)

    ax.set_ylabel("Score")

    ax.set_title(
        "Benchmark Telco - Validacion cruzada (k=5)"
    )

    ax.legend(
        loc="upper right",
        fontsize=9,
    )

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "telco_model_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print("Grafica benchmark guardada")

    return telco_res

# Interpretabilidad SHAP
def shap_shopping_best_model() -> None:

    try:
        import shap

    except Exception as exc:

        print(f"[WARN] SHAP no disponible: {exc}")

        return

    model_artifact = ROOT / "best_model.pkl"

    shopping_path = ROOT / "shopping_trends.csv"

    if not model_artifact.exists():

        raise FileNotFoundError(
            "No se encontro best_model.pkl. "
            "Ejecuta primero: python model_training.py"
        )

    if not shopping_path.exists():

        raise FileNotFoundError(
            "No se encontro shopping_trends.csv."
        )

    with open(model_artifact, "rb") as f:

        payload = pickle.load(f)

    model = payload["modelo"]

    scaler = payload.get("scaler")

    feature_columns = payload.get(
        "feature_columns",
        [],
    )

    df_raw = pd.read_csv(shopping_path)

    df = construir_churn(
        df_raw,
        verbose=False,
    )

    X, y, local_scaler = preparar_features(df)

    if scaler is not None:

        X, y, _ = preparar_features(
            df,
            scaler=scaler,
            fit_scaler=False,
        )

    else:

        scaler = local_scaler

    if feature_columns:

        X = X.reindex(
            columns=feature_columns,
            fill_value=0,
        )

    model.fit(X, y)

    preds = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, preds)

    print(
        f"[OK] Mejor modelo listo para SHAP. "
        f"AUC train={auc:.4f}"
    )

    # Muestra aleatoria para SHAP

    sample_n = min(400, len(X))

    X_sample = X.sample(
        sample_n,
        random_state=RANDOM_STATE,
    )

    # Crear explainer SHAP

    explainer = shap.Explainer(
        model,
        X_sample,
    )

    shap_values = explainer(X_sample)

    # Summary plot

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=12,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # Waterfall plot del primer cliente

    plt.figure()

    shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_waterfall_cliente_1.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print("Graficas SHAP guardadas")

# Pipeline principal
def main() -> None:

    print("=" * 60)
    print(" BENCHMARK TELCO + SHAP ")
    print("=" * 60)

    telco_df = benchmark_telco()

    print("\nMetricas promedio en Telco:")

    print(
        telco_df
        .round(4)
        .to_string(index=False)
    )

    shap_shopping_best_model()

    print("\nArchivos generados:")

    print("  · telco_metrics.csv")
    print("  · telco_model_comparison.png")
    print("  · shap_summary.png")
    print("  · shap_waterfall_cliente_1.png")

    print(f"\nSalida: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()