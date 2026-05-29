"""
ChurnWatch — Entrenamiento de Modelos
======================================
Entrena y evalúa los tres modelos del proyecto:
  · Logistic Regression  (baseline)
  · Random Forest
  · XGBoost

Incluye:
  · Validación cruzada estratificada (k=5)
  · Métricas: ROC-AUC, Precision, Recall, F1
  · Matriz de confusión
  · Comparativa final de modelos
  · Guardado del mejor modelo en disco

Uso:
    python model_training.py
    → Genera: best_model.pkl  y  scaler.pkl
"""

from __future__ import annotations

import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier

# funciones del motor churn
from churn_engine import construir_churn, preparar_features

warnings.filterwarnings("ignore")

# configuracion general del entrenamiento

CSV_PATH = "shopping_trends.csv"
OUTPUT_DIR = Path(".")
K_FOLDS = 5
RANDOM_STATE = 42
SCORING = ["roc_auc", "precision", "recall", "f1"]

MODELOS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# carga del dataset y preparacion de features

def cargar_datos(csv_path: str = CSV_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el CSV, construye Churn y aplica feature engineering."""

    print(f"\n{'=' * 55}")
    print("  CARGANDO DATOS")
    print(f"{'=' * 55}")

    df_raw = pd.read_csv(csv_path)
    df = construir_churn(df_raw, verbose=True)

    X, y, scaler = preparar_features(df)

    print(f"\n  Features tras encoding : {X.shape[1]} columnas")
    print(f"  Churn = 0 (activos)    : {(y == 0).sum():,}")
    print(f"  Churn = 1 (en riesgo)  : {(y == 1).sum():,}")
    print(
        f"  Ratio desbalance       : "
        f"1 : {(y == 0).sum() / max((y == 1).sum(), 1):.1f}\n"
    )

    # guardar scaler para usarlo en la app
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("  ✅  scaler.pkl guardado")

    return X, y, scaler


# validacion cruzada de modelos

def evaluar_con_cv(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Corre validación cruzada estratificada (k=5) para cada modelo
    y devuelve un DataFrame con las métricas promedio.
    """

    print(f"\n{'=' * 55}")
    print(f"  VALIDACIÓN CRUZADA  (k = {K_FOLDS})")
    print(f"{'=' * 55}")

    cv = StratifiedKFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    resultados = []

    for nombre, modelo in MODELOS.items():

        print(f"\n  ▶  {nombre} …", end=" ", flush=True)

        scores = cross_validate(
            modelo,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
        )

        fila = {
            "Modelo": nombre,
            "ROC-AUC": scores["test_roc_auc"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
        }

        resultados.append(fila)

        print(
            f"AUC={fila['ROC-AUC']:.3f}  "
            f"Prec={fila['Precision']:.3f}  "
            f"Recall={fila['Recall']:.3f}  "
            f"F1={fila['F1']:.3f}"
        )

    df_res = pd.DataFrame(resultados).set_index("Modelo")

    print(f"\n{'=' * 55}")
    print("  COMPARATIVA FINAL")
    print(f"{'=' * 55}")

    print(df_res.round(4).to_string())

    return df_res


# entrenamiento final y metricas detalladas

def entrenar_y_evaluar(
    X: pd.DataFrame,
    y: pd.Series,
    df_cv: pd.DataFrame,
) -> tuple[object, str]:
    """
    Entrena cada modelo con TODO el dataset, calcula métricas en conjunto
    completo, grafica matrices de confusión y curvas ROC, y devuelve
    el mejor modelo junto con su nombre.
    """

    print(f"\n{'=' * 55}")
    print("  ENTRENAMIENTO FINAL + MATRICES DE CONFUSIÓN")
    print(f"{'=' * 55}")

    mejor_nombre = df_cv["ROC-AUC"].idxmax()
    mejor_modelo = None

    # figura principal con metricas y graficos
    fig = plt.figure(figsize=(18, 5 * len(MODELOS)))

    gs = gridspec.GridSpec(
        len(MODELOS),
        3,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
    )

    for i, (nombre, modelo) in enumerate(MODELOS.items()):

        modelo.fit(X, y)

        y_pred = modelo.predict(X)
        y_proba = modelo.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, y_proba)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        tag = " ★ MEJOR" if nombre == mejor_nombre else ""

        print(
            f"\n  {nombre}{tag}\n"
            f"    AUC={auc:.4f}  "
            f"Precision={prec:.4f}  "
            f"Recall={rec:.4f}  "
            f"F1={f1:.4f}"
        )

        if nombre == mejor_nombre:
            mejor_modelo = modelo

        # matriz de confusion
        ax_cm = fig.add_subplot(gs[i, 0])

        cm = confusion_matrix(y, y_pred)

        disp = ConfusionMatrixDisplay(
            cm,
            display_labels=["Activo", "Churn"],
        )

        disp.plot(
            ax=ax_cm,
            colorbar=False,
            cmap="Blues",
        )

        ax_cm.set_title(
            f"{nombre}\nMatriz de Confusión",
            fontsize=11,
        )

        # curva roc del modelo
        ax_roc = fig.add_subplot(gs[i, 1])

        fpr, tpr, _ = roc_curve(y, y_proba)

        ax_roc.plot(
            fpr,
            tpr,
            color="#185FA5",
            lw=2,
            label=f"AUC = {auc:.3f}",
        )

        ax_roc.plot(
            [0, 1],
            [0, 1],
            "k--",
            lw=1,
            alpha=.4,
        )

        ax_roc.fill_between(
            fpr,
            tpr,
            alpha=.08,
            color="#185FA5",
        )

        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")

        ax_roc.set_title(
            f"{nombre}\nCurva ROC",
            fontsize=11,
        )

        ax_roc.legend(
            loc="lower right",
            fontsize=10,
        )

        # importancia de variables
        ax_imp = fig.add_subplot(gs[i, 2])

        if hasattr(modelo, "feature_importances_"):

            importances = pd.Series(
                modelo.feature_importances_,
                index=X.columns,
            ).nlargest(10).sort_values()

            importances.plot(
                kind="barh",
                ax=ax_imp,
                color="#185FA5",
                alpha=.8,
            )

            ax_imp.set_title(
                f"{nombre}\nTop 10 Features",
                fontsize=11,
            )

            ax_imp.set_xlabel("Importancia")

        else:

            # logistic regression usa coeficientes
            coefs = pd.Series(
                np.abs(modelo.coef_[0]),
                index=X.columns,
            ).nlargest(10).sort_values()

            coefs.plot(
                kind="barh",
                ax=ax_imp,
                color="#5DCAA5",
                alpha=.8,
            )

            ax_imp.set_title(
                f"{nombre}\nTop 10 Coeficientes |β|",
                fontsize=11,
            )

            ax_imp.set_xlabel("|Coeficiente|")

    fig.suptitle(
        "Evaluación de Modelos — ChurnWatch",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    plt.savefig(
        OUTPUT_DIR / "evaluacion_modelos.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print("\n  📊  evaluacion_modelos.png guardado")

    return mejor_modelo, mejor_nombre


# comparativa visual entre modelos

def graficar_comparativa(df_cv: pd.DataFrame) -> None:
    """Barras lado a lado con las 4 métricas para los 3 modelos."""

    metricas = ["ROC-AUC", "Precision", "Recall", "F1"]

    x = np.arange(len(metricas))
    width = 0.22

    colores = [
        "#B4B2A9",
        "#5DCAA5",
        "#185FA5",
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for idx, (nombre, fila) in enumerate(df_cv.iterrows()):

        vals = [fila[m] for m in metricas]

        bars = ax.bar(
            x + idx * width,
            vals,
            width,
            label=nombre,
            color=colores[idx],
            alpha=.9,
        )

        for bar, val in zip(bars, vals):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + .005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(metricas)

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")

    ax.set_title(
        "Comparativa de Modelos — Validación Cruzada (k=5)",
        fontsize=12,
        fontweight="bold",
    )

    ax.legend(
        loc="upper right",
        fontsize=9,
    )

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "comparativa_modelos.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print("  📊  comparativa_modelos.png guardado")


# guardar mejor modelo en disco

def guardar_mejor_modelo(
    modelo: object,
    nombre: str,
    scaler: object,
    feature_columns: list[str],
) -> None:

    path = OUTPUT_DIR / "best_model.pkl"

    with open(path, "wb") as f:

        pickle.dump(
            {
                "nombre": nombre,
                "modelo": modelo,
                "scaler": scaler,
                "feature_columns": feature_columns,
            },
            f,
        )

    print(f"\n  ✅  Mejor modelo guardado → {path}")
    print(f"     Modelo: {nombre}")


# flujo principal del entrenamiento

def main() -> None:

    # cargar y preparar datos
    X, y, scaler = cargar_datos()

    # ejecutar validacion cruzada
    df_cv = evaluar_con_cv(X, y)

    # entrenar modelos y generar metricas
    mejor_modelo, mejor_nombre = entrenar_y_evaluar(X, y, df_cv)

    # generar comparativa visual
    graficar_comparativa(df_cv)

    # guardar mejor modelo
    guardar_mejor_modelo(
        mejor_modelo,
        mejor_nombre,
        scaler,
        X.columns.tolist(),
    )

    print(f"\n{'=' * 55}")
    print("  LISTO — archivos generados:")
    print("    · best_model.pkl")
    print("    · scaler.pkl")
    print("    · evaluacion_modelos.png")
    print("    · comparativa_modelos.png")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()