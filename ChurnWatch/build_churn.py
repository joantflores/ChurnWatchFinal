"""
Generador de variable proxy `churn` para Customer Shopping Trends.

Objetivo: Crear una columna binaria llamada `churn` utilizando reglas de negocio
basadas en comportamiento de compra.

Regla aplicada:
Un cliente se considera churn (= 1) cuando:
- Tiene baja frecuencia de compra:
    Annually
    Every 3 Months
    Quarterly

Y además tiene pocas compras previas:
    Previous Purchases <= percentil 25 del dataset
    En este caso el valor utilizado es 13

Salida:
Se genera un nuevo archivo: shopping_trends_churn.csv
"""

# Primero , se importan las librerías necesarias para el procesamiento de datos 
# y la manipulación de archivos.
from pathlib import Path

import pandas as pd

# Funciones principales del motor de churn, incluyendo el umbral de entrenamiento 
# y la función para calcular churn.
from churn_engine import (
    CHURN_UMBRAL_ENTRENAMIENTO,
    calcular_churn_regla,
    normalizar_columnas_df,
)

# Primero se define la ruta raíz del proyecto, que se utiliza para 
# localizar los archivos de entrada y salida.
ROOT = Path(__file__).resolve().parent

# Definir la ruta del archivo de entrada, que se espera que sea un CSV con los datos
INFILE = ROOT / "shopping_trends.csv"

# Archivo de salida con la columna churn incluida
OUTFILE = ROOT / "shopping_trends_churn.csv"

# Funcion principal para construir la variable `churn` a partir del DataFrame original.
# Esta función normaliza los nombres de las columnas, calcula el churn usando la
# regla de negocio definida, y agrega la nueva columna al DataFrame.

def build_churn(
    df: pd.DataFrame,
    p25: int | None = CHURN_UMBRAL_ENTRENAMIENTO
) -> pd.DataFrame:

    # Normalizar nombres de columnas
    df = normalizar_columnas_df(df)

    # Calcular churn usando reglas de negocio
    churn, _ = calcular_churn_regla(
        df,
        umbral_fijo=p25
    )

    # Agregar columna churn al DataFrame
    df["churn"] = churn.values

    return df


# Función principal que ejecuta todo el proceso: lectura del dataset, generación de la variable churn,
# guardado del nuevo CSV, y muestra de métricas básicas sobre el churn generado.
def main() -> None:

    # Verificar existencia del archivo
    if not INFILE.exists():

        print(
            f"No se encontró {INFILE}. "
            "Coloca el CSV de Customer Shopping Trends "
            "como 'shopping_trends.csv'"
        )

        return

    # Intentar leer el archivo con diferentes codificaciones para evitar errores de decodificación.
    try:
        # Intentar lectura estándar UTF-8
        df = pd.read_csv(INFILE, encoding="utf-8")

    except UnicodeDecodeError:

        try:
            # Segundo intento usando latin-1
            df = pd.read_csv(INFILE, encoding="latin-1")

        except Exception:

            # Último recurso:
            # leer binario y reemplazar caracteres inválidos
            with open(INFILE, "rb") as f:
                raw = f.read()

            import io as _io

            text = raw.decode(
                "utf-8",
                errors="replace"
            )

            df = pd.read_csv(_io.StringIO(text))


    # Construir la variable churn usando la función definida anteriormente
    out = build_churn(df)

    # Guardar el nuevo DataFrame con la columna churn en un nuevo archivo CSV
    out.to_csv(OUTFILE, index=False)


    # Mostrar métricas básicas sobre el churn generado, como el balance de clases y la tasa de churn.
    counts = out["churn"].value_counts()

    # Calcular la tasa de churn como el porcentaje de clientes que se consideran churn.
    churn_rate = out["churn"].mean() * 100

    # Imprimir la ruta del archivo guardado para confirmar que el proceso se completó correctamente.
    print(f"Guardado: {OUTFILE}")

    # Mostrar el balance de clases (cuántos clientes son churn vs no churn) y la tasa de churn.
    print(
        f"Balance de clases:\n"
        f"{counts.to_string()}"
    )

    print(
        f"Churn rate proxy: "
        f"{churn_rate:.2f}%"
    )

if __name__ == "__main__":
    main()