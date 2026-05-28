"""
ChurnWatch — App Principal
Paso 2: Carga de archivo CSV/XLSX + tabla de predicciones + detalle de cliente.
Ejecutar: python app.py
Abrir:    http://localhost:8050
"""

import base64
import io
from pathlib import Path

import dash
from dash import html, dcc, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
from typing import Any, Optional

from churn_engine import (
    CHURN_UMBRAL_ENTRENAMIENTO,
    FREQUENCIAS_COMPRA,
    calcular_churn_regla,
    explicar_factores_churn,
    normalizar_columnas_df,
    probabilidad_desde_factores,
    rellenar_factores_probabilidad,
)

ROOT = Path(__file__).resolve().parent

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="ChurnWatch",
    suppress_callback_exceptions=True,
)

BLUE        = "#185FA5"
BLUE_LIGHT  = "#E6F1FB"
BLUE_MID    = "#B5D4F4"
RED         = "#E24B4A"
RED_LIGHT   = "#FCEBEB"
RED_DARK    = "#A32D2D"
AMBER       = "#BA7517"
AMBER_LIGHT = "#FAEEDA"
GREEN       = "#3B6D11"
GREEN_LIGHT = "#EAF3DE"
TEXT_PRI    = "#1a1a1a"
TEXT_SEC    = "#6b7280"
BG_PAGE     = "#f5f5f3"
BG_CARD     = "#ffffff"
BG_SEC      = "#f0efea"
BORDER      = "0.5px solid #e5e7eb"
FONT        = "'Inter', 'Segoe UI', sans-serif"

# ── Landing components ───────────────────────────────────────────────────────
card_style = {
    "background": BG_CARD,
    "border": BORDER,
    "borderRadius": "12px",
    "padding": "1.25rem 1rem",
    "textAlign": "center",
    "flex": "1",
}

features = [
    ("📂", "Sube tu archivo",      "CSV o Excel con los datos de tus clientes"),
    ("⚡", "Análisis automático",  "El sistema evalúa cada cliente al instante"),
    ("🎯", "Actúa a tiempo",       "Recibe recomendaciones personalizadas"),
]

feature_cards = [
    html.Div(
        [
            html.Div(icon, style={"fontSize": "22px", "marginBottom": "8px"}),
            html.P(title, style={"fontSize": "13px", "fontWeight": "500",
                                  "color": TEXT_PRI, "margin": "0 0 4px"}),
            html.P(desc,  style={"fontSize": "11px", "color": TEXT_SEC,
                                  "margin": 0, "lineHeight": "1.4"}),
        ],
        style=card_style,
    )
    for icon, title, desc in features
]

def landing_layout():
    return html.Div(
        style={
            "minHeight": "100vh",
            "background": BG_PAGE,
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontFamily": FONT,
            "padding": "2rem",
        },
        children=[
            # Contenedor central
            html.Div(
                style={"maxWidth": "520px", "width": "100%",
                       "display": "flex", "flexDirection": "column",
                       "alignItems": "center", "textAlign": "center"},
                children=[
                    # Badge
                    html.Span(
                        "Detección de churn · Retail",
                        style={
                            "fontSize": "11px", "padding": "4px 14px",
                            "borderRadius": "20px", "background": BLUE_LIGHT,
                            "color": BLUE, "marginBottom": "1.5rem",
                            "display": "inline-block",
                        },
                    ),
                    # Título principal
                    html.H1(
                        [
                            "Mantén a tus clientes donde deben estar: ",
                            html.Span("contigo", style={"color": BLUE}),
                        ],
                        style={
                            "fontSize": "clamp(24px, 4vw, 32px)",
                            "fontWeight": "500",
                            "color": TEXT_PRI,
                            "lineHeight": "1.3",
                            "marginBottom": "1rem",
                        },
                    ),
                    # Subtítulo
                    html.P(
                        "Sube los datos de tu negocio y descubre qué clientes "
                        "están en riesgo de abandonarte — sin necesitar "
                        "conocimientos técnicos.",
                        style={
                            "fontSize": "15px", "color": TEXT_SEC,
                            "lineHeight": "1.6", "marginBottom": "2rem",
                            "maxWidth": "400px",
                        },
                    ),

                    # Tarjetas de caracteristicas
                    html.Div(
                        feature_cards,
                        style={
                            "display": "flex", "gap": "10px",
                            "width": "100%", "marginBottom": "2rem",
                        },
                    ),
                    # Boton CTA — navega a la app principal
                    dcc.Link(
                        html.Button(
                            "Entrar al sistema →",
                            style={
                                "background": BLUE,
                                "color": "#E6F1FB",
                                "border": "none",
                                "padding": "12px 36px",
                                "borderRadius": "12px",
                                "fontSize": "14px",
                                "fontWeight": "500",
                                "cursor": "pointer",
                            },
                        ),
                        href="/app", 
                    ),
                ],
            ),
        ],
    )

def get_risk(pct: int) -> str:
    if pct >= 65:
        return "Alto"

    if pct >= 40:
        return "Medio"

    return "Bajo"

def risk_badge(risk: str) -> html.Span:
    colors = {
        "Alto":  (RED_LIGHT,   RED_DARK),
        "Medio": (AMBER_LIGHT, AMBER),
        "Bajo":  (GREEN_LIGHT, GREEN),
    }

    bg, fg = colors.get(risk, (BG_SEC, TEXT_SEC))

    return html.Span(
        f"{risk} riesgo",
        style={
            "fontSize": "11px", "padding": "3px 10px",
            "borderRadius": "20px", "background": bg, "color": fg,
        },
    )

RECOMENDACIONES = {
    "Alto":  "Contactar pronto con una oferta personalizada — cupón de "
             "reactivación o descuento especial. Incluir en campaña de "
             "email segmentada.",

    "Medio": "Una pequeña acción puede marcar la diferencia: recordatorio "
             "de productos favoritos, oferta de envío gratis o puntos dobles.",
    "Bajo":  "Cliente estable. Mantener comunicación regular y considerar "
             "programa de fidelidad para reforzar la relación.",
}

MODEL_INFO = {"enabled": False, "mode_label": "Modo regla de negocio"}
MODEL_PAYLOAD: dict[str, Any] = {}
CHURN_UMBRAL = CHURN_UMBRAL_ENTRENAMIENTO

try:
    model_file = ROOT / "best_model.pkl"
    if model_file.exists():
        loaded = joblib.load(model_file)
        if isinstance(loaded, dict) and loaded.get("modelo") is not None:
            MODEL_PAYLOAD = loaded
            cfg = loaded.get("churn_config") or {}
            CHURN_UMBRAL = int(cfg.get("umbral_previous_purchases", CHURN_UMBRAL_ENTRENAMIENTO))
            model_name = loaded.get("nombre", "Modelo")
            MODEL_INFO = {"enabled": True, "mode_label": f"Modo ML: {model_name}"}

except Exception:
    MODEL_INFO = {"enabled": False, "mode_label": "Modo regla de negocio"}

COL_ALIASES = {
    "customer_id":             ["customer_id", "id", "customer", "cliente", "customer id"],
    "frequency_of_purchases":  ["frequency_of_purchases", "frequency", "frecuencia", "frequency of purchases"],
    "previous_purchases":      ["previous_purchases", "purchases", "compras", "previous purchases"],
    "subscription_status":     ["subscription_status", "subscription", "suscripcion", "subscription status"],
    "purchase_amount_usd":     ["purchase_amount_usd", "amount", "monto", "purchase amount (usd)", "purchase amount"],
    "discount_applied":        ["discount_applied", "discount", "descuento", "discount applied"],
    "promo_code_used":         ["promo_code_used", "promo code used", "promo code"],
    "payment_method":          ["payment_method", "payment method", "payment"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas del archivo a los nombres estándar del sistema."""
    return normalizar_columnas_df(df)

def parse_file(contents: str, filename: str) -> Optional[pd.DataFrame]:
    """Decodifica y parsea el archivo subido (CSV o Excel)."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    ext = filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "csv":
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None
        df.columns = [c.lower().strip() for c in df.columns]
        df = normalize_columns(df)
        return df
    except Exception:
        return None

def _prob_y_factores(row: pd.Series, umbral: int) -> tuple[int, str]:
    pct = probabilidad_desde_factores(
        row.get("frequency_of_purchases", "Monthly"),
        row.get("previous_purchases", 0),
        subscription=row.get("subscription_status", "Yes"),
        discount=row.get("discount_applied", "Yes"),
        amount=row.get("purchase_amount_usd", 60),
        umbral=umbral,
    )
    txt = explicar_factores_churn(
        row.get("frequency_of_purchases", "Monthly"),
        row.get("previous_purchases", 0),
        subscription=row.get("subscription_status", "Yes"),
        discount=row.get("discount_applied", "Yes"),
        amount=row.get("purchase_amount_usd", 60),
        umbral=umbral,
    )
    return pct, txt


def build_results(df: pd.DataFrame) -> pd.DataFrame:
    """Churn (regla) + probabilidad con 5 factores de negocio."""
    df = rellenar_factores_probabilidad(df)

    try:
        churn_s, umbral = calcular_churn_regla(df, umbral_fijo=CHURN_UMBRAL)
    except ValueError:
        df["__churn_regla"] = 0
        df["__pct"] = 0
        df["__risk"] = "Bajo"
        df["__name"] = df.get("customer_id", pd.Series([f"Cliente {i+1}" for i in range(len(df))])).astype(str)
        df["__model_mode"] = "Error: faltan columnas de frecuencia o compras previas"
        return df

    df["__churn_regla"] = churn_s.values
    prob_fact = df.apply(lambda r: _prob_y_factores(r, umbral), axis=1)
    df["__pct"] = prob_fact.apply(lambda x: x[0])
    df["__factors"] = prob_fact.apply(lambda x: x[1])
    df["__risk"] = df["__pct"].apply(get_risk)
    df["__name"] = df.get(
        "customer_id",
        pd.Series([f"Cliente {i+1}" for i in range(len(df))]),
    ).fillna("—").astype(str)
    df["__model_mode"] = ""
    return df

def topnav() -> html.Div:
    return html.Div(
        style={
            "display": "flex", "alignItems": "center",
            "justifyContent": "space-between",
            "padding": ".75rem 1.5rem",
            "background": BG_CARD, "borderBottom": BORDER,
        },
        children=[
            html.Span(
                ["Churn", html.Span("Watch", style={"color": BLUE})],
                style={"fontSize": "15px", "fontWeight": "500",
                       "color": TEXT_PRI, "fontFamily": FONT},
            ),
            html.Div(
                style={
                    "width": "30px", "height": "30px", "borderRadius": "50%",
                    "background": BLUE_LIGHT, "display": "flex",
                    "alignItems": "center", "justifyContent": "center",
                    "fontSize": "12px", "fontWeight": "500", "color": BLUE,
                },
                children="US",
            ),
        ],
    )


def upload_zone() -> html.Div:
    return html.Div(
        id="upload-section",
        children=[
            html.Div(
                style={"marginBottom": "1.25rem"},
                children=[
                    html.Div("Sube los datos de tus clientes",
                             style={"fontSize": "16px", "fontWeight": "500",
                                    "color": TEXT_PRI, "marginBottom": "4px"}),
                    html.Div(
                        "La probabilidad usa: frecuencia, compras previas, suscripción, "
                        "descuento y monto (USD).",
                        style={"fontSize": "13px", "color": TEXT_SEC},
                    ),
                ],
            ),

            dcc.Upload(
                id="upload-data",
                children=html.Div(
                    style={"textAlign": "center"},
                    children=[
                        html.Div("📊", style={"fontSize": "28px",
                                              "marginBottom": "10px"}),
                        html.Div("Arrastra tu archivo aquí",
                                 style={"fontSize": "15px", "fontWeight": "500",
                                        "color": TEXT_PRI, "marginBottom": "6px"}),
                        html.Div("o haz clic para seleccionarlo desde tu computadora",
                                 style={"fontSize": "13px", "color": TEXT_SEC,
                                        "marginBottom": "1rem"}),
                        html.Span(
                            "Seleccionar archivo",
                            style={
                                "background": BLUE, "color": "#E6F1FB",
                                "padding": "8px 22px", "borderRadius": "8px",
                                "fontSize": "13px", "fontWeight": "500",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div("Formatos aceptados: .csv · .xlsx · .xls",
                                 style={"fontSize": "11px", "color": TEXT_SEC,
                                        "marginTop": "1rem"}),
                    ],
                ),
                style={
                    "border": f"1.5px dashed {BLUE_MID}",
                    "borderRadius": "12px",
                    "padding": "2.5rem 2rem",
                    "background": BLUE_LIGHT,
                    "cursor": "pointer",
                    "marginBottom": "1.25rem",
                },
                multiple=False,
            ),

            # Guia de columnas
            html.Div(
                style={"marginTop": "1rem"},
                children=[
                    html.Div("Columnas que debe incluir tu archivo:",
                             style={"fontSize": "12px", "fontWeight": "500",
                                    "color": TEXT_SEC, "marginBottom": "10px"}),
                    html.Div(
                        style={"display": "grid",
                               "gridTemplateColumns": "repeat(3, 1fr)",
                               "gap": "8px"},
                        children=[
                            html.Div(
                                [html.Div(col, style={"fontWeight": "500",
                                                       "fontSize": "11px",
                                                       "color": TEXT_PRI,
                                                       "marginBottom": "2px",
                                                       "fontFamily": "monospace"}),
                                 html.Div(desc, style={"fontSize": "11px",
                                                        "color": TEXT_SEC})],
                                style={
                                    "background": BG_CARD, "border": BORDER,
                                    "borderRadius": "8px", "padding": "8px 10px",
                                },
                            )
                            for col, desc in [
                                ("frequency_of_purchases", "Frecuencia · Weekly… Annually"),
                                ("previous_purchases",     "Cantidad de compras"),
                                ("subscription_status",    "Yes / No"),
                                ("discount_applied",       "Yes / No"),
                                ("purchase_amount_usd",    "Monto total USD"),
                                ("customer_id",            "Opcional · ID"),
                            ]
                        ],
                    ),
                ],
            ),
        ],
    )


def results_section() -> html.Div:
    return html.Div(
        id="results-section",
        style={"display": "none"},
        children=[
            # Encabezado
            html.Div(
                style={"display": "flex", "alignItems": "center",
                       "justifyContent": "space-between",
                       "marginBottom": "1rem", "flexWrap": "wrap", "gap": "8px"},
                children=[
                    html.Div([
                        html.Div(id="res-filename",
                                 style={"fontSize": "15px", "fontWeight": "500",
                                        "color": TEXT_PRI}),
                        html.Div(id="res-count",
                                 style={"fontSize": "12px", "color": TEXT_SEC,
                                        "marginTop": "2px"}),
                    ]),
                    html.Div(
                        style={"display": "flex", "gap": "8px",
                               "alignItems": "center"},
                        children=[
                            html.Div(id="summary-pills",
                                     style={"display": "flex", "gap": "8px"}),
                            html.Button(
                                "↑ Nuevo archivo",
                                id="btn-reset",
                                style={
                                    "padding": "6px 14px", "fontSize": "12px",
                                    "border": BORDER, "borderRadius": "8px",
                                    "background": "none", "color": TEXT_SEC,
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                    ),
                ],
            ),

            # Filtros
            html.Div(
                style={"display": "flex", "gap": "10px",
                       "marginBottom": "12px", "flexWrap": "wrap",
                       "alignItems": "center"},
                children=[
                    dcc.Input(
                        id="res-search",
                        placeholder="Buscar cliente…",
                        debounce=True,
                        style={
                            "padding": "7px 12px", "fontSize": "13px",
                            "border": f"0.5px solid #d1d5db",
                            "borderRadius": "8px", "width": "200px",
                            "background": BG_CARD, "color": TEXT_PRI,
                        },
                    ),
                    html.Div(
                        id="filter-pills",
                        style={"display": "flex", "gap": "6px"},
                        children=[
                            html.Button(
                                label,
                                id=f"pill-{fid}",
                                n_clicks=0,
                                style={
                                    "padding": "5px 14px", "fontSize": "12px",
                                    "border": f"0.5px solid {'#B5D4F4' if fid=='todos' else '#d1d5db'}",
                                    "borderRadius": "20px", "cursor": "pointer",
                                    "background": BLUE_LIGHT if fid == "todos" else "none",
                                    "color": BLUE if fid == "todos" else TEXT_SEC,
                                    "fontWeight": "500" if fid == "todos" else "400",
                                },
                            )
                            for fid, label in [
                                ("todos", "Todos"),
                                ("alto",  "Alto riesgo"),
                                ("medio", "Riesgo medio"),
                                ("bajo",  "Bajo riesgo"),
                            ]
                        ],
                    ),
                ],
            ),

            # Tabla
            html.Div(
                style={
                    "background": BG_CARD, "border": BORDER,
                    "borderRadius": "12px", "overflow": "hidden",
                },
                children=[
                    html.Table(
                        style={"width": "100%", "borderCollapse": "collapse",
                               "fontSize": "13px"},
                        children=[
                            html.Thead(
                                html.Tr([
                                    html.Th(h, style={
                                        "textAlign": "left",
                                        "fontSize": "11px", "fontWeight": "500",
                                        "color": TEXT_SEC,
                                        "padding": "10px 14px",
                                        "borderBottom": BORDER,
                                    })
                                    for h in ["Cliente", "Compras previas",
                                              "Frecuencia", "Suscripción",
                                              "Probabilidad", "Riesgo"]
                                ])
                            ),
                            html.Tbody(id="results-tbody"),
                        ],
                    ),
                ],
            ),
        ],
    )

def modal_detalle() -> dbc.Modal:
    return dbc.Modal(
        id="modal-cliente",
        is_open=False,
        centered=True,
        children=[
            dbc.ModalHeader(
                dbc.ModalTitle(html.Span(id="modal-nombre")),
                close_button=True,
            ),
            dbc.ModalBody([
                html.Div(id="modal-id",
                         style={"fontSize": "12px", "color": TEXT_SEC,
                                "marginBottom": "12px"}),
                html.Div(id="modal-riesgo-badge",
                         style={"marginBottom": "16px"}),
                html.Div("Información del cliente",
                         style={"fontSize": "11px", "fontWeight": "500",
                                "color": TEXT_SEC, "textTransform": "uppercase",
                                "letterSpacing": ".05em", "marginBottom": "8px"}),
                html.Div(id="modal-rows"),
                html.Div("Recomendación",
                         style={"fontSize": "11px", "fontWeight": "500",
                                "color": TEXT_SEC, "textTransform": "uppercase",
                                "letterSpacing": ".05em",
                                "marginTop": "16px", "marginBottom": "8px"}),
                html.Div(id="modal-rec",
                         style={
                             "fontSize": "13px", "lineHeight": "1.5",
                             "color": TEXT_SEC, "background": BG_SEC,
                             "borderRadius": "8px", "padding": "10px 12px",
                             "whiteSpace": "pre-wrap",
                         }),
            ]),
        ],
    )

def view_selector() -> html.Div:
    return html.Div(
        style={"display": "flex", "gap": "1rem", "marginBottom": "2rem", "borderBottom": BORDER},
        children=[
            html.Button("Carga Masiva", id="tab-btn-batch", n_clicks=0,
                        style={"background": "none", "border": "none", "borderBottom": f"2px solid {BLUE}", "color": BLUE, "padding": "10px 15px", "cursor": "pointer", "fontWeight": "500"}),
            html.Button("Ingreso Manual", id="tab-btn-manual", n_clicks=0,
                        style={"background": "none", "border": "none", "borderBottom": "2px solid transparent", "color": TEXT_SEC, "padding": "10px 15px", "cursor": "pointer", "fontWeight": "500"}),
        ]
    )

def manual_entry_section() -> html.Div:
    return html.Div(
        id="manual-section",
        style={"display": "none"},
        children=[
            html.Div("Evaluación de cliente individual",
                     style={"fontSize": "16px", "fontWeight": "500", "color": TEXT_PRI, "marginBottom": "1rem"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1rem", "marginBottom": "1.5rem"},
                children=[
                    html.Div([
                        html.Label("Frecuencia de compra", style={"fontSize": "12px", "color": TEXT_SEC}),
                        dcc.Dropdown(
                            id="manual-freq",
                            options=[{"label": f, "value": f} for f in FREQUENCIAS_COMPRA],
                            value="Monthly",
                            clearable=False,
                        )
                    ]),
                    html.Div([
                        html.Label("Compras previas", style={"fontSize": "12px", "color": TEXT_SEC}),
                        dbc.Input(id="manual-prev", type="number", min=0, value=5, style={"height": "36px"})
                    ]),
                    html.Div([
                        html.Label("Monto promedio (USD)", style={"fontSize": "12px", "color": TEXT_SEC}),
                        dbc.Input(id="manual-amount", type="number", min=0, value=50, style={"height": "36px"})
                    ]),
                    html.Div([
                        html.Label("Suscripción activa", style={"fontSize": "12px", "color": TEXT_SEC}),
                        dcc.Dropdown(
                            id="manual-subs",
                            options=[{"label": "Sí", "value": "Yes"}, {"label": "No", "value": "No"}],
                            value="No",
                            clearable=False
                        )
                    ]),
                    html.Div([
                        html.Label("Descuento aplicado", style={"fontSize": "12px", "color": TEXT_SEC}),
                        dcc.Dropdown(
                            id="manual-disc",
                            options=[{"label": "Sí", "value": "Yes"}, {"label": "No", "value": "No"}],
                            value="No",
                            clearable=False
                        )
                    ]),
                ]
            ),
            html.Button(
                "Evaluar cliente",
                id="btn-manual-predict",
                style={
                    "background": BLUE, "color": "#E6F1FB", "border": "none",
                    "padding": "10px 24px", "borderRadius": "8px", "fontSize": "13px",
                    "fontWeight": "500", "cursor": "pointer"
                }
            ),
            html.Div(id="manual-results", style={"marginTop": "2rem"})
        ]
    )

app.layout = html.Div(
    style={"fontFamily": FONT, "minHeight": "100vh",
           "background": BG_PAGE, "color": TEXT_PRI},
    children=[
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content'),
    ],
)

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/app':
        return html.Div([
            # Datos en memoria (sin persistencia entre sesiones)
            dcc.Store(id="store-results"),  # lista de dicts con los resultados
            dcc.Store(id="store-filter", data="todos"),
            dcc.Store(id="store-selected"),  # fila seleccionada para el modal

            topnav(),

            html.Div(
                style={"maxWidth": "960px", "margin": "0 auto",
                       "padding": "1.5rem 1rem"},
                children=[
                    view_selector(),
                    upload_zone(),
                    manual_entry_section(),
                    results_section(),
                ],
            ),

            modal_detalle(),
        ])
    else:
        return landing_layout()

@app.callback(
    Output("store-results", "data"),
    Output("upload-section", "style"),
    Output("results-section", "style"),
    Output("res-filename", "children"),
    Output("res-count", "children"),
    Output("summary-pills", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def process_upload(contents, filename):
    if not contents:
        raise dash.exceptions.PreventUpdate

    df = parse_file(contents, filename)
    if df is None:
        return (dash.no_update,) * 6

    results_df = build_results(df)
    records = results_df.to_dict("records")

    alto  = sum(1 for r in records if r.get("__risk") == "Alto")
    medio = sum(1 for r in records if r.get("__risk") == "Medio")
    bajo  = sum(1 for r in records if r.get("__risk") == "Bajo")

    pills = [
        html.Span(f"{alto} alto riesgo",
                  style={"fontSize": "12px", "padding": "4px 12px",
                         "borderRadius": "20px", "fontWeight": "500",
                         "background": RED_LIGHT, "color": RED_DARK}),
        html.Span(f"{medio} riesgo medio",
                  style={"fontSize": "12px", "padding": "4px 12px",
                         "borderRadius": "20px", "fontWeight": "500",
                         "background": AMBER_LIGHT, "color": AMBER}),
        html.Span(f"{bajo} bajo riesgo",
                  style={"fontSize": "12px", "padding": "4px 12px",
                         "borderRadius": "20px", "fontWeight": "500",
                         "background": GREEN_LIGHT, "color": GREEN}),
    ]

    return (
        records,
        {"display": "none"},          
        {"display": "block"},         
        filename,
        f"{len(records)} clientes analizados",
        pills,
    )

@app.callback(
    Output("upload-section", "style", allow_duplicate=True),
    Output("manual-section", "style", allow_duplicate=True),
    Output("results-section", "style", allow_duplicate=True),
    Output("tab-btn-batch", "style", allow_duplicate=True),
    Output("tab-btn-manual", "style", allow_duplicate=True),
    Input("tab-btn-batch", "n_clicks"),
    Input("tab-btn-manual", "n_clicks"),
    State("store-results", "data"),
    prevent_initial_call=True
)
def toggle_tabs(n_batch, n_manual, records):
    ctx_id = ctx.triggered_id
    active_style = {"background": "none", "border": "none", "borderBottom": f"2px solid {BLUE}", "color": BLUE, "padding": "10px 15px", "cursor": "pointer", "fontWeight": "500"}
    inactive_style = {"background": "none", "border": "none", "borderBottom": "2px solid transparent", "color": TEXT_SEC, "padding": "10px 15px", "cursor": "pointer", "fontWeight": "500"}
    
    if ctx_id == "tab-btn-manual":
        return {"display": "none"}, {"display": "block"}, {"display": "none"}, inactive_style, active_style
    else:
        if records:
            return {"display": "none"}, {"display": "none"}, {"display": "block"}, active_style, inactive_style
        return {"display": "block"}, {"display": "none"}, {"display": "none"}, active_style, inactive_style

@app.callback(
    Output("store-results", "data",     allow_duplicate=True),
    Output("upload-section", "style",   allow_duplicate=True),
    Output("results-section", "style",  allow_duplicate=True),
    Output("manual-section", "style",   allow_duplicate=True),
    Output("tab-btn-batch", "style",    allow_duplicate=True),
    Output("tab-btn-manual", "style",   allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_upload(n):
    active_style = {"background": "none", "border": "none", "borderBottom": f"2px solid {BLUE}", "color": BLUE, "padding": "10px 15px", "cursor": "pointer", "fontWeight": "500"}
    inactive_style = {"background": "none", "border": "none", "borderBottom": "2px solid transparent", "color": TEXT_SEC, "padding": "10px 15px", "cursor": "pointer", "fontWeight": "500"}
    return None, {"display": "block"}, {"display": "none"}, {"display": "none"}, active_style, inactive_style

@app.callback(
    Output("store-filter", "data"),
    Output("pill-todos", "style"),
    Output("pill-alto",  "style"),
    Output("pill-medio", "style"),
    Output("pill-bajo",  "style"),
    Input("pill-todos", "n_clicks"),
    Input("pill-alto",  "n_clicks"),
    Input("pill-medio", "n_clicks"),
    Input("pill-bajo",  "n_clicks"),
    prevent_initial_call=True,
)
def update_filter(*_):
    active = ctx.triggered_id.replace("pill-", "") if ctx.triggered_id else "todos"

    def pill_style(fid):
        is_active = fid == active
        return {
            "padding": "5px 14px", "fontSize": "12px",
            "border": f"0.5px solid {'#B5D4F4' if is_active else '#d1d5db'}",
            "borderRadius": "20px", "cursor": "pointer",
            "background": BLUE_LIGHT if is_active else "none",
            "color": BLUE if is_active else TEXT_SEC,
            "fontWeight": "500" if is_active else "400",
        }

    return active, pill_style("todos"), pill_style("alto"), pill_style("medio"), pill_style("bajo")

@app.callback(
    Output("results-tbody", "children"),
    Input("store-results", "data"),
    Input("store-filter",  "data"),
    Input("res-search",    "value"),
)
def render_table(records, active_filter, search):
    if not records:
        return []

    search = (search or "").lower()
    filtered = [
        r for r in records
        if (active_filter == "todos" or r.get("__risk", "").lower() == active_filter)
        and (not search or search in str(r.get("__name", "")).lower()
             or search in str(r.get("customer_id", "")).lower())
    ]

    if not filtered:
        return [html.Tr(html.Td(
            "Sin resultados para este filtro.",
            colSpan=6,
            style={"textAlign": "center", "padding": "2rem",
                   "color": TEXT_SEC, "fontSize": "13px"},
        ))]

    risk_colors = {
        "Alto":  (RED_LIGHT,   RED_DARK),
        "Medio": (AMBER_LIGHT, AMBER),
        "Bajo":  (GREEN_LIGHT, GREEN),
    }
    risk_labels = {"Alto": "Alto riesgo", "Medio": "Riesgo medio", "Bajo": "Bajo riesgo"}

    rows = []
    for r in filtered:
        pct  = r.get("__pct", 0)
        risk = r.get("__risk", "Bajo")
        name = r.get("__name", "—")
        cid  = str(r.get("customer_id", ""))
        bg, fg = risk_colors.get(risk, (BG_SEC, TEXT_SEC))

        # Barra de probabilidad
        bar = html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "6px"},
            children=[
                html.Div(
                    style={"width": "70px", "height": "6px",
                           "background": BG_SEC, "borderRadius": "4px"},
                    children=html.Div(style={
                        "width": f"{pct}%", "height": "6px",
                        "background": fg, "borderRadius": "4px",
                    }),
                ),
                html.Span(f"{pct}%", style={"fontSize": "12px",
                                             "color": TEXT_PRI}),
            ],
        )

        rows.append(
            html.Tr(
                id={"type": "row-cliente", "index": cid},
                n_clicks=0,
                style={"cursor": "pointer", "borderBottom": BORDER},
                children=[
                    html.Td([
                        html.Div(name, style={"fontWeight": "500"}),
                        html.Div(cid, style={"fontSize": "11px",
                                              "color": TEXT_SEC}),
                    ], style={"padding": "10px 14px"}),
                    html.Td(r.get("previous_purchases", "—"),
                            style={"padding": "10px 14px"}),
                    html.Td(r.get("frequency_of_purchases", "—"),
                            style={"padding": "10px 14px", "fontSize": "12px",
                                   "color": TEXT_SEC}),
                    html.Td(r.get("subscription_status", "—"),
                            style={"padding": "10px 14px", "fontSize": "12px"}),
                    html.Td(bar, style={"padding": "10px 14px"}),
                    html.Td(
                        html.Span(
                            risk_labels.get(risk, risk),
                            style={"fontSize": "11px", "padding": "3px 10px",
                                   "borderRadius": "20px",
                                   "background": bg, "color": fg},
                        ),
                        style={"padding": "10px 14px"},
                    ),
                ],
            )
        )

    return rows

@app.callback(
    Output("modal-cliente",     "is_open"),
    Output("modal-nombre",      "children"),
    Output("modal-id",          "children"),
    Output("modal-riesgo-badge","children"),
    Output("modal-rows",        "children"),
    Output("modal-rec",         "children"),
    Input({"type": "row-cliente", "index": dash.ALL}, "n_clicks"),
    State("store-results", "data"),
    prevent_initial_call=True,
)
def open_modal(n_clicks_list, records):
    if not any(n_clicks_list) or not records:
        raise dash.exceptions.PreventUpdate

    triggered = ctx.triggered_id
    if not triggered:
        raise dash.exceptions.PreventUpdate

    cid = triggered["index"]
    row = next((r for r in records if str(r.get("customer_id", "")) == cid), None)
    if not row:
        raise dash.exceptions.PreventUpdate

    name = row.get("__name", cid)
    pct  = row.get("__pct", 0)
    risk = row.get("__risk", "Bajo")

    risk_colors = {
        "Alto":  (RED_LIGHT,   RED_DARK),
        "Medio": (AMBER_LIGHT, AMBER),
        "Bajo":  (GREEN_LIGHT, GREEN),
    }
    risk_labels = {"Alto": "Alto riesgo", "Medio": "Riesgo medio", "Bajo": "Bajo riesgo"}
    bg, fg = risk_colors.get(risk, (BG_SEC, TEXT_SEC))

    badge = html.Span(
        f"{risk_labels.get(risk, risk)} · {pct}% probabilidad de abandono",
        style={"fontSize": "12px", "padding": "4px 14px",
               "borderRadius": "20px", "background": bg, "color": fg},
    )

    info_rows = []
    campos = [
        ("Compras anteriores",  "previous_purchases"),
        ("Frecuencia de compra","frequency_of_purchases"),
        ("Monto promedio (USD)","purchase_amount_usd"),
        ("Suscripción activa",  "subscription_status"),
        ("Descuento aplicado",  "discount_applied"),
    ]
    for label, key in campos:
        val = row.get(key)
        if val is not None and str(val).strip() not in ("", "nan"):
            info_rows.append(
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between",
                           "fontSize": "13px", "padding": "5px 0",
                           "borderBottom": BORDER},
                    children=[
                        html.Span(label, style={"color": TEXT_SEC}),
                        html.Span(str(val), style={"fontWeight": "500",
                                                    "color": TEXT_PRI}),
                    ],
                )
            )

    return (
        True,
        name,
        cid,
        badge,
        info_rows,
        (
            RECOMENDACIONES.get(risk, "")
            if not row.get("__factors")
            else f"{RECOMENDACIONES.get(risk, '')}\n\nFactores: {row.get('__factors')}"
        ),
    )

@app.callback(
    Output("manual-results", "children"),
    Input("btn-manual-predict", "n_clicks"),
    State("manual-freq", "value"),
    State("manual-prev", "value"),
    State("manual-amount", "value"),
    State("manual-subs", "value"),
    State("manual-disc", "value"),
    prevent_initial_call=True
)
def process_manual(n_clicks, freq, prev, amount, subs, disc):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
        
    df = pd.DataFrame([{
        "customer_id": "Manual",
        "frequency_of_purchases": freq,
        "previous_purchases": prev,
        "purchase_amount_usd": amount,
        "subscription_status": subs,
        "discount_applied": disc
    }])
    
    res_df = build_results(df)
    row = res_df.iloc[0]
    
    pct = row.get("__pct", 0)
    risk = row.get("__risk", "Bajo")
    factors = row.get("__factors", "")
    
    risk_colors = {
        "Alto":  (RED_LIGHT,   RED_DARK),
        "Medio": (AMBER_LIGHT, AMBER),
        "Bajo":  (GREEN_LIGHT, GREEN),
    }
    bg, fg = risk_colors.get(risk, (BG_SEC, TEXT_SEC))
    
    rec = RECOMENDACIONES.get(risk, "")
    if factors:
        rec += f"\n\nFactores: {factors}"
        
    return html.Div(
        style={"background": BG_CARD, "border": BORDER, "borderRadius": "12px", "padding": "1.5rem", "animation": "fadeIn 0.5s ease-in"},
        children=[
            html.Div("Resultado del análisis", style={"fontSize": "15px", "fontWeight": "500", "color": TEXT_PRI, "marginBottom": "1rem"}),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "1rem", "marginBottom": "1.5rem"},
                children=[
                    html.Div(
                        style={"width": "200px", "height": "8px", "background": BG_SEC, "borderRadius": "4px"},
                        children=html.Div(style={"width": f"{pct}%", "height": "100%", "background": fg, "borderRadius": "4px"})
                    ),
                    html.Span(f"{pct}% Probabilidad", style={"fontSize": "14px", "fontWeight": "500", "color": fg}),
                    html.Span(f"Riesgo {risk}", style={"fontSize": "12px", "padding": "4px 12px", "borderRadius": "20px", "background": bg, "color": fg, "fontWeight": "500"})
                ]
            ),
            html.Div("Recomendación", style={"fontSize": "11px", "fontWeight": "500", "color": TEXT_SEC, "textTransform": "uppercase", "letterSpacing": ".05em", "marginBottom": "8px"}),
            html.Div(rec, style={"fontSize": "13px", "lineHeight": "1.5", "color": TEXT_SEC, "background": BG_SEC, "borderRadius": "8px", "padding": "10px 12px", "whiteSpace": "pre-wrap"})
        ]
    )

if __name__ == "__main__":
    app.run(debug=True)
