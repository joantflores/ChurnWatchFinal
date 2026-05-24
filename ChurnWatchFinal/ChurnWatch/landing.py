"""
ChurnWatch — Landing Page
Paso 1: Landing page de bienvenida.
Ejecutar: python landing.py
Abrir:    http://localhost:8050
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="ChurnWatch",
)

BLUE       = "#185FA5"
BLUE_LIGHT = "#E6F1FB"
TEXT_PRI   = "#1a1a1a"
TEXT_SEC   = "#6b7280"
BG_PAGE    = "#f5f5f3"
BG_CARD    = "#ffffff"
BORDER     = "0.5px solid #e5e7eb"

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

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": BG_PAGE,
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "fontFamily": "'Inter', 'Segoe UI', sans-serif",
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

                # Tarjetas de características
                html.Div(
                    feature_cards,
                    style={
                        "display": "flex", "gap": "10px",
                        "width": "100%", "marginBottom": "2rem",
                    },
                ),

                # Botón CTA — navega a la app principal
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

if __name__ == "__main__":
    app.run(debug=True)