"""
ChurnWatch — Landing Page
Paso 1: Landing page de bienvenida.
Ejecutar: python landing.py
Abrir:    http://localhost:8050
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Inicializar aplicación Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="ChurnWatch",
)

# Paleta principal de colores
BLUE = "#185FA5"
BLUE_LIGHT = "#E6F1FB"

# Colores de texto
TEXT_PRI = "#1a1a1a"
TEXT_SEC = "#6b7280"

# Colores de fondo y bordes
BG_PAGE = "#f5f5f3"
BG_CARD = "#ffffff"
BORDER = "0.5px solid #e5e7eb"

# Estilo reutilizable para tarjetas informativas
card_style = {
    "background": BG_CARD,
    "border": BORDER,
    "borderRadius": "12px",
    "padding": "1.25rem 1rem",
    "textAlign": "center",
    "flex": "1",
}

# Lista de características principales del sistema
features = [
    (
        "📂",
        "Sube tu archivo",
        "CSV o Excel con los datos de tus clientes",
    ),
    (
        "⚡",
        "Análisis automático",
        "El sistema evalúa cada cliente al instante",
    ),
    (
        "🎯",
        "Actúa a tiempo",
        "Recibe recomendaciones personalizadas",
    ),
]

# Construcción dinámica de tarjetas de funcionalidades
feature_cards = [
    html.Div(
        [
            # Ícono principal de la tarjeta
            html.Div(
                icon,
                style={
                    "fontSize": "22px",
                    "marginBottom": "8px",
                },
            ),

            # Título de la funcionalidad
            html.P(
                title,
                style={
                    "fontSize": "13px",
                    "fontWeight": "500",
                    "color": TEXT_PRI,
                    "margin": "0 0 4px",
                },
            ),

            # Descripción corta
            html.P(
                desc,
                style={
                    "fontSize": "11px",
                    "color": TEXT_SEC,
                    "margin": 0,
                    "lineHeight": "1.4",
                },
            ),
        ],
        style=card_style,
    )
    for icon, title, desc in features
]

# Layout principal de la landing page
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

        # Contenedor central de contenido
        html.Div(
            style={
                "maxWidth": "520px",
                "width": "100%",
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "textAlign": "center",
            },
            children=[

                # Badge superior descriptivo
                html.Span(
                    "Detección de churn · Retail",
                    style={
                        "fontSize": "11px",
                        "padding": "4px 14px",
                        "borderRadius": "20px",
                        "background": BLUE_LIGHT,
                        "color": BLUE,
                        "marginBottom": "1.5rem",
                        "display": "inline-block",
                    },
                ),

                # Título principal de la página
                html.H1(
                    [
                        "Mantén a tus clientes donde deben estar: ",
                        html.Span(
                            "contigo",
                            style={"color": BLUE},
                        ),
                    ],
                    style={
                        "fontSize": "clamp(24px, 4vw, 32px)",
                        "fontWeight": "500",
                        "color": TEXT_PRI,
                        "lineHeight": "1.3",
                        "marginBottom": "1rem",
                    },
                ),

                # Texto descriptivo de la plataforma
                html.P(
                    "Sube los datos de tu negocio y descubre qué clientes "
                    "están en riesgo de abandonarte — sin necesitar "
                    "conocimientos técnicos.",
                    style={
                        "fontSize": "15px",
                        "color": TEXT_SEC,
                        "lineHeight": "1.6",
                        "marginBottom": "2rem",
                        "maxWidth": "400px",
                    },
                ),

                # Sección de tarjetas informativas
                html.Div(
                    feature_cards,
                    style={
                        "display": "flex",
                        "gap": "10px",
                        "width": "100%",
                        "marginBottom": "2rem",
                    },
                ),

                # Botón principal para entrar al sistema
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

                    # Ruta hacia la aplicación principal
                    href="/app",
                ),
            ],
        ),
    ],
)

# Ejecutar servidor en modo debug
if __name__ == "__main__":
    app.run(debug=True)