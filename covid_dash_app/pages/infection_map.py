from dash import dcc, html
import dash

dash.register_page(__name__,name="Infection map",order=1)

layout = html.Div([dcc.Markdown(
    """
    In development, to be announced...
    """,
    style={'white-space':'normal',
           'text-align':'left'}
    )
])