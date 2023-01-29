import dash_bootstrap_components as dbc
import dash
from dash import html
import os 
from covid_dash_app.visualization import app

server = app.server

# navigation bar
nav = dbc.Nav(
    [dbc.NavLink(page['name'],href=page["relative_path"])
     for page in dash.page_registry.values() if page["relative_path"]!="/"
     ],
    id='nav_bar',
    fill=True,
    style={"background-color":"DarkCyan"},
    pills=True
)

app.layout = html.Div([
    html.A(html.P("COVID DASHBOARD",
           style={"text-align":"center","color":"yellow","font-size":"40px"})
           ,href="/",
           style={"text-decoration":'None'}
          ),
    nav,
    dash.page_container,
])

if __name__ == '__main__':
    
    check_path = [('cache/output_cache',250),
                  ('cache/server_cache',70),
                  ('cache/census_cache',30)]
    #handle auto delete cache if too many exists on local machine
    for item in check_path:
        path = item[0]
        list_dir = os.listdir(path)
        if len(list_dir)>item[1]:
            for file in list_dir:
                os.remove(os.path.join(path, file))
    
    app.run_server(debug=False,use_reloader=False)