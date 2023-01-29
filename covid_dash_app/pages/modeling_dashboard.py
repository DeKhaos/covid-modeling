import dash
from dash import html,callback,ctx
from dash.dependencies import Input, Output
import re
import numpy as np
import dash_bootstrap_components as dbc
from dash_extensions.enrich import ALL

from covid_dash_app.pages import modeling_tab1,modeling_tab2,modeling_tab3

dash.register_page(__name__,name="Modelling",order=0)

param_tabs = dbc.Tabs([
    dbc.Tab(modeling_tab1.tab1_content,label="Preparation"),
    dbc.Tab(modeling_tab2.tab2_content,label="Processing"),
    dbc.Tab(modeling_tab3.tab3_content,label="Modeling"),
    ],
    id="model_param_tabs",
    active_tab="tab-0",
)

modals = html.Div([modeling_tab1.tab1_modal,
                   modeling_tab2.tab2_modal,
                   modeling_tab3.tab3_modal])
tooltips = html.Div([modeling_tab1.tab1_tooltip,
                     modeling_tab2.tab2_tooltip,
                     modeling_tab3.tab3_tooltip])

figure_tabs = [modeling_tab1.tab1_figure,
               modeling_tab2.tab2_figure,
               modeling_tab3.tab3_figure]

left_col = dbc.Col(dbc.Row([
                            param_tabs,
                            modals,
                            tooltips,
                            ]),
                   width=3)


right_col = dbc.Col(
    [dbc.Collapse(item,is_open=False,id={"model_figures":idx})
     for idx,item in enumerate(figure_tabs)],
    id="model_figures_wrapper",
    width=9
    )

#page layout
layout = dbc.Row([left_col,right_col])

@callback(
    Output({'model_figures':ALL},'is_open'),
    Input('model_param_tabs','active_tab')
    )
def select_model_figure(tab_click):
    """
    Open the matching figure with relevant tab and hide the rest.
    """
    n = len(ctx.outputs_list)
    collapse_array = np.full(n,False)
    tab_n = int(re.search(r"(?<=-).*",tab_click)[0])
    collapse_array[tab_n]=True
    return collapse_array.tolist()
