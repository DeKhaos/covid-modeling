import pandas as pd
import numpy as np
import dash
from dash import callback,ctx
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash_extensions.enrich import Input,Output,State,ServersideOutput,dcc,\
    html,ALL,MATCH,FileSystemStore
from dash_extensions.enrich import callback as callback_extension
import datetime,json,os,time
from multiprocessing import Manager,Pool
import re
from functools import partial
from zipfile import ZipFile
from covid_science import utility_func
from covid_science.workers import raw_wrapper_func,vaccine_wrapper_func
from covid_science.c_preparation import model_initial,model_parameter
from covid_science.modeling import fill_empty_param,reproduction_number
#------------------------------------------------------------------

#Tab2: Preparation data

tab2_stored_data = html.Div([
    #check criteria,True mean ok
    dcc.Store(id={'code':'p1t2','name':'normal_input',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t2','name':'b_d_rate',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t2','name':'recovery_dist',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t2','name':'death_dist',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t2','name':'tab2_w_vaccine_weight',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t2','name':'tab2_vn_vaccine_weight',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t2','name':'tab2_base_vac_ratio',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t2','name':'tab2_vaccine_target',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t2','name':'tab2_empty_f_weight',"data_type":"input_check"},
              data=True),
    dcc.Store(id='tab2_initial_status',data=0),
    
    #preloaded dataframe
    dcc.Store(id={'code':'p1t2','name':'vn_processed_data',"data_type":"server"}),
    dcc.Store(id={'code':'p1t2','name':'vn_processed_data_by_province',"data_type":"server"}),
    dcc.Store(id={'code':'p1t2','name':'vn_modeling_data',"data_type":"server"}),
    dcc.Store(id={'code':'p1t2','name':'vn_modeling_data_by_province',"data_type":"server"}),
    dcc.Store(id={'code':'p1t2','name':'w_processed_data',"data_type":"server"}),
    dcc.Store(id={'code':'p1t2','name':'w_processed_data_by_country',"data_type":"server"}),
    dcc.Store(id={'code':'p1t2','name':'w_modeling_data',"data_type":"server"}),
    dcc.Store(id={'code':'p1t2','name':'w_modeling_data_by_country',"data_type":"server"}),
    
    #modified dataframe
    dcc.Store(id={'code':'p1t2','name':'vn_processed_data_modified',"data_type":"update"}),
    dcc.Store(id={'code':'p1t2','name':'vn_processed_data_by_province_modified',
                  "data_type":"update"}),
    dcc.Store(id={'code':'p1t2','name':'vn_modeling_data_modified',"data_type":"update"}),
    dcc.Store(id={'code':'p1t2','name':'vn_modeling_data_by_province_modified',
                  "data_type":"update"}),
    dcc.Store(id={'code':'p1t2','name':'w_processed_data_modified',"data_type":"update"}),
    dcc.Store(id={'code':'p1t2','name':'w_processed_data_by_country_modified',
                  "data_type":"update"}),
    dcc.Store(id={'code':'p1t2','name':'w_modeling_data_modified',"data_type":"update"}),
    dcc.Store(id={'code':'p1t2','name':'w_modeling_data_by_country_modified',
                  "data_type":"update"}),
    
    dcc.Store(id={'code':'p1t2','name':'birth_data',"data_type":"census"}),
    dcc.Store(id={'code':'p1t2','name':'death_data',"data_type":"census"}),
    dcc.Store(id={'code':'p1t2','name':'w_population',"data_type":"census"}),
    dcc.Store(id={'code':'p1t2','name':'vn_population',"data_type":"census"}),
    #plot triggering
    dcc.Store(id='tab2_figure_plot_trigger',data=0),
    dcc.Store(id='tab2_figure_dropdown_trigger',data=0),
    
    #vaccine weight
    dcc.Store(id={'code':'p1t2','name':'tab2_w_vac_weight',"data_type":"internal"}),
    dcc.Store(id={'code':'p1t2','name':'tab2_vn_vac_weight',"data_type":"internal"}),
    #store processing parameters
    dcc.Store(id="tab2_process_params"),
    #check if update database button switch necessary
    dcc.Store(id={'code':'p1t2','name':'switch_database_button'},
              data=[False,False,False,False])
])

tab2_content = [dbc.Form([tab2_stored_data,
    dbc.Alert("Please fill in all required parameters.",
              id='tab2_inputs_alert',is_open=False,color='danger',
              class_name="c_alert"),
    dcc.Interval(id='tab2_data_initiator',max_intervals=1,n_intervals=0,
                 disabled=True),
    dbc.Label(html.Li("Color code:"),class_name="ms-1",width='auto',
              id='tab2_color_code_label'),
    dbc.Row([dbc.Col([dbc.Button("Default",size='sm',
                                 style = {'font-size':'0.8em'}),
                      dbc.Button("Data customizations unsuccessful",
                                 size='sm',color="warning",
                                 style = {'font-size':'0.8em'}),
                      dbc.Button("Data customizations successful",size='sm',
                                 color="success",
                                 style = {'font-size':'0.8em'}),],
                     width='auto')],
            justify='start'),
    
    dbc.Label(html.Li('Processed database selection:'),class_name="ms-1",width='auto',
              id='tab2_pre_model_select_label'),
    dbc.Label('Vietnam:',width=7,style={'font-size':'0.8em'}),
    html.Div(dbc.RadioItems(
            id={'code':'p1t2','name':'select_vn_database',"purpose":"database_check"},
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Default", "value": 1},
                {"label": "Modified", "value": 2,"disabled":True},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    dbc.Label('Each Vietnam\'s province:',width=7,style={'font-size':'0.8em'}),
    html.Div(dbc.RadioItems(
            id={'code':'p1t2','name':'select_vnbp_database',"purpose":"database_check"},
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Default", "value": 1},
                {"label": "Modified", "value": 2,"disabled":True},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    dbc.Label('World:',width=7,style={'font-size':'0.8em'}),
    html.Div(dbc.RadioItems(
            id={'code':'p1t2','name':'select_w_database',"purpose":"database_check"},
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Default", "value": 1},
                {"label": "Modified", "value": 2,"disabled":True},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    dbc.Label('Each country:',width=7,style={'font-size':'0.8em'}),
    html.Div(dbc.RadioItems(
            id={'code':'p1t2','name':'select_wbc_database',"purpose":"database_check"},
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Default", "value": 1},
                {"label": "Modified", "value": 2,"disabled":True},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    html.Hr(),
    dbc.Label(html.Li('Preparation database:'),class_name="ms-1",width='auto',
              id='tab2_pre_database_label'),
    dbc.Switch(id="tab2_proc_database_all_switch",
               label_id="tab2_proc_database_all_switch_label",
               label="Use all databases",
               value=False,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    dbc.Collapse([
    dbc.Row(dbc.RadioItems(options=[{"label": 'VietNam', "value": 'vietnam'},
                                    {"label":'World',"value":'world'}],
                           id='tab2_proc_database',
                           value='vietnam',
                           label_style = {'font-size':'0.8em'},
                           inline=True)
            ),
    dbc.Switch(id="tab2_proc_database_division_switch",
               label="Use database by each province",
               value=False,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    dbc.Collapse(dbc.Button("Choose section",size='sm',className="ms-auto",
                            id="tab2_proc_database_division_button",
                            style = {'font-size':'0.8em'}),
                 id='tab2_proc_database_division_button_collapse',
                 is_open=False)],
        id='tab2_proc_database_all_switch_collapse',
        is_open=True),
    
    dbc.Label(html.Li('Census parameters:'),class_name="ms-1",width='auto',
              id='tab2_census_label'),
    
    dbc.Row([dbc.Label('Recruit non-vaccinated ratio:',width=7,
                       id='tab2_p_label',style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t2","name":"tab2_p","type":"input"},
                               type='number',size='sm',required=True,
                               min=0.0,max=1.0,step=0.01,value=1.0,
                               className="mt-1 mb-1",
                               persistence_type="memory",
                               persistence=True),
                     width=4)
             ]),
    
    dbc.Switch(id="tab2_auto_b_d_rate",
               label="Auto calculate birth & death rate from database",
               label_id="tab2_auto_b_d_rate_label",
               value=True,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    dbc.Collapse(
        [dcc.RangeSlider(
            min=2010,
            max=2021,
            step=1,
            marks={2010:{'label':'2010'},
                   2021:{'label':'2021'}},
            value=[2019,2021],
            tooltip={"placement": "top"},
            id="tab2_range_b_d_rate")],
        id='tab2_range_b_d_rate_collapse',
        is_open=True),
    dbc.Collapse(
        [dbc.Row([
            dbc.Label(
                'Birth rate per day:',width=7,
                id='tab2_manual_b_rate_label',
                style={'font-size':'0.8em'}),
            dbc.Col(dbc.Input(
                id={"code":"p1t2","name":"tab2_manual_b_d_rate",
                    "type":"special_input","index":0},
                type='number',required=True,size='sm',
                min=0,max=1,step=0.0000001,value=0.0000418,
                className="mt-1 mb-1"),
                    width=4)]),
        dbc.Row([
            dbc.Label(
                'Death rate per day:',width=7,id='tab2_manual_d_rate_label',
                style={'font-size':'0.8em'}),
            dbc.Col(dbc.Input(
                id={"code":"p1t2","name":"tab2_manual_b_d_rate",
                    "type":"special_input","index":1},
                type='number',required=True,size='sm',
                min=0,max=1,step=0.0000001,value=0.0000184,
                className="mt-1 mb-1"),
                    width=4)]),
        ],
        id='tab2_manual_b_d_rate_collapse',
        is_open=False),
    
    dbc.Row([
        dbc.Label(
            'Starting year of disease:',width=7,id='tab2_population_year_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_population_year","type":"input"},
            type='number',required=True,
            size='sm',min=2010,max=2021,step=1,value=2019,
            className="mt-1 mb-1"),
                width=4)]),
    html.Hr(),
    dbc.Label(html.Li('Processing parameters of case data:'),
              class_name="ms-1",width='auto',
              id='tab2_raw_process_label'),
    
    dbc.Row([
        dbc.Label(
            'Raw data rolling window:',
            width=7,id='tab2_raw_rolling_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_raw_rolling","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=1,value=10,
            className="mt-1 mb-1"),
                width=4)]),
    
    dbc.Row([
        dbc.Label(
            'Outlier trim percent:',
            width=7,id='tab2_trim_perc_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_trim_perc","type":"input"},
            type='number',required=True,
            size='sm',min=0,max=100,value=25,
            className="mt-1 mb-1"),
                width=4)]),
    
    dbc.Row([
        dbc.Label(
            'Potential recovery after (days):',
            width=7,id='tab2_recovery_after_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_recovery_after","type":"input"},
            type='number',required=True,
            size='sm',min=0,value=7,step=1,
            className="mt-1 mb-1"),
                width=4)]),
    
    dbc.Row([
        dbc.Label(
            'Number of weeks for recovery distribution:',
            width=7,id='tab2_recovery_dist_week_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_recovery_dist_week","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=1,value=5,
            className="mt-1 mb-1"),
                width=4)]),
    
    dbc.Label("Recovery ratio distribution per week:",
              id='tab2_recovery_dist_label',
              style={'font-size':'0.8em'},class_name="ms-1",width='auto'),
    dmc.SimpleGrid(
        id='tab2_recovery_dist',
        cols=4,
        children=[
            dbc.Input(
                id={"code":"p1t2","name":"recovery_dist","type":"special_input",
                    "week":1},
                step=0.0001,max=1,min=0,value=0.3,required=True,
                type='number',size='sm'),
            dbc.Input(
                id={"code":"p1t2","name":"recovery_dist","type":"special_input",
                    "week":2},
                step=0.0001,max=1,min=0,value=0.5,required=True,
                type='number',size='sm'),
            dbc.Input(
                id={"code":"p1t2","name":"recovery_dist","type":"special_input",
                    "week":3},
                step=0.0001,max=1,min=0,value=0.1,required=True,
                type='number',size='sm'),
            dbc.Input(
                id={"code":"p1t2","name":"recovery_dist","type":"special_input",
                    "week":4},
                step=0.0001,max=1,min=0,value=0.05,required=True,
                type='number',size='sm'),
            dbc.Input(
                id={"code":"p1t2","name":"recovery_dist","type":"special_input",
                    "week":5},
                step=0.0001,max=1,min=0,value=0.05,required=True,
                type='number',size='sm'),
        ],
        spacing='xs',
        class_name='m-1 p-1 border border-1 border-secondary'
    ),
    dbc.Alert("The sum of recovery ratio distribution must be 1.",
              id='tab2_recovery_dist_alert',is_open=False,color='danger',
              class_name="c_alert"),
    
    dbc.Row([
        dbc.Label(
            'Fatality percent rolling window:',
            width=7,id='tab2_death_rolling_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_death_rolling","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=1,value=14,
            className="mt-1 mb-1"),
                width=4)]),
    
    dbc.Switch(id="tab2_use_original_death",
               label="Use original fatality case",
               label_id="tab2_use_original_death_label",
               value=True,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    
    dbc.Row([
        dbc.Label(
            'Potential fatality after (days):',
            width=7,id='tab2_death_after_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_death_after","type":"input"},
            type='number',required=True,
            size='sm',min=0,value=4,step=1,
            className="mt-1 mb-1"),
                width=4)]),
    
    dbc.Row([
        dbc.Label(
            'Number of weeks for fatality distribution:',
            width=7,id='tab2_death_dist_week_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_death_dist_week","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=1,value=7,
            className="mt-1 mb-1"),
                width=4)]),
    
    dbc.Label("Fatality distribution per week:",id='tab2_death_dist_label',
              style={'font-size':'0.8em'},class_name="ms-1",width='auto'),
    dmc.SimpleGrid(
        id='tab2_death_dist',
        cols=4,
        children=[
            dbc.Input(
                id={"code":"p1t2","name":"death_dist","type":"special_input",
                    "week":1},
                step=0.0001,max=1,min=0,value=0.4,required=True,
                type='number',size='sm'),
            dbc.Input(
                id={"code":"p1t2","name":"death_dist","type":"special_input",
                    "week":2},
                step=0.0001,max=1,min=0,value=0.3,required=True,
                type='number',size='sm'),
            dbc.Input(
                id={"code":"p1t2","name":"death_dist","type":"special_input",
                    "week":3},
                step=0.0001,max=1,min=0,value=0.2,required=True,
                type='number',size='sm'),
            dbc.Input(
                id={"code":"p1t2","name":"death_dist","type":"special_input",
                    "week":4},
                step=0.0001,max=1,min=0,value=0.025,required=True,
                type='number',size='sm'),
            dbc.Input(
                id={"code":"p1t2","name":"death_dist","type":"special_input",
                    "week":5},
                step=0.0001,max=1,min=0,value=0.025,required=True,
                type='number',size='sm'),
            dbc.Input(
                id={"code":"p1t2","name":"death_dist","type":"special_input",
                    "week":6},
                step=0.0001,max=1,min=0,value=0.025,required=True,
                type='number',size='sm'),
            dbc.Input(
                id={"code":"p1t2","name":"death_dist","type":"special_input",
                    "week":7},
                step=0.0001,max=1,min=0,value=0.025,required=True,
                type='number',size='sm'),
        ],
        spacing='xs',
        class_name='m-1 p-1 border border-1 border-secondary'
    ),
    dbc.Alert("The sum of death ratio distribution must be 1.",
              id='tab2_death_dist_alert',is_open=False,color='danger',
              class_name="c_alert"),
    
    dbc.Row([
        dbc.Label(
            'Raw data randomness limit:',
            width=7,id='tab2_raw_method_limit_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_raw_method_limit","type":"input"},
            type='number',required=True,
            size='sm',min=0,step=1,value=0,
            className="mt-1 mb-1"),
                width=4)]),
    
    dbc.Row([
        dbc.Label(
            'Raw random seed:',
            width=7,id='tab2_raw_seed_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_raw_seed","type":"input"},
            type='number',required=True,
            size='sm',min=0,step=1,value=1,
            className="mt-1 mb-1"),
                width=4)]),
    html.Hr(),
    dbc.Label(html.Li('Processing parameters of vaccine data:'),
              class_name="ms-1",width='auto',
              id='tab2_vaccine_process_label'),
    
    dbc.Label(html.U('Vaccine type attributes:'),
              class_name="ms-1",width='auto',
              id='tab2_vaccine_type_label'),
    
    html.Br(),
    
    dbc.Label(html.I('Type1: Non Replicating Viral Vector'),
              class_name="ms-1",width='auto',
              id='tab2_vaccine_type1_label',
              style={'font-size':'0.8em'}),
    
    dmc.Grid(
        id='tab2_vaccine_type1',
        children=[
            dmc.Col(dbc.Label(
                    'Interval for 2nd dose:',
                    width='auto',
                    id='tab2_vaccine_type1_label1',
                    style={'font-size':'0.7em'}),
                span=4),
            
            dmc.Col(dbc.Input(
                id={"code":"p1t2","name":"tab2_vaccine_type1_param1",
                    "type":"input"},
                step=1,min=0,value=56,required=True,
                type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Immunity threshold after 2nd dose:',
                    width='auto',
                    id='tab2_vaccine_type1_label2',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type1_param2",
                        "type":"input"},
                    step=1,min=0,value=14,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Immunity threshold after booster dose:',
                    width='auto',
                    id='tab2_vaccine_type1_label3',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type1_param3",
                        "type":"input"},
                    step=1,min=0,value=14,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    '2nd dose Wear-off time:',
                    width='auto',
                    id='tab2_vaccine_type1_label4',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type1_param4",
                        "type":"input"},
                    step=1,min=0,value=150,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Booster dose Wear-off time:',
                    width='auto',
                    id='tab2_vaccine_type1_label5',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type1_param5",
                        "type":"input"},
                    step=1,min=0,value=150,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            
        ],
        gutter='xs',
        align='center',
        class_name='mx-1'
    ),
    
    dbc.Label(html.I('Type2: Messenger RNA'),
              class_name="ms-1",width='auto',
              id='tab2_vaccine_type2_label',
              style={'font-size':'0.8em'}),
    dmc.Grid(
        id='tab2_vaccine_type2',
        children=[
            dmc.Col(dbc.Label(
                    'Interval for 2nd dose:',
                    width='auto',
                    id='tab2_vaccine_type2_label1',
                    style={'font-size':'0.7em'}),
                span=4),
            
            dmc.Col(dbc.Input(
                id={"code":"p1t2","name":"tab2_vaccine_type2_param1",
                    "type":"input"},
                step=1,min=0,value=28,required=True,
                type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Immunity threshold after 2nd dose:',
                    width='auto',
                    id='tab2_vaccine_type2_label2',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type2_param2",
                        "type":"input"},
                    step=1,min=0,value=14,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Immunity threshold after booster dose:',
                    width='auto',
                    id='tab2_vaccine_type2_label3',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type2_param3",
                        "type":"input"},
                    step=1,min=0,value=14,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    '2nd dose Wear-off time:',
                    width='auto',
                    id='tab2_vaccine_type2_label4',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type2_param4",
                        "type":"input"},
                    step=1,min=0,value=150,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Booster dose Wear-off time:',
                    width='auto',
                    id='tab2_vaccine_type2_label5',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type2_param5",
                        "type":"input"},
                    step=1,min=0,value=150,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            
        ],
        gutter='xs',
        align='center',
        class_name='mx-1'
    ),
    
    dbc.Label(html.I('Type3: Inactivated vaccine'),
              class_name="ms-1",width='auto',
              id='tab2_vaccine_type3_label',
              style={'font-size':'0.8em'}),
    dmc.Grid(
        id='tab2_vaccine_type3',
        children=[
            dmc.Col(dbc.Label(
                    'Interval for 2nd dose:',
                    width='auto',
                    id='tab2_vaccine_type3_label1',
                    style={'font-size':'0.7em'}),
                span=4),
            
            dmc.Col(dbc.Input(
                id={"code":"p1t2","name":"tab2_vaccine_type3_param1",
                    "type":"input"},
                step=1,min=0,value=28,required=True,
                type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Immunity threshold after 2nd dose:',
                    width='auto',
                    id='tab2_vaccine_type3_label2',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type3_param2",
                        "type":"input"},
                    step=1,min=0,value=14,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Immunity threshold after booster dose:',
                    width='auto',
                    id='tab2_vaccine_type3_label3',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type3_param3",
                        "type":"input"},
                    step=1,min=0,value=14,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    '2nd dose Wear-off time:',
                    width='auto',
                    id='tab2_vaccine_type3_label4',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type3_param4",
                        "type":"input"},
                    step=1,min=0,value=150,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Booster dose Wear-off time:',
                    width='auto',
                    id='tab2_vaccine_type3_label5',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_type3_param5",
                        "type":"input"},
                    step=1,min=0,value=150,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            
        ],
        gutter='xs',
        align='center',
        class_name='mx-1'
    ),
    
    dbc.Switch(id="tab2_set_vaccine_perc",
               label="Manually set ratio of each vaccine type",
               label_id='tab2_set_vaccine_perc_label',
               value=False,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    
    dbc.Collapse([
        dbc.Label(html.U('Base vaccine ratio:'),
                  class_name="ms-1",width='auto',
                  id='tab2_base_vac_ratio_label'),
    
        dmc.Grid(
            [
            dmc.Col(dbc.Label(
                    'Type1:',
                    width='auto',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Type2:',
                    class_name="ms-1",width='auto',
                    style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Type3:',
                    class_name="ms-1",width='auto',
                    style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Start-end date:',
                    class_name="ms-1",width='auto',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dmc.ActionIcon(
                    DashIconify(icon="ic:baseline-plus",color='LightBlue'), 
                    id="tab2_add_base_vac_ratio", 
                    n_clicks=0),
                span=1),
            dmc.Col(dmc.ActionIcon(
                    DashIconify(icon="ic:baseline-minus",color='LightBlue'), 
                    id="tab2_remove_base_vac_ratio", 
                    n_clicks=0),
                span=1),
            
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_base_vac_ratio",
                       "type":"special_input","index":0,"params":0},
                   step=0.01,min=0,value=1,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_base_vac_ratio",
                       "type":"special_input","index":0,"params":1},
                   step=0.01,min=0,value=1,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_base_vac_ratio",
                       "type":"special_input","index":0,"params":2},
                   step=0.01,min=0,value=1,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dmc.DateRangePicker(
                   id={"code":"p1t2","name":"tab2_base_vac_ratio",
                       "type":"special_input","index":0,"params":3},
                   minDate=datetime.date(2019, 12, 1),
                   maxDate=datetime.datetime.now().date(),
                   allowSingleDateInRange=True,
                   hideOutsideDates=True,
                   value=[datetime.date(2019, 12, 1),
                          datetime.datetime.now().date()],
                   inputFormat="YYYY-MM-DD",
                   amountOfMonths=2,
                   size='xs',
                   labelSeparator=':'
                   ),
                span=6),
            ],
            gutter=5,
            align='center',
            class_name='mx-1 border border-1 border-secondary',
            id='tab2_base_vac_ratio'),
        ],
        id='tab2_base_vac_ratio_collapse',
        is_open=False),
    
    dbc.Collapse([
        dbc.Label(html.U('World: vaccine volumn weight:'),
                  class_name="ms-1",width='auto',
                  id='tab2_w_vaccine_weight_label'),
        dmc.Grid(
            [
            dmc.Col(dbc.Label(
                    'Vaccine name:',
                    width='auto',style={'font-size':'0.7em'}),
                span=6),
            dmc.Col(dbc.Label(
                    'Type:',
                    class_name="ms-1",width='auto',
                    style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Weight:',
                    class_name="ms-1",width='auto',
                    style={'font-size':'0.7em'}),
                span=2),
            
            dmc.Col(dmc.ActionIcon(
                    DashIconify(icon="ic:baseline-plus",color='LightBlue'), 
                    id="tab2_w_add_vaccine_name", 
                    n_clicks=0),
                span=1),
            dmc.Col(dmc.ActionIcon(
                    DashIconify(icon="ic:baseline-minus",color='LightBlue'), 
                    id="tab2_w_remove_vaccine_name", 
                    n_clicks=0),
                span=1),
            
            dmc.Col(dbc.Label(
                    'Vaxzevria',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"label_name","index":0}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":0,"params":0},
                   step=1,min=0,max=2,value=0,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":0,"params":1},
                   step=0.01,min=0,value=10,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
             
            dmc.Col(dbc.Label(
                    'Ad26.COV 2-S',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"label_name","index":1}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":1,"params":0},
                   step=1,min=0,max=2,value=0,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":1,"params":1},
                   step=0.01,min=0,value=4,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
              
            dmc.Col(dbc.Label(
                    'Covishield',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"label_name","index":2}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":2,"params":0},
                   step=1,min=0,max=2,value=0,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":2,"params":1},
                   step=0.01,min=0,value=0.2,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
               
            dmc.Col(dbc.Label(
                    'Gam-Covid-Vac',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"label_name","index":3}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":3,"params":0},
                   step=1,min=0,max=2,value=0,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":3,"params":1},
                   step=0.01,min=0,value=0.2,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
                
            dmc.Col(dbc.Label(
                    'Comirnaty',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"label_name","index":4}),
                span=6),
             dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"special_input","index":4,"params":0},
                    step=1,min=0,max=2,value=1,required=True,disabled=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                 span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":4,"params":1},
                   step=0.01,min=0,value=100,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            
            dmc.Col(dbc.Label(
                    'Spikevax',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"label_name","index":5}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":5,"params":0},
                   step=1,min=0,max=2,value=1,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":5,"params":1},
                   step=0.01,min=0,value=35,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
             
            dmc.Col(dbc.Label(
                    'BBIBP-CorV',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"label_name","index":6}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":6,"params":0},
                   step=1,min=0,max=2,value=2,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":6,"params":1},
                   step=0.01,min=0,value=2,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
              
            dmc.Col(dbc.Label(
                    'Coronavac',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"label_name","index":7}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":7,"params":0},
                   step=1,min=0,max=2,value=2,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":7,"params":1},
                   step=0.01,min=0,value=3,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
               
            dmc.Col(dbc.Label(
                    'Others',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"label_name","index":8}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":8,"params":0},
                   step=1,min=0,max=2,value=2,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"special_input","index":8,"params":1},
                   step=0.01,min=0,value=0.5,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            ],
            # gutter='xs',
            gutter=3,
            align='center',
            class_name='mx-1 border border-1 border-secondary',
            id='tab2_w_vaccine_weight'),
        dbc.Label(html.U('Vietnam: vaccine volumn weight:'),
                  class_name="ms-1",width='auto',
                  id='tab2_vn_vaccine_weight_label'),
        dmc.Grid(
            [
            dmc.Col(dbc.Label(
                    'Vaccine name:',
                    width='auto',style={'font-size':'0.7em'}),
                span=6),
            dmc.Col(dbc.Label(
                    'Type:',
                    class_name="ms-1",width='auto',
                    style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Weight:',
                    class_name="ms-1",width='auto',
                    style={'font-size':'0.7em'}),
                span=2),
            
            dmc.Col(dmc.ActionIcon(
                    DashIconify(icon="ic:baseline-plus",color='LightBlue'), 
                    id="tab2_vn_add_vaccine_name", 
                    n_clicks=0),
                span=1),
            dmc.Col(dmc.ActionIcon(
                    DashIconify(icon="ic:baseline-minus",color='LightBlue'), 
                    id="tab2_vn_remove_vaccine_name", 
                    n_clicks=0),
                span=1),
            
            dmc.Col(dbc.Label(
                    'Vaxzevria',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                        "type":"label_name","index":0}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":0,"params":0},
                   step=1,min=0,max=2,value=0,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":0,"params":1},
                   step=0.01,min=0,value=28,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
             
            dmc.Col(dbc.Label(
                    'Gam-Covid-Vac',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                        "type":"label_name","index":1}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":1,"params":0},
                   step=1,min=0,max=2,value=0,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":1,"params":1},
                   step=0.01,min=0,value=0.7,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
              
            dmc.Col(dbc.Label(
                    'CIGB-66',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                        "type":"label_name","index":2}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":2,"params":0},
                   step=1,min=0,max=2,value=0,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":2,"params":1},
                   step=0.01,min=0,value=2.2,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
               
            dmc.Col(dbc.Label(
                    'Comirnaty',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                        "type":"label_name","index":3}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":3,"params":0},
                   step=1,min=0,max=2,value=1,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":3,"params":1},
                   step=0.01,min=0,value=40,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
                
            dmc.Col(dbc.Label(
                    'Spikevax',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                        "type":"label_name","index":4}),
                span=6),
             dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                        "type":"special_input","index":4,"params":0},
                    step=1,min=0,max=2,value=1,required=True,disabled=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                 span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":4,"params":1},
                   step=0.01,min=0,value=6,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            
            dmc.Col(dbc.Label(
                    'BBIBP-CorV',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                        "type":"label_name","index":5}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":5,"params":0},
                   step=1,min=0,max=2,value=2,required=True,disabled=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":5,"params":1},
                   step=0.01,min=0,value=22.6,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            
            dmc.Col(dbc.Label(
                    'Others',
                    width='auto',style={'font-size':'0.7em'},
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                        "type":"label_name","index":6}),
                span=6),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":6,"params":0},
                   step=1,min=0,max=2,value=2,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"special_input","index":6,"params":1},
                   step=0.01,min=0,value=0.5,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            ],
            gutter=3,
            align='center',
            class_name='mx-1 border border-1 border-secondary',
            id='tab2_vn_vaccine_weight'),
        ],
        id='tab2_vaccine_weight_collapse',
        is_open=True),
    
    dbc.Switch(id="tab2_set_vaccine_target",
               label="Set Vaccination priority",
               label_id="tab2_set_vaccine_target_label",
               value=False,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    dbc.Collapse([
        dbc.Label(html.U('Vaccination priority:'),
                  class_name="ms-1",width='auto',
                  id='tab2_vaccine_target_label'),
        dmc.Grid([
            dmc.Col(dbc.Label(
                    '1st dose:',
                    width='auto',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    '2nd dose:',
                    class_name="ms-1",width='auto',
                    style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Booster:',
                    class_name="ms-1",width='auto',
                    style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Label(
                    'Start-end date:',
                    class_name="ms-1",width='auto',
                    style={'font-size':'0.7em'}),
                span=4),
            dmc.Col(dmc.ActionIcon(
                    DashIconify(icon="ic:baseline-plus",color='LightBlue'), 
                    id="tab2_add_vaccine_target", 
                    n_clicks=0),
                span=1),
            dmc.Col(dmc.ActionIcon(
                    DashIconify(icon="ic:baseline-minus",color='LightBlue'), 
                    id="tab2_remove_vaccine_target", 
                    n_clicks=0),
                span=1),
            
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vaccine_target",
                       "type":"special_input","index":0,"params":0},
                   step=0.01,min=0,value=1,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vaccine_target",
                       "type":"special_input","index":0,"params":1},
                   step=0.01,min=0,value=1,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vaccine_target",
                       "type":"special_input","index":0,"params":2},
                   step=0.01,min=0,value=1,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
            dmc.Col(dmc.DateRangePicker(
                   id={"code":"p1t2","name":"tab2_vaccine_target",
                       "type":"special_input","index":0,"params":3},
                   minDate=datetime.date(2019, 12, 1),
                   maxDate=datetime.datetime.now().date(),
                   allowSingleDateInRange=True,
                   hideOutsideDates=True,
                   value=[datetime.date(2019, 12, 1),
                          datetime.datetime.now().date()],
                   inputFormat="YYYY-MM-DD",
                   amountOfMonths=2,
                   size='xs',
                   labelSeparator=':'
                   ),
                span=6),
            ],
            gutter=5,
            align='center',
            class_name='mx-1 border border-1 border-secondary',
            id='tab2_vaccine_target'),
        ],
        id='tab2_vaccine_target_collapse',
        is_open=False),
    
    dbc.Row([
        dbc.Label(
            'Vaccination data randomness limit:',
            width=7,id='tab2_vaccine_method_limit_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_vaccine_method_limit","type":"input"},
            type='number',required=True,
            size='sm',min=0,step=1,value=0,
            className="mt-1 mb-1"),
                width=4)]),
    
    dbc.Row([
        dbc.Label(
            'Vaccine random seed:',
            width=7,id='tab2_vaccine_seed_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_vaccine_seed","type":"input"},
            type='number',required=True,
            size='sm',min=0,step=1,value=1,
            className="mt-1 mb-1"),
                width=4)]),
    
    html.Hr(),
    dbc.Label(html.Li('Processing pre-modeling data:'),
              class_name="ms-1",width='auto',
              id='tab2_premodel_process_label'),
    
    dbc.Row([
        dbc.Label(
            'Average immunity time of post-covid patients:',
            width=7,id='tab2_r_protect_time_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_r_protect_time","type":"input"},
            type='number',required=True,
            size='sm',min=0,step=1,value=150,
            className="mt-1 mb-1"),
                width=4)],
        align='center'),
    
    dbc.Row([
        dbc.Label(
            'Infection change ratio between vaccinated and non-vaccinated individuals:',
            width=7,id='tab2_s_vac_ratio_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_s_vac_ratio","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=0.01,value=3,
            className="mt-1 mb-1"),
                width=4)],
        align='center'),
    
    dbc.Row([
        dbc.Label(
            'Force of infection rolling window:',
            width=7,id='tab2_avg_infect_t_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_avg_infect_t","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=1,value=1,
            className="mt-1 mb-1"),
                width=4)],
        align='center'),
    
    dbc.Row([
        dbc.Label(
            'Force of fatality rolling window:',
            width=7,id='tab2_avg_death_t_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_avg_death_t","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=1,value=1,
            className="mt-1 mb-1"),
                width=4)],
        align='center'),
    
    dbc.Row([
        dbc.Label(
            'Force of recovery rolling window:',
            width=7,id='tab2_avg_recov_t_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_avg_recov_t","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=1,value=1,
            className="mt-1 mb-1"),
                width=4)],
        align='center'),
    
    dbc.Row([
        dbc.Label(
            'Force of immunity rolling window:',
            width=7,id='tab2_avg_rotate_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_avg_rotate","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=1,value=1,
            className="mt-1 mb-1"),
                width=4)],
        align='center'),
        
    dbc.Label(html.U('Empty parameters filling:'),class_name="ms-1",
              id='tab2_empty_fill_label',width='auto'),
    
    dbc.Row([dbc.Label('Data points:',width=7,id='tab2_avg_data_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t2","name":"tab2_avg_data","type":"input"},
                               type='number',size='sm',
                               min=0,step=1,value=14,required=True,
                               className="mt-1 mb-1"),
                     width=4)
             ]),
    
    dbc.Label("Filling method:",class_name="mb-0",style={'font-size':'0.8em'},
              width='auto',id='tab2_empty_f_method_label'),
    dbc.Row([dbc.Col(dbc.RadioItems(options=[
                        {"label": 'linear nearest', "value": 'linear_nearest'},
                        {"label":'custom weight:',"value":2}],
                                    id='tab2_empty_f_method',
                                    value=2,
                                    label_style = {'font-size':'0.8em'}),
                    width=7),
            dbc.Col(dbc.Input(id='tab2_empty_f_weight',type='number',size='sm',
                              min=1,max=50,value=2),
                    width=4,align="end")
            ]),
    dbc.Alert("Input weight not in range [1,50].",
              id='tab2_empty_f_alert',is_open=False,color="danger",
              class_name="c_alert"),
    
    dbc.Label("Include zero values:",class_name="mb-0 mt-2",width='auto',
              style={'font-size':'0.8em'},id='tab2_include_zero_label'),
    dbc.Row(dbc.RadioItems(options=[
                        {"label": 'True', "value": True},
                        {"label":'False',"value":False}],
                                    id='tab2_include_zero',
                                    value=False,
                                    label_style = {'font-size':'0.8em'},
                            inline=True)
            ),
    
    dbc.Label(html.U(["Reproduction number R",html.Sub("o"),":"]),
              class_name="ms-1",width='auto',id='tab2_r0_label'),
    
    dbc.Row([dbc.Label('Data points:',width=7,id='tab2_r0_n_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t2","name":"tab2_r0_n","type":"input"},
                               type='number',size='sm',
                               min=0,step=1,value=30,required=True,
                               className="mt-1 mb-1"),
                     width=4)
             ]),
    
    dbc.Row([dbc.Label('Outlier trim %:',width=7,id='tab2_r0_trim_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t2","name":"tab2_r0_trim","type":"input"},
                               type='number',size='sm',
                               min=0,max=100,step=0.01,value=25,required=True,
                               className="mt-1 mb-1"),
                     width=4)                              
             ]),
    
    html.Hr(),
    dbc.Label(html.Li('Other parameters:'),class_name="ms-1",width='auto',
              id='tab2_other_param_label'),
    
    dbc.Row([
        dbc.Label(
            'Maximum processes:',
            width=7,id='tab2_max_worker_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t2","name":"tab2_max_worker","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=1,value=round(os.cpu_count()/2)+1,
            className="mt-1 mb-1"),
                width=4)]),
    dbc.Collapse([dbc.Alert(id='tab2_processing_alert',className='c_alert')],
                 id='tab2_processing_alert_collapse',
                 is_open=False,),
    dcc.Download(id="tab2_download"),
    
    ],
    className="mt-1 mb-1 bg-secondary",
    style={'color':'white'}),
    
    dbc.Row([dbc.Col([dbc.Button("Apply",id="tab2_model_Apply_Button",
                                 n_clicks=0,size='sm'),
                     dbc.Button("Download",id="tab2_model_Download_Button",
                                n_clicks=0,size='sm'),
                     ],
                     width='auto')
             ],
            justify='end'),
    ]

#Tab2 Modals

tab2_modal = html.Div([
    dbc.Modal(
        [dbc.ModalHeader('Choose data sections',close_button=False),
          dbc.ModalBody([dbc.Switch(id="tab2_proc_database_all_division",
                                    label="Select all",
                                    value=True,
                                    style = {'font-size':'0.8em'},
                                    className="mb-0 mt-2"),
                         dbc.Alert("At least 1 option must be chosen.",
                                   id='tab2_proc_database_division_checklist_alert',
                                   is_open=False,color="danger",
                                   class_name="c_alert"),
                        dbc.Checklist(id="tab2_proc_database_division_checklist",
                                      label_style = {'font-size':'0.8em'})
                        ]),
          dbc.ModalFooter(dbc.Button("Ok", id="tab2_division_option_ok",
                          className="ms-auto",n_clicks=0))
        ],
        id="tab2_division_option",
        is_open=False,
        backdrop='static',
        scrollable=True),
    dbc.Modal(
        [dbc.ModalHeader('Processing status',close_button=False),
          dbc.ModalBody([
                         dbc.Row([dbc.Col(dbc.Spinner(color="primary",size='sm'),
                                           width='auto'),
                                   dbc.Col(html.P(children="Processing...",
                                                  id='tab2_update_process_info',
                                                  style = {'font-size':'0.8em'}),
                                           width='auto')
                                   ]),
                         dbc.Collapse([dbc.Progress(
                                 id='tab2_update_process',
                                 striped=True)],
                             id='tab2_update_process_collapse',
                             is_open=False),
                        ]),
          dbc.ModalFooter(dbc.Button("Cancel", id="tab2_model_Cancel_Button1",
                          className="ms-auto",size='sm',n_clicks=0))
        ],
        id="tab2_processing_status",
        is_open=False,
        backdrop='static',
        centered=True),
    dbc.Modal(
        [dbc.ModalHeader('Downloading status',close_button=False),
          dbc.ModalBody([
                         dbc.Row([dbc.Col(dbc.Spinner(color="primary",size='sm'),
                                           width='auto'),
                                   dbc.Col(html.P(children="Downloading databases...",
                                                  style = {'font-size':'0.8em'}),
                                           width='auto')
                                   ]),
                        ]),
          dbc.ModalFooter(dbc.Button("Cancel", id="tab2_model_Cancel_Button2",
                          className="ms-auto",size='sm',n_clicks=0))
        ],
        id="tab2_downloading_status",
        is_open=False,
        backdrop='static',
        centered=True),
    
])

#Tab2 plots

tab2_figure = html.Div([
    dbc.Tabs([
    dbc.Tab(
        [utility_func.add_row_choices(
            ['Database:'], 
            [[{"label":"Processed Vietnam data","value":"vn_processed_data"},
              {"label": "Processed Vietnam data by province", 
               "value": "vn_processed_data_by_province"},
              {"label":"Vietnam modeling input data","value":"vn_modeling_data"},
              {"label":"Vietnam modeling input data by province",
               "value":"vn_modeling_data_by_province"},
              {"label":"Processed World data","value":"w_processed_data"},
              {"label":"Processed World data by country",
               "value":"w_processed_data_by_country"},
              {"label":"World modeling input data","value":"w_modeling_data"},
              {"label":"World modeling input data by country",
               "value":"w_modeling_data_by_country"}],
             ], 
            ['vn_processed_data'], 
            [{"tab2_figure_tabs":'t1','type':'select_data',"data_type":"server"}],
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        dbc.Row([
            dbc.Label('Plot points:',id='tab2_plot_point_label1',
                      style={'font-size':'0.9em'},width='auto'),
            dbc.Col(dbc.Input(id={"tab2_figure_tabs":'t1','type':'plot_point'},
                              type='number',size='sm',
                              min=2,step=1,value=240,required=True,
                              className="mt-1 mb-1"),
                    width=1),
            dbc.Col(dbc.Switch(id={"tab2_figure_tabs":'t1','type':'plot_position'},
                       label_id="tab2_plot_position_label1",
                       label="Plot from latest day",
                       value=True,
                       style = {'font-size':'0.8em'},
                       className="mb-0 mt-2"),
                    width='auto')
            ])
        ],
        label='Server data',
        id={"tab2_figure_tabs":'t1'},
        ),
    dbc.Tab(["No data",
             dbc.Collapse(
                 dbc.Row([
                     dbc.Label('Plot points:',id='tab2_plot_point_label2',
                               style={'font-size':'0.9em'},width='auto'),
                     dbc.Col(dbc.Input(id={"tab2_figure_tabs":'t2','type':'plot_point'},
                                       type='number',size='sm',
                                       min=2,step=1,value=240,required=True,
                                       className="mt-1 mb-1"),
                             width=1),
                     dbc.Col(dbc.Switch(id={"tab2_figure_tabs":'t2','type':'plot_position'},
                                label_id="tab2_plot_position_label2",
                                label="Plot from latest day",
                                value=True,
                                style = {'font-size':'0.8em'},
                                className="mb-0 mt-2"),
                             width='auto')
                     ]),
                 is_open=False
                 )
             ],
            label="Modified data",
            id={"tab2_figure_tabs":'t2'}),
    ],
    id="tab2_plot_tabs",
    active_tab="tab-0",
    style = {'font-size':'0.7em'},
    class_name='mb-1'),
    html.Div(id='tab2_figure_add_dropdown'),
    html.Div(id='tab2_figure_output'),
])

#Tab2 Tooltips

tab2_tooltip = html.Div([
    dbc.Tooltip(html.Div('''
        Color code to show the status of the relevant databases.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_color_code_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Trigger the switch to select between default or modified databases for
        the modeling step.
        
        If a database is processed , user can switch between 
        default & modified database for modeling step.
        If a database fails to process, it won't override the old modified
        database and change the color code.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_pre_model_select_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The options below allow which database is chosen for processing.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_pre_database_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Processing all available databases, including worldwide, each
        country and each Vietnam's province.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_proc_database_all_switch_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Settings for choosing appropriate census data for processing.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_census_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown(
        '''p: The fraction of birth population that is is not vaccinated.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_p_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Auto calculate daily birth & death rate per day (fraction) based on
        the number of selected years.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_auto_b_d_rate_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Apply the manually input birth rate per day for processing all selected 
        databases.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_manual_b_rate_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Apply the manually input death rate per day for processing all selected 
        databases.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_manual_d_rate_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The selected starting year, used to get population data on that year.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_population_year_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Parameters on how to process raw cases & death cases.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_raw_process_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The range of data points which used to average cases & death cases, 
        as well as remove certain outliers from that data range.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_raw_rolling_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The percentage of outlier that will be removed from rolling window.
        
        E.g: with rolling_window=16 and outlier_trim=25 meaning 4 outlier points
        will be removed, 2 from the bottom and 2 from the top of data range.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_trim_perc_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Minimum days required for probability of recovery.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_recovery_after_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The number of weeks used for recovery case distribution.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_recovery_dist_week_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The probability distribution of recovery between weeks.
        
        E.g: 30% a patient will recovered in the 1st week after 'n days' since 
        positive test, 50% on the 2nd week, 10% on the 3rd week,etc.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_recovery_dist_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The range of data points which used to average the fatality percentage, 
        as well as remove certain outliers from that data range.
        
        NOTE: If value is smaller that 'raw rolling window', it will be auto
        set to be equal to 'raw rolling window' when processing.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_death_rolling_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        It means to obtain original infected case of death case from data by 
        substracting that case instead  of using fatality percentage.
        Turn on this option to make sure that after a period of time:
            'the total starting infected case = death cases + recovery cases' 
        after that period of time.
        
        E.g: 1 infected case in day 1, 5 in day 2 and 3 in day 3. 
        After 5 days, there are total 5 fatalities. Estimation of death case 
        origin will take at most 1 from day 1, 5 from day 2, 3 from day 3 by
        random percentage distrubition. But if we use fatality percentage from 
        calculation, the real case limit of each day will be ignored.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_use_original_death_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Minimum days required for probability of fatality.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_death_after_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The number of weeks used for fatality case distribution.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_death_dist_week_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The probability distribution of fatality between weeks.
        
        E.g: 30% a patient will pass away in the 1st week after 'n days' since 
        positive test, 50% on the 2nd week, 10% on the 3rd week,etc.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_death_dist_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The upper limit which randomness will be used instead of fast 
        filling method. The main purpose of this is to keep the randomness 
        to a certain degree if necessary but still keep a fast calculation 
        speed.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_raw_method_limit_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Seed for random generation. If 'randomness limit'=0, seed won't be 
        applied.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_raw_seed_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        There will 3 main vaccine types: Non Replicating Viral Vector, RNA and
        Inactivated vaccine.
        
        Assumpt that all vaccine type will need 2 doses before fully functional.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_vaccine_process_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Choose between auto estimation of each vaccine type volume based on vaccine 
        weights or manually set the ratio between 3 vaccine types.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_set_vaccine_perc_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Manually set the vaccine volumn between 3 types over a period.
        If start date = end date, it mean that ratio will be kept from that date
        onward.
        NOTE: from top to bottom, the next ratios will override the 
        previous if conflicted.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_base_vac_ratio_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Add new vaccine name| edit the vaccine weight|type to match the current
        vaccine volumes if necessary. This will be used to automatically estimat
        the ratio between 3 vaccine types.
        
        NOTE: Vaccine metadata is obtained from WHO website.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_w_vaccine_weight_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Add new vaccine name| edit the vaccine weight|type to match the current
        vaccine volumes if necessary. This will be used to automatically estimat
        the ratio between 3 vaccine types.
        
        NOTE: Vaccine metadata is obtained from WHO website.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_vn_vaccine_weight_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        If enable, manually set the priority ratio of vaccine volumne distribution
        to 3 group of required target: 1st dose, 2nd dose, booster dose.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_set_vaccine_target_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Manually set the distribution priority between 3 target over a period.
        If start date = end date, it mean that ratio will be kept from that date
        onward.
        NOTE: from top to bottom, the next ratios will override the 
        previous if conflicted.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_vaccine_target_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The upper limit which randomness will be used instead of fast 
        filling method. The main purpose of this is to keep the randomness 
        to a certain degree if necessary but still keep a fast calculation 
        speed.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_vaccine_method_limit_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Seed for random generation. If 'randomness limit'=0, seed won't be 
        applied.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_vaccine_seed_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Parameters for preprocessing modeling input data.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_premodel_process_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The number of day which a recovered patient is considered to be like a
        fully vaccinated person.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_r_protect_time_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The ratio which vaccinated individuals are less likely to be infected
        comparing to non-vaccinated individuals.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_s_vac_ratio_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Estimate beta,beta_v parameters at a specific date based on data of n 
        days ago.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_avg_infect_t_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Estimate theta parameter at a specific date based on data of n 
        days ago.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_avg_death_t_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Estimate gamma parameter at a specific date based on data of n 
        days ago.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_avg_recov_t_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Estimate alpha,alpha0 parameters at a specific date based on data of n 
        days ago.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_avg_rotate_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''Define how to fill the empty parameter values in database left by
                         processing steps.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab2_empty_fill_label',
                delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''The number of previous data points will be taken 
                         and used for calculating the empty value.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab2_avg_data_label',
                delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Choosing the method:\n
        -linear nearest: the weight of data points will 
        decrease linearly from nearest data point to the 
        furthest.\n
        -custom weight: the next data point will weight n 
        times smaller than the previous starting from the 
        nearest data point.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_empty_f_method_label',
                delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown(
        '''Choose to include zero points in the database into calculation or not.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_include_zero_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown(
        '''The transmissability or contagiousness of an infectious disease.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_r0_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown(
        '''The number of previous data points used for calculating point average.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_r0_n_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
         The percent of data outlier will be removed when calculating point average.''',
         style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_r0_trim_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The maximum number of process which will be used for multiprocessing.
        
        It is recommended not to set this number too high (normally use around 
        half of CPU processors), otherwise it might slow down the modifying 
        process.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab2_max_worker_label',
        delay={'show': 1000}),
    dbc.Tooltip(html.Div('''The number of data points from database which will
                         be plotted.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab2_plot_point_label1',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''The number of data points from database which will
                         be plotted.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab2_plot_point_label2',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''Choose to plot from starting date or latest date.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab2_plot_position_label1',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''Choose to plot from starting date or latest date.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab2_plot_position_label2',
                delay={'show': 1000}),
])

#Tab2 callbacks

##initial data loading--------------------------------------------------------

@callback(
    Output('tab2_data_initiator','disabled'),
    Input('tab1_initial_status','data'),
    State('tab2_data_initiator','n_intervals'),
    prevent_initial_call=True
)
def tab2_trigger_data_process(initial_trigger,count):
    """
    Triggered only the first time.
    """
    if count>=1:
        return dash.no_update
    else:
        return False
    
@callback_extension(
    ServersideOutput({'code':'p1t2','name':'birth_data',"data_type":"census"},'data'),
    ServersideOutput({'code':'p1t2','name':'death_data',"data_type":"census"},'data'),
    Input('tab2_data_initiator','n_intervals'),
    Input('tab2_range_b_d_rate','value'),
    State({'code':'p1t1','name':'birth_data',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'death_data',"data_type":"census"},'data'),
    prevent_initial_call=True
)
def tab2_process_b_d_rate(trigger,year_range,birth_df,death_df):
    """
    Save processed birth&death rate data.
    """
    year_range = range(year_range[0],year_range[1]+1)
    birth_df = birth_df.loc[birth_df.Year.isin(year_range)]
    birth_df = birth_df.groupby(['Region, subregion, country or area *',
                                 'Location code',
                                 'ISO3 Alpha-code'], 
                                dropna=False).mean().reset_index()
    
    death_df = death_df.loc[death_df.Year.isin(year_range)]
    death_df = death_df.groupby(['Region, subregion, country or area *',
                                 'Location code',
                                 'ISO3 Alpha-code'], 
                                dropna=False).mean().reset_index()
    return birth_df,death_df

@callback_extension(
    ServersideOutput({'code':'p1t2','name':'w_population',"data_type":"census"},'data'),
    ServersideOutput({'code':'p1t2','name':'vn_population',"data_type":"census"},'data'),
    Input('tab2_data_initiator','n_intervals'),
    Input({"code":"p1t2","name":"tab2_population_year","type":"input"},'value'),
    State({'code':'p1t1','name':'w_population',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'vn_population',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'vn_province_code',"data_type":"internal"},'data'),
    prevent_initial_call=True
)
def tab2_process_population(trigger,start_year,w_population,vn_population,vn_province_code):
    """
    Save processed population data.
    """
    if start_year is None:
        return dash.no_update,dash.no_update
    else:
        w_population = w_population.loc[w_population['Year']==start_year].reset_index(drop=True)
        w_population['Total Population, as of 1 January'] =\
            w_population['Total Population, as of 1 January (thousands)']*1000
            
        vn_province_code = pd.DataFrame(vn_province_code,
                                        columns=['provinceName','id'])
        vn_population = vn_population.loc[
            (vn_population['year']==start_year)
            &(vn_population['criteria']=="Average population(thousand)")]
        vn_population = vn_population.melt(id_vars=['year','criteria'],
                                           var_name='provinceName',
                                           value_name='population')
        vn_population['population']=vn_population['population']*1000
        vn_population['criteria']='Average population'
        vn_population = pd.merge(vn_province_code,vn_population,on='provinceName')
        return w_population,vn_population
    
server_backend = FileSystemStore(cache_dir="./cache/server_cache",threshold=50,
                             default_timeout=0
                             )
@callback_extension(
    [ServersideOutput({'code':'p1t2','name':'vn_processed_data',"data_type":"server"},
                      'data',arg_check=False,session_check=False,
                      backend=server_backend),
     ServersideOutput({'code':'p1t2','name':'vn_processed_data_by_province',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t2','name':'vn_modeling_data',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t2','name':'vn_modeling_data_by_province',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t2','name':'w_processed_data',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t2','name':'w_processed_data_by_country',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t2','name':'w_modeling_data',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t2','name':'w_modeling_data_by_country',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
    ],
    Input('tab2_data_initiator','n_intervals'),
    memoize=True,
    prevent_initial_call=True
    )
def tab2_load_default_cached_database(click):
    """
    Save databases cache for faster data loading.
    """
    vn_processed_data=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_input/VietNamData/vn_processed_data.csv",
        parse_dates=['date'])
    vn_processed_data_by_province=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_input/VietNamData/vn_processed_data_by_province.csv",
        parse_dates=['date'])
    vn_modeling_data=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_input/VietNamData/vn_modeling_data.csv",
        parse_dates=['date'])
    vn_modeling_data_by_province=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_input/VietNamData/vn_modeling_data_by_province.csv",
        parse_dates=['date'])
    w_processed_data=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_input/WorldData/w_processed_data.csv",
        parse_dates=['date'])
    w_processed_data_by_country=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_input/WorldData/w_processed_data_by_country.csv",
        parse_dates=['date'])
    w_modeling_data=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_input/WorldData/w_modeling_data.csv",
        parse_dates=['date'])
    w_modeling_data_by_country=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_input/WorldData/w_modeling_data_by_country.csv",
        parse_dates=['date'])
    
    return [
        vn_processed_data,
        vn_processed_data_by_province,
        vn_modeling_data,
        vn_modeling_data_by_province,
        w_processed_data,
        w_processed_data_by_country,
        w_modeling_data,
        w_modeling_data_by_country
        ]

@callback(
    Output('tab2_figure_dropdown_trigger','data'),
    Input('tab2_data_initiator','n_intervals'),
    Input({'code':'p1t2','name':'vn_processed_data',"data_type":"server"},'data'),
    State('tab2_figure_dropdown_trigger','data'),
    prevent_initial_call=True
    )
def tab2_trigger_plot_process(initial_trigger1,initial_trigger2,
                              count):
    """
    Synchronize initial trigger condition of plotting.
    """
    return count+1

@callback_extension(
    [ServersideOutput({'code':'p1t2','name':'vn_processed_data_modified',
                      "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t2','name':'vn_processed_data_by_province_modified',
                  "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t2','name':'vn_modeling_data_modified',
                      "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t2','name':'vn_modeling_data_by_province_modified',
                  "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t2','name':'w_processed_data_modified',
                      "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t2','name':'w_processed_data_by_country_modified',
                  "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t2','name':'w_modeling_data_modified',
                      "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t2','name':'w_modeling_data_by_country_modified',
                  "data_type":"update"},'data')],
    Input('tab2_data_initiator','n_intervals'),
    prevent_initial_call=True
    )
def tab2_generate_uid(update_state):
    """
    Generate uid for cache access of modified databases.
    """
    outputs = [None for _ in range(len(ctx.outputs_list))]
    return outputs

#handle checking of preloaded data
@callback(
    output = Output('tab2_initial_status','data'),
    inputs = [
        Input('tab2_data_initiator','n_intervals'),
        Input({'code':'p1t2','name':'birth_data',"data_type":"census"},'data'),
        Input({'code':'p1t2','name':'vn_processed_data',"data_type":"server"},'data')
        ],
    prevent_initial_call=True
    )
def tab2_initial_data_loaded(trigger1,trigger2,trigger3):
    """Use to check if all tab preloaded data is ready"""
    return 1

##buttons interactions--------------------------------------------------------

@callback(
    Output('tab2_proc_database_all_switch_collapse','is_open'),
    Input('tab2_proc_database_all_switch','value'),
    prevent_initial_call=True
    )
def tab2_choose_all_database(switch):
    """
    Switch between choosing all database or Vietnam|World.
    """
    if switch:
        return False
    else:
        return True

@callback_extension(
    Output('tab2_proc_database_division_switch','label'),
    Output('tab2_proc_database_division_switch','value'),
    Output('tab2_proc_database_division_checklist','options'),
    Output('tab2_proc_database_division_checklist','value'),
    Output('tab2_proc_database_all_division','value'),
    Input('model_param_tabs','active_tab'),
    Input('tab2_proc_database','value'),
    Input('tab2_proc_database_all_division','value'),
    Input('tab2_proc_database_division_checklist','value'),
    State('tab2_proc_database','value'),
    State('tab1_select_w_database','value'),
    State({'code':'p1t1','name':'iso_code',"data_type":"internal"},'data'),
    State({'code':'p1t1','name':'birth_data',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'w_covid_data_latest',"data_type":"update"},'data'),
    State('tab1_select_vn_database','value'),
    State({'code':'p1t1','name':'vn_province_code',"data_type":"internal"},'data'),
    State({'code':'p1t1','name':'vn_death_data',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_death_data_latest',"data_type":"update"},'data'),
    prevent_initial_call=True
    )
def tab2_change_database(
        tab2_trigger,
        database_switch,
        all_switch,
        checklist,
        current_database,
        w_switch,iso_code,check_df,w_df_updated,
        vn_switch,province_code,vn_df,vn_df_updated):
    """
    Change between Vietnam and world modeling database and synchronize all 
    relevant buttons and modal.
    """
    check_df = check_df[['Region, subregion, country or area *',
             'ISO3 Alpha-code']].drop_duplicates()
    iso_code_dict = dict(iso_code)
    iso_code_list = np.array(iso_code)[:,0].tolist()
    province_code = dict(province_code)
    if current_database=="vietnam":
        if vn_switch==1:
            all_options = [province_code[item] for item in 
                           vn_df.columns[3:].tolist()]
        else:
            all_options = [province_code[item] for item in 
                           vn_df_updated.columns[3:].tolist()]
    else:
        if w_switch==1:
            all_options = iso_code_list
        else:
            dummy_df = w_df_updated[['iso_code','location']].drop_duplicates()
            
            all_options = pd.merge(
                check_df,dummy_df,
                left_on='ISO3 Alpha-code',
                right_on='iso_code')['iso_code'].values.tolist()
    if ctx.triggered_id in ["tab2_proc_database",
                            "model_param_tabs"]:
        if database_switch=="vietnam":
            if vn_switch==1:
                options = [{'label':item,'value':province_code[item]} for 
                           item in vn_df.columns[3:].tolist()]
                values = all_options
            else:
                options = [{'label':item,'value':province_code[item]} for 
                           item in vn_df_updated.columns[3:].tolist()]
                values = all_options
            return "Use database by each province",False,options,values,True
        else:
            if w_switch==1:
                options = [{'label':iso_code_dict[item],'value':item} 
                           for item in iso_code_list]
                values = all_options
            else:
                dummy_df = w_df_updated[['iso_code','location']].drop_duplicates()
                
                items = pd.merge(dummy_df,check_df,
                    left_on='iso_code',
                    right_on='ISO3 Alpha-code')[['iso_code','location']].values.tolist()
                
                options = [{'label':item[1],'value':item[0]} for item in 
                           items]
                values = all_options
            return "Use database by each country",False,options,values,True
    elif ctx.triggered_id =="tab2_proc_database_all_division":
        if all_switch:
            return dash.no_update,dash.no_update,dash.no_update,all_options,dash.no_update
        else:
            return dash.no_update,dash.no_update,dash.no_update,[],dash.no_update
    elif ctx.triggered_id=="tab2_proc_database_division_checklist":
        if np.all(np.isin(all_options,checklist)):
            return dash.no_update,dash.no_update,dash.no_update,dash.no_update,True
        else:
            return dash.no_update,dash.no_update,dash.no_update,dash.no_update,False
        
@callback_extension(
    Output({"code":"p1t2","name":"tab2_manual_b_d_rate",
        "type":"special_input","index":0},'value'),
    Output({"code":"p1t2","name":"tab2_manual_b_d_rate",
        "type":"special_input","index":1},'value'),
    Input('tab2_proc_database_all_switch','value'),
    Input('tab2_proc_database','value'),
    Input('tab2_auto_b_d_rate','value'),
    State({'code':'p1t2','name':'birth_data',"data_type":"census"},'data'),
    State({'code':'p1t2','name':'death_data',"data_type":"census"},'data'),
    prevent_initial_call=True
    )
def tab2_change_initial_b_d_rate_manual(trigger1,trigger2,trigger3,birth_df,death_df):
    """
    Update initial manual birth & death rate to match which database is selected.
    """
    if trigger1:
        b_rate = birth_df.loc[birth_df['Location code']==900,
                              'Crude Birth Rate (births per 1,000 population)'
                              ].values[0]
        d_rate = death_df.loc[birth_df['Location code']==900,
                              'Crude Death Rate (deaths per 1,000 population)'
                              ].values[0]
    else:
        if trigger2=="vietnam":
            b_rate = birth_df.loc[birth_df['ISO3 Alpha-code']=="VNM",
                                  'Crude Birth Rate (births per 1,000 population)'
                                  ].values[0]
            d_rate = death_df.loc[birth_df['ISO3 Alpha-code']=="VNM",
                                  'Crude Death Rate (deaths per 1,000 population)'
                                  ].values[0]
        else:
            b_rate = birth_df.loc[birth_df['Location code']==900,
                                  'Crude Birth Rate (births per 1,000 population)'
                                  ].values[0]
            d_rate = death_df.loc[birth_df['Location code']==900,
                                  'Crude Death Rate (deaths per 1,000 population)'
                                  ].values[0]
    birth_rate = round((1+(b_rate/1000))**(1/365)-1,7)
    death_rate = round((1+(d_rate/1000))**(1/365)-1,7)
    return birth_rate,death_rate

@callback(Output('tab2_proc_database_division_button_collapse','is_open'),
          Input('tab2_proc_database_division_switch','value'),
          prevent_initial_call=True)
def tab2_open_section_button(switch):
    '''
    Hide/unhide the sections button.
    '''
    if switch:
        return True
    else:
        return False
    
@callback([Output('tab2_division_option','is_open'),
           Output('tab2_proc_database_division_checklist_alert','is_open')],
          [Input('tab2_proc_database_division_button','n_clicks'),
           Input('tab2_division_option_ok','n_clicks')],
          State('tab2_proc_database_division_checklist','value'),
          prevent_initial_call=True)
def tab2_open_section(button1,button2,option_state):
    '''
    Open and close the data section modal, can't close if no data section was 
    chosen
    '''
    if dash.callback_context.triggered_id =="tab2_proc_database_division_button":
        return [True,dash.no_update]
    else:
        if option_state ==[]:
            return [True,True]
        else:
            return [False,False]
        
@callback(
    Output('tab2_range_b_d_rate_collapse','is_open'),
    Output('tab2_manual_b_d_rate_collapse','is_open'),
    Input('tab2_auto_b_d_rate','value'),
    prevent_initial_call=True
    )
def tab2_switch_auto_OR_manual_b_and_d_rate(trigger):
    """
    Switch between auto generate birth & death rate or not.
    """
    if trigger:
        return True,False
    else:
        return False,True
    
@callback(
    Output({'code':'p1t2','name':'b_d_rate',"data_type":"input_check"},'data'),
    Input('tab2_auto_b_d_rate','value'),
    Input({"code":"p1t2","name":"tab2_manual_b_d_rate","type":"special_input",
           "index":ALL},'value'),
    prevent_initial_call=True
    )
def tab2_check_b_d_rate(switch,values):
    """
    Check input status of death & birth rate manual input.
    """
    if switch:
        return True
    else:
        check_array = np.array(values,dtype='float')
        check_value = np.any(np.isnan(check_array))
        return not check_value
    
@callback(
    Output('tab2_recovery_dist','children'),
    Input({"code":"p1t2","name":"tab2_recovery_dist_week","type":"input"},'value'),
    prevent_initial_call=True
    )
def tab2_redistribution_recovery_ratio(week):
    """
    Auto redistribution the recovery ratio based on the number of weeks
    """
    if week is None:
        return dash.no_update
    else:
        per_dist = np.full(week,1/week,dtype='float')
        per_dist = np.round(per_dist,4)
        per_dist[-1]=1-(per_dist[:-1].sum())
        output = []
        for index,item in np.ndenumerate(per_dist):
            output.append(
                dbc.Input(
                    id={"code":"p1t2","name":"recovery_dist",
                        "type":"special_input","week":index[0]+1},
                    step=0.0001,max=1,min=0,value=item,required=True,
                    type='number',size='sm')
                )
        return output


@callback(
    Output('tab2_death_dist','children'),
    Input({"code":"p1t2","name":"tab2_death_dist_week","type":"input"},'value'),
    prevent_initial_call=True
    )
def tab2_redistribution_death_ratio(week):
    """
    Auto redistribution the death ratio based on the number of weeks
    """
    if week is None:
        return dash.no_update
    else:
        per_dist = np.cumsum(range(1,week+1),dtype='float')
        per_dist = np.flip(per_dist)/per_dist.sum()

        per_dist = np.round(per_dist,4)
        per_dist[-1]=1-per_dist[:-1].sum()
        output = []
        for index,item in np.ndenumerate(per_dist):
            output.append(
                dbc.Input(
                    id={"code":"p1t2","name":"death_dist",
                        "type":"special_input","week":index[0]+1},
                    step=0.0001,max=1,min=0,value=item,required=True,
                    type='number',size='sm')
                )
        return output

@callback(
    Output('tab2_recovery_dist_alert','is_open'),
    Output('tab2_recovery_dist','class_name'),
    Output({'code':'p1t2','name':'recovery_dist',"data_type":"input_check"},'data'),
    Input({"code":"p1t2","name":"recovery_dist",
           "type":"special_input","week":ALL},'value'),
    prevent_initial_call=True
    )
def tab2_redistribution_recov_popup_alert(data):
    """
    Pop-up alert for redistribution of recovery ratio per week.
    """
    data_array = np.array(data,dtype='float')
    sum_perc = np.nansum(data_array)
    if np.isclose(sum_perc,1) and (not np.any(np.isnan(data_array))):
        return False,'m-1 p-1 border border-1 border-secondary',True
    else:
        return True,'m-1 p-1 border border-1 border-danger',False
    
@callback(
    Output('tab2_death_dist_alert','is_open'),
    Output('tab2_death_dist','class_name'),
    Output({'code':'p1t2','name':'death_dist',"data_type":"input_check"},'data'),
    Input({"code":"p1t2","name":"death_dist",
           "type":"special_input","week":ALL},'value'),
    prevent_initial_call=True
    )
def tab2_redistribution_death_popup_alert(data):
    """
    Pop-up alert for redistribution of death ratio per week.
    """
    data_array = np.array(data,dtype='float')
    sum_perc = np.nansum(data_array)
    if np.isclose(sum_perc,1) and (not np.any(np.isnan(data_array))):
        return False,'m-1 p-1 border border-1 border-secondary',True
    else:
        return True,'m-1 p-1 border border-1 border-danger',False

@callback(
    Output('tab2_base_vac_ratio_collapse','is_open'),
    Output('tab2_vaccine_weight_collapse','is_open'),
    Input('tab2_set_vaccine_perc','value'),
    prevent_initial_call=True
    )
def tab2_switch_auto_manual_vac_percent(trigger):
    """
    Switch between manually set vaccine type percent or auto.
    """
    if trigger:
        return True,False
    else:
        return False,True

@callback(
    Output('tab2_base_vac_ratio','children'),
    Input('tab2_add_base_vac_ratio','n_clicks'),
    Input('tab2_remove_base_vac_ratio','n_clicks'),
    State('tab2_base_vac_ratio','children'),
    prevent_initial_call=True
    )
def tab2_add_vaccine_base_ratio(add,remove,state):
    """
    Add or remove vaccine distribition ratio based on usage.
    """
    idx = ((len(state)-2)//4)-1
    if ctx.triggered_id=='tab2_add_base_vac_ratio':
        
        added = [
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_base_vac_ratio",
                       "type":"special_input","index":idx,"params":0},
                   step=0.01,min=0,value=1,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
             dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_base_vac_ratio",
                        "type":"special_input","index":idx,"params":1},
                    step=0.01,min=0,value=1,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                 span=2),
             dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_base_vac_ratio",
                        "type":"special_input","index":idx,"params":2},
                    step=0.01,min=0,value=1,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                 span=2),
             dmc.Col(dmc.DateRangePicker(
                    id={"code":"p1t2","name":"tab2_base_vac_ratio",
                        "type":"special_input","index":idx,"params":3},
                    minDate=datetime.date(2019, 12, 1),
                    maxDate=datetime.datetime.now().date(),
                    allowSingleDateInRange=True,
                    hideOutsideDates=True,
                    value=[datetime.date(2019, 12, 1),
                           datetime.date(2019, 12, 1)],
                    inputFormat="YYYY-MM-DD",
                    amountOfMonths=2,
                    size='xs',
                    labelSeparator=':'
                    ),
                 span=6),
            ]
        state.extend(added)
    elif ctx.triggered_id=='tab2_remove_base_vac_ratio':
        if idx>1:
            state = state[:-4]
        else:
            return dash.no_update
    return state

@callback(
    Output('tab2_base_vac_ratio','class_name'),
    Output({'code':'p1t2','name':'tab2_base_vac_ratio',
            "data_type":"input_check"},'data'),
    Output({"code":"p1t2","name":"tab2_base_vac_ratio",
        "type":"special_input","index":ALL,"params":3},'class_name'),
    Input('tab2_set_vaccine_perc','value'),
    Input({"code":"p1t2","name":"tab2_base_vac_ratio",
        "type":"special_input","index":ALL,"params":ALL},'value'),
    State({"code":"p1t2","name":"tab2_base_vac_ratio",
        "type":"special_input","index":ALL,"params":3},'value'),
    prevent_initial_call=True
    )
def tab2_vaccine_ratio_popup_alert(switch,data,date_state):
    """
    Pop-up alert for vaccine distribution ratio based on usage.
    """
    if switch:
        data_array = np.array(data,dtype='O')
        count_nan = data_array[data_array==None].size
        date_array = np.array(date_state,dtype='O')
        border_array = ["border border-3 border-danger" if item is None else None 
                        for item in date_array]
        
        if count_nan==0:
            return 'mx-1 border border-1 border-secondary',True,border_array
        else:
            return 'mx-1 border border-1 border-danger',False,border_array
    else:
        return dash.no_update,True,[dash.no_update for i in range(len(date_state))]

@callback(
    Output('tab2_w_vaccine_weight','children'),
    Input('tab2_w_add_vaccine_name','n_clicks'),
    Input('tab2_w_remove_vaccine_name','n_clicks'),
    State('tab2_w_vaccine_weight','children'),
    prevent_initial_call=True
    )
def tab2_w_add_vaccine_weight(add,remove,state):
    """
    Add or remove new vaccine weight
    """
    idx = (len(state)//3)-1
    if ctx.triggered_id=='tab2_w_add_vaccine_name':
        
        added = [
            dmc.Col(dbc.Input(
                    value=f'New vaccine No.{idx+1}',
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                       "type":"label_name","index":idx},
                   required=True,
                   type='text',size='sm',style={'font-size':'0.7em'}),
                span=6),
             dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"special_input","index":idx,"params":0},
                    step=1,min=0,max=2,value=0,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                 span=2),
             dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_w_vaccine_weight",
                        "type":"special_input","index":idx,"params":1},
                    step=0.01,min=0,value=0,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                 span=2),
            ]
        state.extend(added)
    elif ctx.triggered_id=='tab2_w_remove_vaccine_name':
        if idx>9:
            state = state[:-3]
        else:
            return dash.no_update
    return state

@callback(
    Output('tab2_w_vaccine_weight','class_name'),
    Output({'code':'p1t2','name':'tab2_w_vaccine_weight',
            "data_type":"input_check"},'data'),
    Input('tab2_set_vaccine_perc','value'),
    Input({"code":"p1t2","name":"tab2_w_vaccine_weight",
        "type":"label_name","index":ALL},'value'),
    Input({"code":"p1t2","name":"tab2_w_vaccine_weight",
        "type":"special_input","index":ALL,"params":0},'value'),
    Input({"code":"p1t2","name":"tab2_w_vaccine_weight",
        "type":"special_input","index":ALL,"params":1},'value'),
    prevent_initial_call=True
    )
def tab2_w_vaccine_weight_popup_alert(switch,input_names,param1,param2):
    """
    Pop-up alert for each vaccine weight setting.
    """
    if not switch:
        data = []
        data.extend(input_names[9:])
        data.extend(param1)
        data.extend(param2)
        data_array = np.array(data,dtype='O')
        data_array[data_array=='']=None
        count_nan = data_array[data_array==None].size
        
        if count_nan==0:
            return 'mx-1 border border-1 border-secondary',True
        else:
            return 'mx-1 border border-1 border-danger',False
    else:
        return dash.no_update,True

@callback(
    Output('tab2_vn_vaccine_weight','children'),
    Input('tab2_vn_add_vaccine_name','n_clicks'),
    Input('tab2_vn_remove_vaccine_name','n_clicks'),
    State('tab2_vn_vaccine_weight','children'),
    prevent_initial_call=True
    )
def tab2_add_vn_vaccine_weight(add,remove,state):
    """
    Add or remove new vaccine weight
    """
    idx = (len(state)//3)-1
    if ctx.triggered_id=='tab2_vn_add_vaccine_name':
        
        added = [
            dmc.Col(dbc.Input(
                    value=f'New vaccine No.{idx+1}',
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                       "type":"label_name","index":idx},
                   required=True,
                   type='text',size='sm',style={'font-size':'0.7em'}),
                span=6),
             dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                        "type":"special_input","index":idx,"params":0},
                    step=1,min=0,max=2,value=0,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                 span=2),
             dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vn_vaccine_weight",
                        "type":"special_input","index":idx,"params":1},
                    step=0.01,min=0,value=0,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                 span=2),
            ]
        state.extend(added)
    elif ctx.triggered_id=='tab2_vn_remove_vaccine_name':
        if idx>7:
            state = state[:-3]
        else:
            return dash.no_update
    return state

@callback(
    Output('tab2_vn_vaccine_weight','class_name'),
    Output({'code':'p1t2','name':'tab2_vn_vaccine_weight',
            "data_type":"input_check"},'data'),
    Input('tab2_set_vaccine_perc','value'),
    Input({"code":"p1t2","name":"tab2_vn_vaccine_weight",
        "type":"label_name","index":ALL},'value'),
    Input({"code":"p1t2","name":"tab2_vn_vaccine_weight",
        "type":"special_input","index":ALL,"params":0},'value'),
    Input({"code":"p1t2","name":"tab2_vn_vaccine_weight",
        "type":"special_input","index":ALL,"params":1},'value'),
    prevent_initial_call=True
    )
def tab2_vn_vaccine_weight_popup_alert(switch,input_names,param1,param2):
    """
    Pop-up alert for each vaccine weight setting.
    """
    if not switch:
        data = []
        data.extend(input_names[7:])
        data.extend(param1)
        data.extend(param2)
        data_array = np.array(data,dtype='O')
        data_array[data_array=='']=None
        count_nan = data_array[data_array==None].size
        
        if count_nan==0:
            return 'mx-1 border border-1 border-secondary',True
        else:
            return 'mx-1 border border-1 border-danger',False
    else:
        return dash.no_update,True

@callback(
    Output('tab2_vaccine_target_collapse','is_open'),
    Input('tab2_set_vaccine_target','value'),
    prevent_initial_call=True
    )
def tab2_on_off_vaccination_target(trigger):
    """
    Switch between manually set vaccination priority or auto.
    """
    if trigger:
        return True
    else:
        return False
    
@callback(
    Output('tab2_vaccine_target','children'),
    Input('tab2_add_vaccine_target','n_clicks'),
    Input('tab2_remove_vaccine_target','n_clicks'),
    State('tab2_vaccine_target','children'),
    prevent_initial_call=True
    )
def tab2_add_vaccination_priority(add,remove,state):
    """
    Add or remove vaccine distribition ratio based on usage.
    """
    idx = ((len(state)-2)//4)-1
    if ctx.triggered_id=='tab2_add_vaccine_target':
        
        added = [
            dmc.Col(dbc.Input(
                   id={"code":"p1t2","name":"tab2_vaccine_target",
                       "type":"special_input","index":idx,"params":0},
                   step=0.01,min=0,value=1,required=True,
                   type='number',size='sm',style={'font-size':'0.7em'}),
                span=2),
             dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_target",
                        "type":"special_input","index":idx,"params":1},
                    step=0.01,min=0,value=1,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                 span=2),
             dmc.Col(dbc.Input(
                    id={"code":"p1t2","name":"tab2_vaccine_target",
                        "type":"special_input","index":idx,"params":2},
                    step=0.01,min=0,value=1,required=True,
                    type='number',size='sm',style={'font-size':'0.7em'}),
                 span=2),
             dmc.Col(dmc.DateRangePicker(
                    id={"code":"p1t2","name":"tab2_vaccine_target",
                        "type":"special_input","index":idx,"params":3},
                    minDate=datetime.date(2019, 12, 1),
                    maxDate=datetime.datetime.now().date(),
                    allowSingleDateInRange=True,
                    hideOutsideDates=True,
                    value=[datetime.date(2019, 12, 1),
                           datetime.date(2019, 12, 1)],
                    inputFormat="YYYY-MM-DD",
                    amountOfMonths=2,
                    size='xs',
                    labelSeparator=':'
                    ),
                 span=6),
            ]
        state.extend(added)
    elif ctx.triggered_id=='tab2_remove_vaccine_target':
        if idx>1:
            state = state[:-4]
        else:
            return dash.no_update
    return state

@callback(
    Output('tab2_vaccine_target','class_name'),
    Output({'code':'p1t2','name':'tab2_vaccine_target',
            "data_type":"input_check"},'data'),
    Output({"code":"p1t2","name":"tab2_vaccine_target",
        "type":"special_input","index":ALL,"params":3},'class_name'),
    Input('tab2_set_vaccine_target','value'),
    Input({"code":"p1t2","name":"tab2_vaccine_target",
        "type":"special_input","index":ALL,"params":ALL},'value'),
    State({"code":"p1t2","name":"tab2_vaccine_target",
        "type":"special_input","index":ALL,"params":3},'value'),
    prevent_initial_call=True
    )
def tab2_vaccination_priority_popup_alert(switch,data,date_state):
    """
    Pop-up alert for vaccination priority based on usage.
    """
    if switch:
        data_array = np.array(data,dtype='O')
        count_nan = data_array[data_array==None].size
        date_array = np.array(date_state,dtype='O')
        border_array = ["border border-3 border-danger" if item is None else None 
                        for item in date_array]
        
        if count_nan==0:
            return 'mx-1 border border-1 border-secondary',True,border_array
        else:
            return 'mx-1 border border-1 border-danger',False,border_array
    else:
        return dash.no_update,True,[dash.no_update for i in range(len(date_state))]

@callback([Output('tab2_empty_f_method','options'),
           Output('tab2_empty_f_method','value')],
          Input('tab2_empty_f_weight','value'),
          State('tab2_empty_f_method','options'),
          prevent_initial_call=True)
def tab2_optional_empty_fill(input_weight,options):
    """
    Connect Input (id=tab2_empty_f_weight) value with custom option in 
    RadioItem (id=tab2_empty_f_method)
    """
    options[-1]['value']=input_weight
    return [options,input_weight]

@callback([Output('tab2_empty_f_alert','is_open'),
           Output('tab2_empty_f_weight','invalid'),
           Output({'code':'p1t2','name':'tab2_empty_f_weight',
                   "data_type":"input_check"},'data')],
          Input('tab2_empty_f_method','value'),
          State('tab2_empty_f_weight','value'),
          prevent_initial_call=True)
def tab2_alert_empty_fill(chosen_option,optional_weight):
    """
    Color input red and raise warning if 'custom weight' is selected and left
    empty.
    """
    if (optional_weight is None) and chosen_option==None:
        return [True,True,False]
    else:
        return [False,False,True]

##input parameters checking----------------------------------------------------

@callback(
    Output({'code':'p1t2','name':'normal_input',"data_type":"input_check"},'data'),
    Input({"code":"p1t2","name":ALL,"type":"input"},'value'),
    prevent_initial_call=True
    )
def tab2_check_normal_input(values):
    """
    Check input status normal inputs.
    """
    check_array = np.array(values,dtype='float')
    check_value = np.any(np.isnan(check_array))
    return not check_value

@callback(
    Output('tab2_inputs_alert','is_open'),
    Input({'code':'p1t2','name':ALL,"data_type":"input_check"},'data'),
    prevent_initial_call=True
    )
def tab2_check_input_warning(values):
    """
    Check all dcc.Store of input status to print fill in requirement.
    """
    check_value = np.all(values)
    return not check_value
##interal data process--------------------------------------------------------

@callback(
    Output({'code':'p1t2','name':'tab2_w_vac_weight',"data_type":"internal"},'data'),
    Input('tab2_w_add_vaccine_name','n_clicks'),
    Input('tab2_w_remove_vaccine_name','n_clicks'),
    #get vaccine names
    Input({"code":"p1t2","name":"tab2_w_vaccine_weight",
        "type":"label_name","index":ALL},'children'),
    Input({"code":"p1t2","name":"tab2_w_vaccine_weight",
        "type":"label_name","index":ALL},'value'),
    #get vaccine group, weight
    Input({"code":"p1t2","name":"tab2_w_vaccine_weight",
        "type":"special_input","index":ALL,"params":0},'value'),
    Input({"code":"p1t2","name":"tab2_w_vaccine_weight",
        "type":"special_input","index":ALL,"params":1},'value'),
    )
def tab2_store_w_vac_weight(trigger1,trigger2,
                            names,input_names,param1,param2):
    """
    Store vaccine weight to a dict.
    """
    names = names[:9]
    input_names = input_names[9:]
    names.extend(input_names)
    output = {item[0]:[item[1],item[2]] for item in zip(names,param1,param2)}
    return output

@callback(
    Output({'code':'p1t2','name':'tab2_vn_vac_weight',"data_type":"internal"},'data'),
    Input('tab2_vn_add_vaccine_name','n_clicks'),
    Input('tab2_vn_remove_vaccine_name','n_clicks'),
    #get vaccine names
    Input({"code":"p1t2","name":"tab2_vn_vaccine_weight",
        "type":"label_name","index":ALL},'children'),
    Input({"code":"p1t2","name":"tab2_vn_vaccine_weight",
        "type":"label_name","index":ALL},'value'),
    #get vaccine group, weight
    Input({"code":"p1t2","name":"tab2_vn_vaccine_weight",
        "type":"special_input","index":ALL,"params":0},'value'),
    Input({"code":"p1t2","name":"tab2_vn_vaccine_weight",
        "type":"special_input","index":ALL,"params":1},'value'),
    )
def tab2_store_vn_vac_weight(trigger1,trigger2,
                             names,input_names,param1,param2):
    """
    Store vaccine weight to a dict.
    """
    names = names[:7]
    input_names = input_names[7:]
    names.extend(input_names)
    output = {item[0]:[item[1],item[2]] for item in zip(names,param1,param2)}
    return output

##plot interaction-------------------------------------------------------------

@callback_extension(
    [Output('tab2_figure_output','children'),
    Output('tab2_figure_add_dropdown','children'),
    Output('tab2_figure_plot_trigger','data')],
    Input('tab2_figure_dropdown_trigger','data'), #trigger
    Input('tab2_plot_tabs','active_tab'),
    Input({'tab2_figure_tabs':ALL,'type':'select_data',"data_type":ALL},'value'),
    
    #server_files
    State({'code':'p1t2','name':'vn_processed_data',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'vn_processed_data_by_province',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data_by_province',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_processed_data',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_processed_data_by_country',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_modeling_data',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_modeling_data_by_country',
           "data_type":"server"},'data'),
    #modified_files
    State({'code':'p1t2','name':'vn_processed_data_modified',
           "data_type":"update"},'data'),
    State({'code':'p1t2','name':'vn_processed_data_by_province_modified',
                  "data_type":"update"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data_modified',
           "data_type":"update"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data_by_province_modified',
                  "data_type":"update"},'data'),
    State({'code':'p1t2','name':'w_processed_data_modified',
           "data_type":"update"},'data'),
    State({'code':'p1t2','name':'w_processed_data_by_country_modified',
                  "data_type":"update"},'data'),
    State({'code':'p1t2','name':'w_modeling_data_modified',
           "data_type":"update"},'data'),
    State({'code':'p1t2','name':'w_modeling_data_by_country_modified',
                  "data_type":"update"},'data'),
    
    State('tab2_figure_plot_trigger','data'),
    prevent_initial_call=True
    )
def tab2_figure_option(initial_trigger,tab_click,data_click,
                       s1,s2,s3,s4,s5,s6,s7,s8,
                       u1,u2,u3,u4,u5,u6,u7,u8,
                       trigger_count):
    """
    Set the dropdown options based on which database was selected.
    
    The id of the select component should be a special dict which allows
    access to format information, as follows:
        id={'code': unique code for each page,
            'tab1_figure_tabs': which tab this belong to,
            "parameters": index of parameter,
            'extra_arg':json.dumps({
                'target':list of targeted element (dataframe| plot),
                'plot_arg':plot attribute which use this dropdown,
                'layout_arg':layout attribute which use this dropdown,
                'code_obj': a code string representative for dataframe process,
                'obj_type': expression or statement,
                'format': the format for code_obj
                })
            }
    The html.Div component which wrap the select component should have a
    special id as follow:
        id={
            'code':unique code for each page,
            'tab1_figure_tabs':which tab this belong to,
            "plot_type":type of plot from plotly.express,
            'xaxis_type': supported data type (number,time),'other' for 
            non-supported type
            'preset':default setting of dataframe and plot {"dataframe",
                                                            "plot",
                                                            "layout"}
            'grid':default setting for grid usage {"grid_arg","col_arg"}
            }
    
    """
    idx = int(re.search(r"(?<=-).*",tab_click)[0])
    database_name = None
    for item in ctx.inputs_list[2]:
        if item['id']['tab2_figure_tabs']==f't{idx+1}':
            database_name = item['value']
            break
    wrapper = html.Div(id={'type':'figure_wrapper',
                           'tab2_figure_tabs':f't{idx+1}'})
    if database_name in ["vn_processed_data","vn_processed_data_modified"]:
        if database_name=="vn_processed_data":
            df = s1
            tab = 't1'
        else:
            df = u1
            tab = 't2'
        output = html.Div(
            id={'code':'p1t2','tab2_figure_tabs':f'{tab}',
                "plot_type":json.dumps(["line","line","line"]),
                'xaxis_type':json.dumps([["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"]]),
                'preset':json.dumps([
                    {'dataframe':None,
                     'plot':{'x':'date','y':['case_avg','death_avg','recovery_case']},
                     'layout':{'layout_title_text':"Time sery of average data",
                               'layout_title_x':0.5}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['death_avg','origin_case','death_percent']},
                     'layout':{'layout_title_text':"Time sery of fatality",
                               'layout_title_x':0.5}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_vaccinations_smoothed',
                                             'curr_full_vaccinated',
                                             'new_full_vaccinated',
                                             'new_boost_req']},
                     'layout':{'layout_title_text':"Time sery of vaccination data",
                               'layout_title_x':0.5}}
                                     ]),
                'grid':json.dumps({'grid_arg':None,
                                   'col_arg':None})
                }
        ),
    elif database_name in ['vn_processed_data_by_province',
                           'vn_processed_data_by_province_modified']:
        if database_name=="vn_processed_data_by_province":
            df = s2
            tab = 't1'
        else:
            df = u2
            tab = 't2'
        output = html.Div(utility_func.add_row_choices(
            ['Province:'], 
            [[{"label":item[1],"value":item[0]} 
              for item in df.iloc[:,[1,2]].drop_duplicates().values]
             ], 
            [df.iloc[:,[1,2]].drop_duplicates().values[0,0]], 
            [{'code':'p1t2','tab2_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                'plot_arg':None,'layout_arg':None,
                'code_obj':'{}.loc[{}["id"]=={}]',
                'obj_type':'expression','format':['df','df','variable']})},
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t2','tab2_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["line","line","line"]),
            'xaxis_type':json.dumps([["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"]]),
            'preset':json.dumps([
                {'dataframe':None,
                 'plot':{'x':'date','y':['case_avg','death_avg','recovery_case']},
                 'layout':{'layout_title_text':"Time sery of average data",
                           'layout_title_x':0.5}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['death_avg','origin_case','death_percent']},
                 'layout':{'layout_title_text':"Time sery of fatality",
                           'layout_title_x':0.5}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['new_vaccinations_smoothed',
                                         'curr_full_vaccinated',
                                         'new_full_vaccinated',
                                         'new_boost_req']},
                 'layout':{'layout_title_text':"Time sery of vaccination data",
                           'layout_title_x':0.5}}
                                 ]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        ),
    elif database_name in ['vn_modeling_data','vn_modeling_data_modified']:
        if database_name=="vn_modeling_data":
            df = s3
            tab = 't1'
        else:
            df = u3
            tab = 't2'
        output = html.Div(
            id={'code':'p1t2','tab2_figure_tabs':f'{tab}',
                "plot_type":json.dumps(["line","line","line","line","line","line",
                                        "line","line","line","line"]),
                'xaxis_type':json.dumps([["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"]]),
                'preset':json.dumps([
                    {'dataframe':None,
                     'plot':{'x':'date','y':['S0','V0','I0','D0']},
                     'layout':{'layout_title_text':"Time sery of initial SVID state",
                               'layout_title_x':0.5}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_case_non_vaccinated','daily_case_vaccinated']},
                     'layout':{'layout_title_text':"Time sery of estimated infected case separation",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['beta','beta_v']},
                     'layout':{'layout_title_text':"Time sery of beta & beta_v",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':'death_avg'},
                     'layout':{'layout_title_text':"Time sery of average death case",
                               'layout_title_x':0.5,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':'gamma'},
                     'layout':{'layout_title_text':"Time sery of gamma",
                               'layout_title_x':0.5,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':'recovery_case'},
                     'layout':{'layout_title_text':"Time sery of recovery case",
                               'layout_title_x':0.5,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':'theta'},
                     'layout':{'layout_title_text':"Time sery of theta",
                               'layout_title_x':0.5,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_full_vaccinated','new_boost_req']},
                     'layout':{'layout_title_text':"Time sery of vaccination",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['alpha','alpha0']},
                     'layout':{'layout_title_text':"Time sery of alpha & alpha0",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_R0','avg_R0']},
                     'layout':{'layout_title_text':"Time sery of reproduction rate",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                                     ]),
                'grid':json.dumps({'grid_arg':{'gutter':0},
                                   'col_arg':[{'span':12},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':12}]
                                   })
                }
        ),
    elif database_name in ['vn_modeling_data_by_province',
                           'vn_modeling_data_by_province_modified']:
        if database_name=="vn_modeling_data_by_province":
            df = s4
            tab = 't1'
        else:
            df = u4
            tab = 't2'
        output = html.Div(utility_func.add_row_choices(
            ['Province:'], 
            [[{"label":item[1],"value":item[0]} 
              for item in df.iloc[:,[1,2]].drop_duplicates().values]
             ], 
            [df.iloc[:,[1,2]].drop_duplicates().values[0,0]], 
            [{'code':'p1t2','tab2_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                'plot_arg':None,'layout_arg':None,
                'code_obj':'{}.loc[{}["id"]=={}]',
                'obj_type':'expression','format':['df','df','variable']})},
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t2','tab2_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["line","line","line","line","line","line",
                                    "line","line","line","line"]),
            'xaxis_type':json.dumps([["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],["date","%Y-%m-%d"]]),
            'preset':json.dumps([
                {'dataframe':None,
                 'plot':{'x':'date','y':['S0','V0','I0','D0']},
                 'layout':{'layout_title_text':"Time sery of initial SVID state",
                           'layout_title_x':0.5}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['daily_case_non_vaccinated','daily_case_vaccinated']},
                 'layout':{'layout_title_text':"Time sery of estimated infected case separation",
                           'layout_title_x':0.5,'layout_legend_y':-0.5,
                           'layout_legend_x':0,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['beta','beta_v']},
                 'layout':{'layout_title_text':"Time sery of beta & beta_v",
                           'layout_title_x':0.5,'layout_legend_y':-0.5,
                           'layout_legend_x':0,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':'death_avg'},
                 'layout':{'layout_title_text':"Time sery of average death case",
                           'layout_title_x':0.5,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':'gamma'},
                 'layout':{'layout_title_text':"Time sery of gamma",
                           'layout_title_x':0.5,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':'recovery_case'},
                 'layout':{'layout_title_text':"Time sery of recovery case",
                           'layout_title_x':0.5,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':'theta'},
                 'layout':{'layout_title_text':"Time sery of theta",
                           'layout_title_x':0.5,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['new_full_vaccinated','new_boost_req']},
                 'layout':{'layout_title_text':"Time sery of vaccination",
                           'layout_title_x':0.5,'layout_legend_y':-0.5,
                           'layout_legend_x':0,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['alpha','alpha0']},
                 'layout':{'layout_title_text':"Time sery of alpha & alpha0",
                           'layout_title_x':0.5,'layout_legend_y':-0.5,
                           'layout_legend_x':0,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['daily_R0','avg_R0']},
                 'layout':{'layout_title_text':"Time sery of reproduction rate",
                           'layout_title_x':0.5,'layout_legend_y':-0.5,
                           'layout_legend_x':0,'layout_margin_r':10}},
                                 ]),
            'grid':json.dumps({'grid_arg':{'gutter':0},
                               'col_arg':[{'span':12},{'span':6},{'span':6},
                                          {'span':6},{'span':6},{'span':6},
                                          {'span':6},{'span':6},{'span':6},
                                          {'span':12}]
                               })
            }
        ),    
    elif database_name in ['w_processed_data','w_processed_data_modified']:
        if database_name=="w_processed_data":
            df = s5
            tab = 't1'
        else:
            df = u5
            tab = 't2'
        output = html.Div(
            id={'code':'p1t2','tab2_figure_tabs':f'{tab}',
                "plot_type":json.dumps(["line","line","line"]),
                'xaxis_type':json.dumps([["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"]]),
                'preset':json.dumps([
                    {'dataframe':None,
                     'plot':{'x':'date','y':['case_avg','death_avg','recovery_case']},
                     'layout':{'layout_title_text':"Time sery of average data",
                               'layout_title_x':0.5}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['death_avg','origin_case','death_percent']},
                     'layout':{'layout_title_text':"Time sery of fatality",
                               'layout_title_x':0.5}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_vaccinations_smoothed',
                                             'curr_full_vaccinated',
                                             'new_full_vaccinated',
                                             'new_boost_req']},
                     'layout':{'layout_title_text':"Time sery of vaccination data",
                               'layout_title_x':0.5}}
                                     ]),
                'grid':json.dumps({'grid_arg':None,
                                   'col_arg':None})
                }
        ),
    elif database_name in ['w_processed_data_by_country',
                           'w_processed_data_by_country_modified']:
        if database_name=="w_processed_data_by_country":
            df = s6
            tab = 't1'
        else:
            df = u6
            tab = 't2'
        output = html.Div(utility_func.add_row_choices(
            ['Country & Area:'], 
            [[{"label":item[1],"value":item[0]} 
              for item in df.iloc[:,[1,3]].drop_duplicates().values]
             ], 
            [df.iloc[:,[1,3]].drop_duplicates().values[0,0]], 
            [{'code':'p1t2','tab2_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                'plot_arg':None,'layout_arg':None,
                'code_obj':'{}.loc[{}["iso_code"]=={}]',
                'obj_type':'expression','format':['df','df','variable']})},
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t2','tab2_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["line","line","line"]),
            'xaxis_type':json.dumps([["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"]]),
            'preset':json.dumps([
                {'dataframe':None,
                 'plot':{'x':'date','y':['case_avg','death_avg','recovery_case']},
                 'layout':{'layout_title_text':"Time sery of average data",
                           'layout_title_x':0.5}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['death_avg','origin_case','death_percent']},
                 'layout':{'layout_title_text':"Time sery of fatality",
                           'layout_title_x':0.5}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['new_vaccinations_smoothed',
                                         'curr_full_vaccinated',
                                         'new_full_vaccinated',
                                         'new_boost_req']},
                 'layout':{'layout_title_text':"Time sery of vaccination data",
                           'layout_title_x':0.5}}
                                 ]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        ),
    elif database_name in ['w_modeling_data','w_modeling_data_modified']:
        if database_name=="w_modeling_data":
            df = s7
            tab = 't1'
        else:
            df = u7
            tab = 't2'
        output = html.Div(
            id={'code':'p1t2','tab2_figure_tabs':f'{tab}',
                "plot_type":json.dumps(["line","line","line","line","line","line",
                                        "line","line","line","line"]),
                'xaxis_type':json.dumps([["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"]]),
                'preset':json.dumps([
                    {'dataframe':None,
                     'plot':{'x':'date','y':['S0','V0','I0','D0']},
                     'layout':{'layout_title_text':"Time sery of initial SVID state",
                               'layout_title_x':0.5}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_case_non_vaccinated','daily_case_vaccinated']},
                     'layout':{'layout_title_text':"Time sery of estimated infected case separation",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['beta','beta_v']},
                     'layout':{'layout_title_text':"Time sery of beta & beta_v",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':'death_avg'},
                     'layout':{'layout_title_text':"Time sery of average death case",
                               'layout_title_x':0.5,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':'gamma'},
                     'layout':{'layout_title_text':"Time sery of gamma",
                               'layout_title_x':0.5,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':'recovery_case'},
                     'layout':{'layout_title_text':"Time sery of recovery case",
                               'layout_title_x':0.5,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':'theta'},
                     'layout':{'layout_title_text':"Time sery of theta",
                               'layout_title_x':0.5,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_full_vaccinated','new_boost_req']},
                     'layout':{'layout_title_text':"Time sery of vaccination",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['alpha','alpha0']},
                     'layout':{'layout_title_text':"Time sery of alpha & alpha0",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_R0','avg_R0']},
                     'layout':{'layout_title_text':"Time sery of reproduction rate",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                                     ]),
                'grid':json.dumps({'grid_arg':{'gutter':0},
                                   'col_arg':[{'span':12},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':12}]
                                   })
                }
        ),
    elif database_name in ['w_modeling_data_by_country',
                           'w_modeling_data_by_country_modified']:
        if database_name=="w_modeling_data_by_country":
            df = s8
            tab = 't1'
        else:
            df = u8
            tab = 't2'
        output = html.Div(utility_func.add_row_choices(
            ['Country & Area:'], 
            [[{"label":item[1],"value":item[0]} 
              for item in df.iloc[:,[1,3]].drop_duplicates().values]
             ], 
            [df.iloc[:,[1,3]].drop_duplicates().values[0,0]], 
            [{'code':'p1t2','tab2_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                'plot_arg':None,'layout_arg':None,
                'code_obj':'{}.loc[{}["iso_code"]=={}]',
                'obj_type':'expression','format':['df','df','variable']})},
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t2','tab2_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["line","line","line","line","line","line",
                                    "line","line","line","line"]),
            'xaxis_type':json.dumps([["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],["date","%Y-%m-%d"]]),
            'preset':json.dumps([
                {'dataframe':None,
                 'plot':{'x':'date','y':['S0','V0','I0','D0']},
                 'layout':{'layout_title_text':"Time sery of initial SVID state",
                           'layout_title_x':0.5}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['daily_case_non_vaccinated','daily_case_vaccinated']},
                 'layout':{'layout_title_text':"Time sery of estimated infected case separation",
                           'layout_title_x':0.5,'layout_legend_y':-0.5,
                           'layout_legend_x':0,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['beta','beta_v']},
                 'layout':{'layout_title_text':"Time sery of beta & beta_v",
                           'layout_title_x':0.5,'layout_legend_y':-0.5,
                           'layout_legend_x':0,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':'death_avg'},
                 'layout':{'layout_title_text':"Time sery of average death case",
                           'layout_title_x':0.5,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':'gamma'},
                 'layout':{'layout_title_text':"Time sery of gamma",
                           'layout_title_x':0.5,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':'recovery_case'},
                 'layout':{'layout_title_text':"Time sery of recovery case",
                           'layout_title_x':0.5,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':'theta'},
                 'layout':{'layout_title_text':"Time sery of theta",
                           'layout_title_x':0.5,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['new_full_vaccinated','new_boost_req']},
                 'layout':{'layout_title_text':"Time sery of vaccination",
                           'layout_title_x':0.5,'layout_legend_y':-0.5,
                           'layout_legend_x':0,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['alpha','alpha0']},
                 'layout':{'layout_title_text':"Time sery of alpha & alpha0",
                           'layout_title_x':0.5,'layout_legend_y':-0.5,
                           'layout_legend_x':0,'layout_margin_r':10}},
                {'dataframe':None,
                 'plot':{'x':'date','y':['daily_R0','avg_R0']},
                 'layout':{'layout_title_text':"Time sery of reproduction rate",
                           'layout_title_x':0.5,'layout_legend_y':-0.5,
                           'layout_legend_x':0,'layout_margin_r':10}},
                                 ]),
            'grid':json.dumps({'grid_arg':{'gutter':0},
                               'col_arg':[{'span':12},{'span':6},{'span':6},
                                          {'span':6},{'span':6},{'span':6},
                                          {'span':6},{'span':6},{'span':6},
                                          {'span':12}]
                               })
            }
        ),    
    else:
        output = []
    return [wrapper,output,trigger_count+1]

@callback(
    Output({"tab2_figure_tabs":'t2'},'children'),
    Input('tab2_processing_alert','children'),
    State({'code':'p1t2','name':ALL,"data_type":"update"},'data'),
    )
def tab2_update_database_dropdown(trigger,processed_databases):
    """
    Update the Modified tab if  new database is processed.
    """
    database_label = {"vn_processed_data_modified":"Processed Vietnam data",
                      "vn_processed_data_by_province_modified":"Processed Vietnam data by province",
                      "vn_modeling_data_modified":"Vietnam modeling input data",
                      "vn_modeling_data_by_province_modified":"Vietnam modeling input data by province",
                      "w_processed_data_modified":"Processed World data",
                      "w_processed_data_by_country_modified":"Processed World data by country",
                      "w_modeling_data_modified":"World modeling input data",
                      "w_modeling_data_by_country_modified":"World modeling input data by country",
                      }
    option_lists={}
    data_list = ctx.states_list[0]
    directory = FileSystemStore(cache_dir="./cache/output_cache",
                                default_timeout=0)
    for item in data_list:
        if directory.get(item.get('value')) is None:
            continue
        else:
            option_lists[item['id']['name']]=database_label[item['id']['name']]
            
    #add dropdown menu
    if option_lists=={}:
        return dash.no_update
    else:
        output=utility_func.add_row_choices(
            ['Database:'], 
            [[{'label':item[1],'value':item[0]} for item in option_lists.items()]
              ], 
            [list(option_lists.keys())[0]], 
            [{"tab2_figure_tabs":'t2','type':'select_data',"data_type":"update"}],
            style={'font-size':'0.9em'},
            class_name='mb-1')[0]
        output = [output,
                  dbc.Row([
                      dbc.Label('Plot points:',id='tab2_plot_point_label2',
                                style={'font-size':'0.9em'},width='auto'),
                      dbc.Col(dbc.Input(id={"tab2_figure_tabs":'t2',
                                            'type':'plot_point'},
                                        type='number',size='sm',
                                        min=2,step=1,value=240,required=True,
                                        className="mt-1 mb-1"),
                              width=1),
                      dbc.Tooltip(dcc.Markdown(
                          '''The number of data points from database which will
                          be plotted.''',
                          style={'white-space':'normal',
                                 'text-align':'left'}),
                              target='tab2_plot_point_label2',
                              delay={'show': 1000}),
                      dbc.Col(dbc.Switch(id={"tab2_figure_tabs":'t2',
                                             'type':'plot_position'},
                                 label_id="tab2_plot_position_label2",
                                 label="Plot from latest day",
                                 value=True,
                                 style = {'font-size':'0.8em'},
                                 className="mb-0 mt-2"),
                              width='auto'),
                      dbc.Tooltip(dcc.Markdown(
                          '''Choose to plot from starting date or latest 
                          date.''',
                          style={'white-space':'normal',
                                 'text-align':'left'}),
                              target='tab2_plot_position_label2',
                              delay={'show': 1000}),
                                         
                      ]),
                  
                  ]
        return output

@callback_extension(
    Output({'type':'figure_wrapper','tab2_figure_tabs':MATCH},'children'),
    Input('tab2_figure_plot_trigger','data'), #trigger
    Input({'code':'p1t2','tab2_figure_tabs':MATCH,"parameters":ALL,
            "extra_arg":ALL},'value'),
    Input({"tab2_figure_tabs":MATCH,'type':'plot_point'},'value'),
    Input({"tab2_figure_tabs":MATCH,'type':'plot_position'},'value'),
    State('tab2_plot_tabs','active_tab'),
    State({'tab2_figure_tabs':MATCH,'type':'select_data',"data_type":ALL},'value'),
    State({'code':'p1t2',"tab2_figure_tabs":MATCH,"plot_type":ALL,'xaxis_type':ALL,
           "preset":ALL,"grid":ALL},'id'),
    #server_files
    State({'code':'p1t2','name':'vn_processed_data',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'vn_processed_data_by_province',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data_by_province',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_processed_data',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_processed_data_by_country',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_modeling_data',
           "data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_modeling_data_by_country',
           "data_type":"server"},'data'),
    #modified_files
    State({'code':'p1t2','name':'vn_processed_data_modified',
           "data_type":"update"},'data'),
    State({'code':'p1t2','name':'vn_processed_data_by_province_modified',
                  "data_type":"update"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data_modified',
           "data_type":"update"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data_by_province_modified',
                  "data_type":"update"},'data'),
    State({'code':'p1t2','name':'w_processed_data_modified',
           "data_type":"update"},'data'),
    State({'code':'p1t2','name':'w_processed_data_by_country_modified',
                  "data_type":"update"},'data'),
    State({'code':'p1t2','name':'w_modeling_data_modified',
           "data_type":"update"},'data'),
    State({'code':'p1t2','name':'w_modeling_data_by_country_modified',
                  "data_type":"update"},'data'),
    prevent_initial_call=True
    )
def tab2_plot(trigger,param_value,plot_point,end_point_plot,
              active_tab,database_name,item_id,
              s1,s2,s3,s4,s5,s6,s7,s8,
              u1,u2,u3,u4,u5,u6,u7,u8):
    """
    Plot databases based on input format from 'id' and select components
    """
    if database_name==[]:
        return []
    else:
        database_name = database_name[0]
    extra_input = ctx.inputs_list[1]
    extra_plot_arg = [json.loads(item['id']['extra_arg']) for item in extra_input]
    extra_def_value = [item for item in param_value]
    preset = json.loads(item_id[0]['preset'])
    plot_type = json.loads(item_id[0]['plot_type'])
    xaxis_type = json.loads(item_id[0]['xaxis_type'])
    grid = json.loads(item_id[0]['grid'])
    
    if plot_point is None:
        return dash.no_update
    if database_name=="vn_processed_data":
        if end_point_plot:
            df = s1.tail(plot_point)
        else:
            df = s1.head(plot_point)
    elif database_name=="vn_processed_data_by_province":
        if end_point_plot:
            df = s2.groupby(['id']).tail(plot_point)
        else:
            df = s2.groupby(['id']).head(plot_point)
    elif database_name=="vn_modeling_data":
        if end_point_plot:
            df = s3.tail(plot_point)
        else:
            df = s3.head(plot_point)
    elif database_name=="vn_modeling_data_by_province":
        if end_point_plot:
            df = s4.groupby(['id']).tail(plot_point)
        else:
            df = s4.groupby(['id']).head(plot_point)
    elif database_name=="w_processed_data":
        if end_point_plot:
            df = s5.tail(plot_point)
        else:
            df = s5.head(plot_point)
    elif database_name=="w_processed_data_by_country":
        if end_point_plot:
            df = s6.groupby(['iso_code']).tail(plot_point)
        else:
            df = s6.groupby(['iso_code']).head(plot_point)
    elif database_name=="w_modeling_data":
        if end_point_plot:
            df = s7.tail(plot_point)
        else:
            df = s7.head(plot_point)
    elif database_name=="w_modeling_data_by_country":
        if end_point_plot:
            df = s8.groupby(['iso_code']).tail(plot_point)
        else:
            df = s8.groupby(['iso_code']).head(plot_point)
    elif database_name=="vn_processed_data_modified":
        if end_point_plot:
            df = u1.tail(plot_point)
        else:
            df = u1.head(plot_point)
    elif database_name=="vn_processed_data_by_province_modified":
        if end_point_plot:
            df = u2.groupby(['id']).tail(plot_point)
        else:
            df = u2.groupby(['id']).head(plot_point)
    elif database_name=="vn_modeling_data_modified":
        if end_point_plot:
            df = u3.tail(plot_point)
        else:
            df = u3.head(plot_point)
    elif database_name=="vn_modeling_data_by_province_modified":
        if end_point_plot:
            df = u4.groupby(['id']).tail(plot_point)
        else:
            df = u4.groupby(['id']).head(plot_point)
    elif database_name=="w_processed_data_modified":
        if end_point_plot:
            df = u5.tail(plot_point)
        else:
            df = u5.head(plot_point)
    elif database_name=="w_processed_data_by_country_modified":
        if end_point_plot:
            df = u6.groupby(['iso_code']).tail(plot_point)
        else:
            df = u6.groupby(['iso_code']).head(plot_point)
    elif database_name=="w_modeling_data_modified":
        if end_point_plot:
            df = u7.tail(plot_point)
        else:
            df = u7.head(plot_point)
    elif database_name=="w_modeling_data_by_country_modified":
        if end_point_plot:
            df = u8.groupby(['iso_code']).tail(plot_point)
        else:
            df = u8.groupby(['iso_code']).head(plot_point)
    
    output = []
    plot_output = []
    store_output = []
    for item in enumerate(zip(plot_type,preset,xaxis_type)):
        fig = utility_func.condition_plot(item[1][0],
                                          df,extra_plot_arg,extra_def_value,
                                          item[1][1])
        
        #invoke fig processing when id contain special arguments
        
        plot_handler = []
        plot_data_store = []
        if item[1][1]['layout'] is not None:
            if 'layout_xaxis_rangeslider_visible' in item[1][1]['layout'].keys():
                plot_handler.append('layout_xaxis_rangeslider_visible')
                plot_data_store.extend(['layout','frames'])
        
        if grid['grid_arg'] is None:
            plot_output.append(
                dcc.Graph(figure=fig,
                          id={'code':item_id[0]['code'],
                              'tab2_figure_tabs':item_id[0]['tab2_figure_tabs'],
                              'plot_idx':item[0],
                              'xaxis_type':json.dumps(item[1][2]),
                              'plot_args':None if plot_handler==None 
                              else json.dumps(plot_handler)
                              }))
        else:
            plot_output.append(dmc.Col(children=
                dcc.Graph(figure=fig,
                          id={'code':item_id[0]['code'],
                              'tab2_figure_tabs':item_id[0]['tab2_figure_tabs'],
                              'plot_idx':item[0],
                              'xaxis_type':json.dumps(item[1][2]),
                              'plot_args':None if plot_handler==None 
                              else json.dumps(plot_handler)
                              }),
                **grid['col_arg'][item[0]])
                )
            
        #only run when there are arguments needed to be process
        for attr in plot_data_store:
            store_output.append(
                dcc.Store(data=getattr(fig,attr),
                          id={'code':item_id[0]['code'],
                              'tab2_figure_tabs':item_id[0]['tab2_figure_tabs'],
                              'plot_idx':item[0],
                              'attr_name':attr
                              })
                )
    
    if grid['grid_arg'] is None:
        output.extend(plot_output)
        output.extend(store_output)
    else:
        output.append(dmc.Grid(children=plot_output,
                               **grid['grid_arg']))
        output.extend(store_output)
    return output

##processing data-------------------------------------------------------------
    
@callback(
    output=dict(
        process_result=Output('tab2_processing_alert','children'),
        option_colors=Output({'code':'p1t2','name':ALL,
                              "purpose":"database_check"},'labelClassName'),
        option_disable=Output({'code':'p1t2','name':ALL,
                              "purpose":"database_check"},'options'),
    ),
    inputs = dict(
        initiator=Input('tab2_process_params','data'),
        
        #census input
        birth_uid=State({'code':'p1t2','name':'birth_data',
                          "data_type":"census"},'data'),
        death_uid = State({'code':'p1t2','name':'death_data',
                            "data_type":"census"},'data'),
        population_uid = State({'code':'p1t2','name':'w_population',
                                "data_type":"census"},'data'),
        vn_population_uid = State({'code':'p1t2','name':'vn_population',
                                "data_type":"census"},'data'),
        #data input
        bc_uid = State({'code':'p1t1','name':'w_covid_data',
                        "data_type":"server"},'data'),
        bc_uid_update = State({'code':'p1t1','name':'w_covid_data_latest',
                                "data_type":"update"},'data'),
        w_meta_uid= State({'code':'p1t1','name':'w_meta_database',
                           "data_type":"server"},'data'),
        w_meta_uid_update = State({'code':'p1t1','name':'w_meta_database_latest',
                                   "data_type":"update"},'data'),
        vn_case_uid = State({'code':'p1t1','name':'vn_total_case',
                      "data_type":"server"},'data'),
        vn_case_uid_update = State({'code':'p1t1','name':'vn_total_case_latest',
                      "data_type":"update"},'data'),
        vn_p_case_uid = State({'code':'p1t1','name':'vn_province_case',
                      "data_type":"server"},'data'),
        vn_p_case_uid_update = State({'code':'p1t1','name':'vn_province_case_latest',
                      "data_type":"update"},'data'),
        vn_d_case_uid = State({'code':'p1t1','name':'vn_death_data',
                      "data_type":"server"},'data'),
        vn_d_case_uid_update = State({'code':'p1t1','name':'vn_death_data_latest',
                      "data_type":"update"},'data'),
        vn_vac_vol_uid = State({'code':'p1t1','name':'vn_vaccine_vol',
                      "data_type":"server"},'data'),
        vn_vac_vol_uid_update = State({'code':'p1t1','name':'vn_vaccine_vol_latest',
                      "data_type":"update"},'data'),
        vn_vac_dist_uid = State({'code':'p1t1','name':'vn_vaccine_dist',
                      "data_type":"server"},'data'),
        vn_vac_dist_uid_update = State({'code':'p1t1','name':'vn_vaccine_dist_latest',
                      "data_type":"update"},'data'),
        state_disable = State({'code':'p1t2','name':ALL,
                              "purpose":"database_check"},'options')
    ),
    background=True,
    running=[(Output('tab2_processing_status','is_open'),True,False)],
    progress=[Output('tab2_update_process_collapse','is_open'),
              Output('tab2_update_process','value'),
              Output('tab2_update_process_info','children')],
    progress_default=[False,0,'Processing...'],
    cancel=Input('tab2_model_Cancel_Button1','n_clicks'),
    prevent_initial_call=True)
def tab2_data_processing(progress_status,initiator,
                         birth_uid,death_uid,population_uid,vn_population_uid,
                         bc_uid,bc_uid_update,w_meta_uid,w_meta_uid_update,
                         vn_case_uid,vn_case_uid_update,vn_p_case_uid,
                         vn_p_case_uid_update,vn_d_case_uid,vn_d_case_uid_update,
                         vn_vac_vol_uid,vn_vac_vol_uid_update,vn_vac_dist_uid,
                         vn_vac_dist_uid_update,
                         state_disable
                          ):
    """
    Processing databases using relevant inputs.
    """
    n_outputs = len(ctx.outputs_list[1])
    color_outputs = [dash.no_update for _ in range(n_outputs)]
    disable_outputs = [dash.no_update for _ in range(n_outputs)]
    
    directory = FileSystemStore(cache_dir="./cache/output_cache",
                                default_timeout=0)
    server_directory = FileSystemStore(cache_dir="./cache/server_cache",
                                default_timeout=0)
    
    packed_params = directory.get(initiator)
    
    (output_uid,chosen_all,chosen_database,data_separation,chosen_sections,
     default_iso,w_default,w_meta_default,vn_default,vn_vac_dist_default,
     auto_b_d_rate,manual_b_rate,manual_d_rate)=packed_params['database_params']
    
    (raw_rolling,trim_percent,death_distribution,death_after,death_rolling,
     use_origin,recovery_distribution,recovery_after,raw_method_limit,
     raw_seed)=packed_params['raw_params']
    
    (manual_base_vac,vaccine_info,base_vaccine_percent,target_priority,
     w_vac_weight,w_other_vac_weight,vn_vac_weight,vn_other_vac_weight,
     vac_method_limit,vac_seed)=packed_params['vac_params']
    
    (r_protect_time,s_vac_ratio,avg_infect_t,avg_death_t,avg_recov_t,
     avg_rotate,avg_data,weights,include_zero,r0_p,r0_n,
     r0_trim)=packed_params['model_params']
    
    (max_worker)=packed_params['general_params']
    
    #get data from uid
    birth_df = directory.get(birth_uid)
    death_df = directory.get(death_uid)
    population_df = directory.get(population_uid)
    vn_population_df = directory.get(vn_population_uid)
    if w_meta_default==1:
        vac_meta_df = server_directory.get(w_meta_uid)
    else:
        vac_meta_df = directory.get(w_meta_uid_update)
        
    #iterator
    options_index=[]
    storage_index=[]
    title_str1=[]
    title_str2=[]
    raw_worker_range=[]
    raw_database = []
    birth_input=[]
    death_input=[]
    population_input=[]
    auto_fill_input=[]
    unsucessful_process = 0
    if chosen_database=="vietnam" or chosen_all:
        if (not data_separation) or chosen_all:
            options_index.append(0)
            storage_index.append(0)
            title_str1.append("Vietnam")
            title_str2.append("")
            auto_fill_input.append(None)
            if not auto_b_d_rate:
                birth_input.append(manual_b_rate)
                death_input.append(manual_d_rate)
            else:
                b_rate = birth_df.loc[
                    birth_df['ISO3 Alpha-code']=="VNM",
                    'Crude Birth Rate (births per 1,000 population)'].values[0] 
                d_rate = death_df.loc[
                    death_df['ISO3 Alpha-code']=="VNM",
                    'Crude Death Rate (deaths per 1,000 population)'].values[0] 
                birth_input.append((1+(b_rate/1000))**(1/365)-1 )
                death_input.append((1+(d_rate/1000))**(1/365)-1 )
            population_input.append(
                int(population_df.loc[
                    population_df['ISO3 Alpha-code']=="VNM",
                    'Total Population, as of 1 January'].values[0]))
            raw_worker_range.append(['VNM'])
            if vn_default==1:
                df_vn_case = server_directory.get(vn_case_uid)
                df_vn_death_case = server_directory.get(vn_d_case_uid)
                df_vn_vaccine = server_directory.get(vn_vac_vol_uid)
            else:
                df_vn_case = directory.get(vn_case_uid_update)
                df_vn_death_case = directory.get(vn_d_case_uid_update)
                df_vn_vaccine = directory.get(vn_vac_vol_uid_update)
            df_vn_case = df_vn_case.loc[df_vn_case.code =="local_case"].copy()
            df_vn_case.drop(axis=1,columns=['case_origin','code'],inplace=True)
            df_vn_death_case = df_vn_death_case[["date","VN_death_case"]]
            #merge DataFrame
            vn_input_df = df_vn_case.merge(df_vn_death_case,on="date")
            vn_input_df['new_vaccinations_smoothed']=df_vn_vaccine['new_vaccinations_smoothed']
            vn_input_df.rename(columns={"case":"new_cases",
                                        "VN_death_case":"new_deaths"},
                               inplace=True)
            raw_database.append(vn_input_df)
        if data_separation or chosen_all:
            options_index.append(1)
            storage_index.append(1)
            title_str1.append("Vietnam")
            title_str2.append("by provinces")
            auto_fill_input.append(['id','provinceName'])
            if not auto_b_d_rate:
                birth_input.append(manual_b_rate)
                death_input.append(manual_d_rate)
            else:
                b_rate = birth_df.loc[
                    birth_df['ISO3 Alpha-code']=="VNM",
                    'Crude Birth Rate (births per 1,000 population)'].values[0] 
                d_rate = death_df.loc[
                    death_df['ISO3 Alpha-code']=="VNM",
                    'Crude Death Rate (deaths per 1,000 population)'].values[0] 
                birth_input.append((1+(b_rate/1000))**(1/365)-1 )
                death_input.append((1+(d_rate/1000))**(1/365)-1 )
            if vn_default==1:
                vn_p_df = server_directory.get(vn_p_case_uid)
                vn_p_death_df = server_directory.get(vn_d_case_uid)
                df_vn_vaccine = server_directory.get(vn_vac_vol_uid)
            else:
                vn_p_df = directory.get(vn_p_case_uid_update)
                vn_p_death_df = directory.get(vn_d_case_uid_update)
                df_vn_vaccine = directory.get(vn_vac_vol_uid_update)
            if vn_vac_dist_default==1:
                vn_vac_dist_df = server_directory.get(vn_vac_dist_uid)
            else:
                vn_vac_dist_df = directory.get(vn_vac_dist_uid_update)
            vn_vac_dist_df['distribution_ratio']=(
                vn_vac_dist_df['totalVaccineAllocatedReality']/
                vn_vac_dist_df['totalVaccineAllocatedReality'].sum())
            
            date_serry = pd.Series([vn_p_df.columns[3:].to_list()]*vn_p_df.shape[0])
            case_serry = pd.Series(vn_p_df.iloc[:,3:].to_numpy().tolist())
            vn_p_df = vn_p_df.iloc[:,:3].copy()
            vn_p_df['date']=date_serry
            vn_p_df['new_cases']=case_serry
            vn_p_df = vn_p_df.explode(['date','new_cases']).copy()
            vn_p_df = vn_p_df.astype({'date':str,'new_cases':np.float64})
            vn_p_df.reset_index(drop=True)
            vn_p_df['date']=pd.to_datetime(vn_p_df['date'])
            
            vn_bp_input_df = pd.DataFrame()
            for province in vn_p_death_df.columns[3:]:
            # for province in vn_p_df['provinceName'].unique():
                case_data = vn_p_df.loc[vn_p_df['provinceName']==province]
                death_data = (vn_p_death_df.loc[:,['date',province]]
                              .rename(columns={province:'new_deaths'}))
                
                vac_distributed = df_vn_vaccine.copy()
                vac_distributed['new_vaccinations_smoothed'] =\
                    (vac_distributed['new_vaccinations_smoothed']*(
                        vn_vac_dist_df.loc[vn_vac_dist_df['provinceName']==province,
                                           'distribution_ratio'].values)
                    ).round()

                dummy_df = pd.merge(case_data,death_data,on='date',how='outer')
                dummy_df = pd.merge(dummy_df,vac_distributed,on='date',how='left')
                vn_bp_input_df = pd.concat([vn_bp_input_df,dummy_df],ignore_index=True)
            vn_bp_input_df['date'] = pd.to_datetime(vn_bp_input_df['date'])
            vn_bp_input_df.drop(columns="population",inplace=True)
            vn_bp_input_df = vn_bp_input_df[
                ['date','id','provinceName','new_cases','new_deaths',
                 'new_vaccinations_smoothed']]
            
            population_input.append(
                (vn_population_df[['id','provinceName','population']]
                 ,'id','population')
                )
            if chosen_all:
                raw_range = vn_bp_input_df['id'].unique().tolist()
            else:
                raw_range = chosen_sections
            raw_worker_range.append(raw_range)
            raw_database.append((vn_bp_input_df,'id'))
    if chosen_database=="world" or chosen_all:
        if (not data_separation) or chosen_all: #process total data
            options_index.append(2)
            storage_index.append(4)
            title_str1.append("World")
            title_str2.append("")
            auto_fill_input.append(['iso_code','continent','location'])
            if not auto_b_d_rate:
                birth_input.append(manual_b_rate)
                death_input.append(manual_d_rate)
            else:
                b_rate = birth_df.loc[
                    birth_df['Location code']==900,
                    'Crude Birth Rate (births per 1,000 population)'].values[0] 
                d_rate = death_df.loc[
                    death_df['Location code']==900,
                    'Crude Death Rate (deaths per 1,000 population)'].values[0] 
                birth_input.append((1+(b_rate/1000))**(1/365)-1 )
                death_input.append((1+(d_rate/1000))**(1/365)-1 )
            population_input.append(
                int(population_df.loc[
                    population_df['Location code']==900,
                    'Total Population, as of 1 January'].values[0]))
            raw_worker_range.append(['OWID_WRL'])
            if w_default==1:
                w_input_df = server_directory.get(bc_uid)
            else:
                w_input_df = directory.get(bc_uid_update)
            w_input_df = w_input_df.loc[
                w_input_df['iso_code']=='OWID_WRL'].copy().reset_index(drop=True)
            raw_database.append(w_input_df)
        if data_separation or chosen_all: #process by country
            options_index.append(3)
            storage_index.append(5)
            title_str1.append("World")
            title_str2.append("by countries")
            auto_fill_input.append(['iso_code','continent','location'])
            if not auto_b_d_rate:
                birth_input.append(manual_b_rate)
                death_input.append(manual_d_rate)
            else:
                birth_input.append((birth_df,'ISO3 Alpha-code',
                                    'Crude Birth Rate (births per 1,000 population)'
                                    ))
                death_input.append((death_df,'ISO3 Alpha-code',
                                    'Crude Death Rate (deaths per 1,000 population)'
                                    ))
            population_input.append((population_df,'ISO3 Alpha-code',
                                     'Total Population, as of 1 January'))
            if chosen_all:
                raw_range = np.array(default_iso)[:,0].tolist()
            else:
                raw_range = chosen_sections
            raw_worker_range.append(raw_range)
            if w_default==1:
                bc_input_df = server_directory.get(bc_uid)
            else:
                bc_input_df = directory.get(bc_uid_update)
            bc_input_df = bc_input_df[['date','iso_code','continent','location',
                            'new_cases','new_deaths',
                            'new_vaccinations_smoothed']].copy()
            raw_database.append((bc_input_df,'iso_code'))
    for input_group in zip(options_index,storage_index,title_str1,title_str2,raw_worker_range,
                           raw_database,birth_input,death_input,population_input,
                           auto_fill_input):
        (idx,store_idx,str1,str2,worker_range,raw_input,birth_data,death_data,
         population_data,auto_fill_data)=input_group
        try:
            #the main code goes here
            progress_status((False,0,f'Proccessing {str1} covid database {str2}...'))
            time.sleep(1)
            
            #raw data multiprocess
            
            progress_status((False,0,f'Proccessing {str1} raw data {str2}...'))
            time.sleep(1)
            
            raw_manager = Manager()
            raw_counter= raw_manager.Array('i',[0,0,0,0])
            raw_time_list = raw_manager.list([])
            raw_error_dict = raw_manager.dict()
            raw_shared_output = raw_manager.Namespace()
            raw_shared_output.shared_df = pd.DataFrame()
            raw_lock = raw_manager.Lock()
            
            worker_func = partial(
                raw_wrapper_func,
                shared_count=raw_counter,
                time_list=raw_time_list,
                error_dict=raw_error_dict,
                shared_output=raw_shared_output,
                lock=raw_lock,
                input_data=raw_input,
                birth_data=birth_data,
                death_data=death_data,
                population_data=population_data,
                raw_rolling=raw_rolling,
                trim_percent=trim_percent,
                death_distribution=death_distribution,
                death_after=death_after,
                death_rolling=death_rolling,
                use_origin=use_origin,
                recovery_distribution=recovery_distribution,
                recovery_after=recovery_after,
                raw_method_limit=raw_method_limit,
                raw_seed=raw_seed,
                auto_fill_col=auto_fill_data
                )
            if worker_range not in [['VNM'],['OWID_WRL']]:
                executor= Pool(processes=max_worker)
                pool_result = executor.map_async(worker_func,worker_range)
            else:
                worker_func(worker_range[0])
            max_count = len(worker_range)
            while raw_counter[0]<max_count:
                time.sleep(1)
                n_done = raw_counter[0]
                n_wait = max_count
                total_time = round(sum(raw_time_list),2)
                if n_done!=0:
                    avg_time = round(total_time/n_done,2)
    
                if n_done!=0:
                    remain_section = n_wait - n_done
                    remain_time = round(remain_section*avg_time,2)
                    real_remain = round(remain_time/max_worker,2)
                else:
                    real_remain = "-- "
                progress_status((True,100*n_done/max_count,
                                  (f'Processing {str1} raw data {str2}:'+
                                    f' {real_remain}s')
                                  ))
            #for debug logging if necessary
            raw_error_result = dict(raw_error_dict)
            raw_time = list(raw_time_list)
            
            raw_processed_df = raw_shared_output.shared_df.copy()
            if type(raw_input)==tuple and str1=="World":
                raw_processed_df = raw_processed_df.sort_values(
                    ['iso_code','date']).reset_index(drop=True)
            elif type(raw_input)==tuple and str1=="Vietnam":
                raw_processed_df = raw_processed_df.sort_values(
                    ['provinceName','date']).reset_index(drop=True)
            
            if worker_range not in [['VNM'],['OWID_WRL']]:
                executor.close()
                executor.join()
            raw_manager.shutdown()
            progress_status((False,100,
                              (f'Processed {str1} raw data {str2}')
                              ))
            time.sleep(1)
            
            #vaccination data multiprocess
            progress_status((False,0,f'Proccessing {str1} vaccination data {str2}...'))
            time.sleep(1)
            
            vac_manager = Manager()
            vac_counter= vac_manager.Array('i',[0,0])
            vac_time_list = vac_manager.list([])
            vac_error_dict = vac_manager.dict()
            vac_shared_output = vac_manager.Namespace()
            vac_shared_output.shared_df = pd.DataFrame()
            vac_lock = vac_manager.Lock()
            
            if type(raw_input)==tuple: #not total Vietnam|World data
                worker_range = raw_processed_df[raw_input[1]].unique().tolist()
                vac_input = (raw_processed_df,raw_input[1])
                base_vac_input = base_vaccine_percent
                if str1=="World":
                    weight_input = w_vac_weight
                    other_weight_input = w_other_vac_weight
                elif str1=="Vietnam":
                    weight_input = vn_vac_weight
                    other_weight_input = vn_other_vac_weight
                if manual_base_vac:
                    vac_meta_input=None
                else:
                    if str1=="World":
                        vac_meta_input = vac_meta_df
                    elif str1=="Vietnam":
                        vac_meta_input = (vac_meta_df,'VNM')
            elif str1=="World":
                worker_range= ['OWID_WRL']
                vac_input = raw_processed_df
                vac_meta_input = None
                weight_input = w_vac_weight
                other_weight_input = w_other_vac_weight
                if manual_base_vac:
                    base_vac_input = base_vaccine_percent
                else:
                    base_vac_input = [14.4,135,5] #TODO: ratio might need edit over time
            elif str1=="Vietnam":
                worker_range = ['VNM']
                vac_input = raw_processed_df
                base_vac_input = base_vaccine_percent
                weight_input = vn_vac_weight
                other_weight_input = vn_other_vac_weight
                if manual_base_vac:
                    vac_meta_input=None
                else:
                    vac_meta_input = (vac_meta_df,'VNM')
            worker_func = partial(
                vaccine_wrapper_func,
                shared_count=vac_counter,
                time_list=vac_time_list,
                error_dict=vac_error_dict,
                shared_output=vac_shared_output,
                lock=vac_lock,
                input_data = vac_input,
                vac_meta_data=vac_meta_input,
                vaccine_info=vaccine_info,
                base_vaccine_percent=base_vac_input,
                target_priority=target_priority,
                vac_weight=weight_input,
                other_vaccine_weight=other_weight_input,
                vac_method_limit=vac_method_limit,
                vac_seed=vac_seed)
            if worker_range not in [['VNM'],['OWID_WRL']]:
                executor= Pool(processes=max_worker)
                pool_result = executor.map_async(worker_func,worker_range)
            else:
                worker_func(worker_range[0])
            
            max_count = len(worker_range)
            while vac_counter[0]<max_count:
                time.sleep(1)
                n_done = vac_counter[0]
                n_wait = max_count
                total_time = round(sum(vac_time_list),2)
                if n_done!=0:
                    avg_time = round(total_time/n_done,2)
    
                if n_done!=0:
                    remain_section = n_wait - n_done
                    remain_time = round(remain_section*avg_time,2)
                    real_remain = round(remain_time/max_worker,2)
                else:
                    real_remain = "-- "
                progress_status((True,100*n_done/max_count,
                                  (f'Processing {str1} vaccination data {str2}:'+
                                    f' {real_remain}s')
                                  ))
            vac_processed_df = vac_shared_output.shared_df.copy()
            if type(raw_input)==tuple and str1=="World":
                vac_processed_df = vac_processed_df.sort_values(
                    ['iso_code','date']).reset_index(drop=True)
            elif type(raw_input)==tuple and str1=="Vietnam":
                vac_processed_df = vac_processed_df.sort_values(
                    ['provinceName','date']).reset_index(drop=True)
            directory.set(output_uid[store_idx],vac_processed_df)
            #for debug logging if necessary
            vac_vaccine_error_result = dict(vac_error_dict)
            vac_vaccine_time = list(vac_time_list)
            if worker_range not in [['VNM'],['OWID_WRL']]:
                executor.close()
                executor.join()
            vac_manager.shutdown()
            progress_status((False,100,
                              (f'Processed {str1} vaccination data {str2}')
                              ))
            time.sleep(1)
            
            #modeling data
            progress_status((False,0,f'Proccessing {str1} modeling data {str2}...'))
            time.sleep(1)
            counter=0
            total_s=0
            model_input_df = vac_processed_df.copy()
            model_df = pd.DataFrame()
            
            if type(raw_input)==tuple: #not total Vietnam|World data
                worker_range = vac_processed_df[raw_input[1]].unique()
            elif str1=="World":
                worker_range= np.array(['OWID_WRL'])
            elif str1=="Vietnam":
                worker_range = np.array(['VNM'])
            
            for code in worker_range:
                start = time.time()
                if type(raw_input)==tuple:
                    dummy_df = model_input_df.loc[
                        model_input_df[raw_input[1]]==code].reset_index(drop=True)
                else:
                    dummy_df=model_input_df
                dummy_df.loc[:,['case_avg','death_avg']]=\
                    dummy_df.loc[:,['case_avg','death_avg']].apply(np.round)
                
                dummy_df[['S0','V0','I0','D0','daily_case_non_vaccinated',
                          'daily_case_vaccinated']] =\
                    model_initial(dummy_df[['case_avg','recovery_case',
                                            'death_avg','curr_full_vaccinated',
                                            'current_pol_N']].to_numpy(),
                                  r_protect_time,
                                  float(s_vac_ratio))
                dummy_df[['beta','beta_v','gamma','theta','alpha','alpha0']] =\
                    model_parameter(dummy_df[['S0','V0','I0','death_avg',
                                              'recovery_case','new_full_vaccinated',
                                              'new_boost_req','daily_case_non_vaccinated',
                                              'daily_case_vaccinated',
                                              'current_pol_N']].to_numpy(),
                                    r_protect_time,
                                    avg_infect_t,
                                    avg_death_t,
                                    avg_recov_t,
                                    avg_rotate)
                
                column_list = ['date','S0','V0','I0','D0','beta','beta_v','gamma',
                                'theta','alpha','alpha0','death_avg','recovery_case',
                                'new_full_vaccinated','new_boost_req',
                                'daily_case_non_vaccinated','daily_case_vaccinated',
                                'current_pol_N']
                if type(raw_input)!=tuple and str1=="Vietnam":
                    pass
                else:
                    column_list[1:1]=auto_fill_data
                concat_df = dummy_df[column_list].copy()
                concat_df[['S0','V0','I0','D0']]=concat_df[
                    ['S0','V0','I0','D0']].astype(np.int64)
                
                #get birth rate & death rate
                if type(birth_data) in [int,float,np.float64,np.int64]:
                    birth_rate = float(birth_data)
                else:
                    birth_df = birth_data[0]
                    birth_rate = birth_df.loc[birth_df[birth_data[1]]==code,
                                              birth_data[2]]
                    birth_rate = (1+(float(birth_rate)/1000))**(1/365)-1 
                
                if type(death_data) in [int,float,np.float64,np.int64]:
                    death_rate = float(death_data)
                else:
                    death_df = death_data[0]
                    death_rate = death_df.loc[death_df[death_data[1]]==code,
                                              death_data[2]]
                    death_rate = (1+(float(death_rate)/1000))**(1/365)-1 
                #add daily recruitment rate
                concat_df.insert(concat_df.columns.get_loc("alpha0")+1,
                                 'pi',
                                 ((concat_df[['S0','V0','I0','D0']].sum(axis=1))\
                                  *birth_rate).astype(np.int64))
                
                #fill empty values of each columns
                concat_df = fill_empty_param(input_df=concat_df,
                                            input_col=['beta','beta_v','gamma',
                                                      'theta','alpha','alpha0'],
                                            avg_data=avg_data,
                                            weights=weights,
                                            include_zero=include_zero)
                
                #fill remaining SVID model parameter with 0
                concat_df.iloc[:,concat_df.columns.get_loc('beta'):
                               (concat_df.columns.get_loc('pi')+1)]=\
                    concat_df.iloc[:,concat_df.columns.get_loc('beta'):
                                   (concat_df.columns.get_loc('pi')+1)].fillna(0)
                
                #calculate reproduction number
                concat_df[['daily_R0','avg_R0']]=reproduction_number(
                    input_df=concat_df,
                    input_col=['beta','beta_v','gamma',
                              'theta','alpha','alpha0','pi'],
                    p=float(r0_p),
                    mu=death_rate,
                    n=r0_n,
                    outlier_trim=r0_trim)
                
                model_df = pd.concat([model_df,concat_df],ignore_index=True)
                
                end=time.time()
                counter += 1
                total_s = total_s + end - start
                avg = total_s/counter
                
                remain_time = round(avg*(worker_range.shape[0]-counter),2)
                
                progress_status((True,100*counter/len(worker_range),
                                  (f'Processing {str1} modeling data {str2}:'+
                                    f' {remain_time}s')
                                  ))
            directory.set(output_uid[store_idx+2],model_df)
            progress_status((False,100,
                              (f'Processed {str1} modeling data {str2}')
                              ))
            time.sleep(1)
            color_outputs[idx]='btn btn-outline-success'
            if state_disable[idx][1]['disabled']: #if disabled ->enable
                state_disable[idx][1]['disabled']=False
                disable_outputs[idx]=state_disable[idx]
            else:
                disable_outputs[idx]=dash.no_update
        except Exception as error:
            color_outputs[idx]='btn btn-outline-warning'
            unsucessful_process +=1
    if unsucessful_process==0:
        process_result="All databases processed successfully."
    else:
        process_result=f"{unsucessful_process} database(s) processed unsuccessfully."
    return dict(process_result=process_result,
                option_colors=color_outputs,
                option_disable=disable_outputs)
@callback(
    Output('tab2_processing_alert_collapse','is_open'),
    Input('tab2_processing_alert','children'),
    Input('tab2_model_Apply_Button','n_clicks'),
    Input('tab2_model_Download_Button','n_clicks'),
    prevent_initial_call=True
    )
def tab2_display_process_result(trigger1,trigger2,trigger3):
    """
    Show processing result.
    """
    if ctx.triggered_id=="tab2_processing_alert":
        return True
    elif ctx.triggered_id in ["tab2_model_Apply_Button",
                              "tab2_model_Download_Button"]:
        return False

@callback_extension(
    output=ServersideOutput('tab2_process_params','data'),
    inputs=dict(
        process_click=Input('tab2_model_Apply_Button','n_clicks'),
        parameter_check = State('tab2_inputs_alert','is_open'),
        ##DATABASE PARAMS
        #output uids
        output_uid = State({'code':'p1t2','name':ALL,"data_type":"update"},'data'),
        #chose all database
        chosen_all = State('tab2_proc_database_all_switch','value'),
        #chose Vietnam|world database
        chosen_database = State('tab2_proc_database','value'),
        #data by province|countries
        data_separation = State('tab2_proc_database_division_switch','value'),
        #chosen provinces/countries code
        chosen_sections = State('tab2_proc_database_division_checklist','value'),
        default_iso = State({'code':'p1t1','name':'iso_code',
                              "data_type":"internal"},'data'),
        #default|latest raw database
        w_default = State('tab1_select_w_database','value'),
        w_meta_default = State('tab1_select_w_meta_database','value'),
        vn_default = State('tab1_select_vn_database','value'),
        vn_vac_dist_default = State('tab1_select_vn_dist_database','value'),
        #birth&death rate manual setting
        auto_b_d_rate = State('tab2_auto_b_d_rate','value'),
        manual_b_rate = State({"code":"p1t2","name":"tab2_manual_b_d_rate",
            "type":"special_input","index":0},'value'),
        manual_d_rate = State({"code":"p1t2","name":"tab2_manual_b_d_rate",
            "type":"special_input","index":1},'value'),
        
        #RAW PARAMETERS INPUT
        raw_rolling=State({"code":"p1t2","name":"tab2_raw_rolling",
                            "type":"input"},'value'),
        trim_percent=State({"code":"p1t2","name":"tab2_trim_perc",
                            "type":"input"},'value'),
        death_distribution=State(
            {"code":"p1t2","name":"death_dist","type":"special_input","week":ALL},
            'value'),
        death_after=State({"code":"p1t2","name":"tab2_death_after","type":"input"},
                          'value'),
        death_rolling=State({"code":"p1t2","name":"tab2_death_rolling","type":"input"},
                            'value'),
        use_origin=State("tab2_use_original_death",'value'),
        recovery_distribution=State(
            {"code":"p1t2","name":"recovery_dist","type":"special_input","week":ALL},
            'value'),
        recovery_after=State({"code":"p1t2","name":"tab2_recovery_after","type":"input"},
                              'value'),
        raw_method_limit=State({"code":"p1t2","name":"tab2_raw_method_limit",
                                "type":"input"},'value'),
        raw_seed=State({"code":"p1t2","name":"tab2_raw_seed","type":"input"},
                        'value'),
        #VACCINATION PARAMETERS INPUT
        vac_info1=(
            State({"code":"p1t2","name":"tab2_vaccine_type1_param1",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type1_param2",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type1_param3",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type1_param4",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type1_param5",
                   "type":"input"},'value'),
            ),
        vac_info2=(
            State({"code":"p1t2","name":"tab2_vaccine_type2_param1",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type2_param2",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type2_param3",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type2_param4",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type2_param5",
                   "type":"input"},'value'),
            ),
        vac_info3=(
            State({"code":"p1t2","name":"tab2_vaccine_type3_param1",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type3_param2",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type3_param3",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type3_param4",
                   "type":"input"},'value'),
            State({"code":"p1t2","name":"tab2_vaccine_type3_param5",
                   "type":"input"},'value'),
            ),
        #check if manual base vaccination percent
        manual_base_vac = State('tab2_set_vaccine_perc','value'),
        base_vac = State({"code":"p1t2","name":"tab2_base_vac_ratio",
            "type":"special_input","index":ALL,"params":ALL},'value'),
        #check if manual target priority
        manual_vac_priority= State('tab2_set_vaccine_target','value'),
        manual_priority = State({"code":"p1t2","name":"tab2_vaccine_target",
            "type":"special_input","index":ALL,"params":ALL},'value'),
        
        w_vac_weight = State({'code':'p1t2','name':'tab2_w_vac_weight',
                            "data_type":"internal"},'data'),
        vn_vac_weight = State({'code':'p1t2','name':'tab2_vn_vac_weight',
                            "data_type":"internal"},'data'),
        vac_method_limit = State(
            {"code":"p1t2","name":"tab2_vaccine_method_limit","type":"input"},
            'value'),
        vac_seed=State({"code":"p1t2","name":"tab2_vaccine_seed",
                        "type":"input"},'value'),
        #PRE-MODEL PARAMETER INPUTS
        r_protect_time = State(
            {"code":"p1t2","name":"tab2_r_protect_time","type":"input"},'value'),
        s_vac_ratio = State(
            {"code":"p1t2","name":"tab2_s_vac_ratio","type":"input"},'value'),
        avg_infect_t = State(
            {"code":"p1t2","name":"tab2_avg_infect_t","type":"input"},'value'),
        avg_death_t = State(
            {"code":"p1t2","name":"tab2_avg_death_t","type":"input"},'value'),
        avg_recov_t = State(
            {"code":"p1t2","name":"tab2_avg_recov_t","type":"input"},'value'),
        avg_rotate = State(
            {"code":"p1t2","name":"tab2_avg_rotate","type":"input"},'value'),
        avg_data = State(
            {"code":"p1t2","name":"tab2_avg_data","type":"input"},'value'),
        weights = State('tab2_empty_f_method','value'),
        include_zero = State('tab2_include_zero','value'),
        r0_p = State({"code":"p1t2","name":"tab2_p","type":"input"},'value'),
        r0_n = State(
            {"code":"p1t2","name":"tab2_r0_n","type":"input"},'value'),
        r0_trim = State(
            {"code":"p1t2","name":"tab2_r0_trim","type":"input"},'value'),
        #GENERAL INPUTS
        max_worker=State({"code":"p1t2","name":"tab2_max_worker","type":"input"},'value'),
        ),
    prevent_initial_call=True
    )
def tab2_process_params_prepare(
    process_click,
    parameter_check,
    #output uids
    output_uid,
    #database params
    chosen_all,chosen_database,data_separation,
    chosen_sections,default_iso,
    w_default,w_meta_default,vn_default,vn_vac_dist_default,
    auto_b_d_rate,manual_b_rate,manual_d_rate,
    #raw parameters
    raw_rolling,trim_percent,death_distribution,
    death_after,death_rolling,use_origin,
    recovery_distribution,recovery_after,
    raw_method_limit,raw_seed,
    #vaccine parameters
    vac_info1,vac_info2,vac_info3,
    manual_base_vac,base_vac,
    manual_vac_priority,manual_priority,
    w_vac_weight,vn_vac_weight,vac_method_limit,vac_seed,
    #modeling input parameters
    r_protect_time,s_vac_ratio,avg_infect_t,avg_death_t,
    avg_recov_t,avg_rotate,avg_data,weights,include_zero,r0_p,
    r0_n,r0_trim,
    #general parameters
    max_worker
    ):
    """
    Due to some kind of limit to background callback dictionary inputs, if you 
    use many keys, the function 'running' argument won't load quickly.
    This function is used to handle that limitation.
    """
    if parameter_check:
        return dash.no_update
    else:
        if death_rolling<raw_rolling: 
            #in this case,working array will only contain NaN,so function can't 
            #work with it,setting equal to raw_rolling for this situation
            death_rolling=raw_rolling
        
        #process vaccination parameters
        vaccine_info = []
        for item in zip(['Non Replicating Viral Vector',
                         'Messenger RNA',
                         'Inactivated vaccine'],
                        [vac_info1,vac_info2,vac_info3]):
            add_item = [item[0]]
            add_item.extend(item[1])
            vaccine_info.append(add_item)
        vaccine_info = tuple(vaccine_info)
        
        if manual_base_vac:
            base_vaccine_percent=[]
            for i in range(len(base_vac)//4):
                item = base_vac[i*4:i*4+4]
                add_item=[item[:3]]
                if item[3][0]==item[3][1]:
                    add_item.extend([item[3][0],None])
                else:
                    add_item.extend([item[3][0],item[3][1]])
                base_vaccine_percent.append(add_item)
            base_vaccine_percent=tuple(base_vaccine_percent)
        else:
            base_vaccine_percent=[1,1,1]
            
        if manual_vac_priority:
            target_priority=[]
            for i in range(len(manual_priority)//4):
                item = manual_priority[i*4:i*4+4]
                add_item=[item[:3]]
                if item[3][0]==item[3][1]:
                    add_item.extend([item[3][0],None])
                else:
                    add_item.extend([item[3][0],item[3][1]])
                target_priority.append(add_item)
            target_priority=tuple(target_priority)
        else:
            target_priority=None
            
        w_other_vac_weight = tuple(w_vac_weight['Others'])
        w_vac_weight.pop('Others')
        vn_other_vac_weight = tuple(vn_vac_weight['Others'])
        vn_vac_weight.pop('Others')
        outputs = dict(
            database_params=[output_uid,chosen_all,chosen_database,data_separation,
                             chosen_sections,default_iso,w_default,
                             w_meta_default,vn_default,vn_vac_dist_default,auto_b_d_rate,
                             manual_b_rate,manual_d_rate],
            raw_params=[raw_rolling,trim_percent,death_distribution,death_after,
                        death_rolling,use_origin,recovery_distribution,
                        recovery_after,raw_method_limit,raw_seed],
            vac_params=[manual_base_vac,vaccine_info,base_vaccine_percent,target_priority,
                        w_vac_weight,w_other_vac_weight,vn_vac_weight,
                        vn_other_vac_weight,vac_method_limit,vac_seed],
            model_params=[r_protect_time,s_vac_ratio,avg_infect_t,avg_death_t,
                          avg_recov_t,avg_rotate,avg_data,weights,include_zero,
                          r0_p,r0_n,r0_trim],
            general_params=max_worker
                       )
        return outputs

##download data-------------------------------------------------------------
@callback(
    output=[
        Output('tab2_download','data')
        ],
    inputs = [
        Input('tab2_model_Download_Button','n_clicks'),
        #server_files
        State({'code':'p1t2','name':ALL,"data_type":"server"},'data'),
        #update_files
        State({'code':'p1t2','name':ALL,"data_type":"update"},'data'),
        ],
    background=True,
    running=[(Output('tab2_downloading_status','is_open'),True,False)],
    cancel=Input('tab2_model_Cancel_Button2','n_clicks'),
    prevent_initial_call=True)
def tab2_add_base_vac_ratio_download_zip(trigger,server_uid,modified_uid):
    """
    Download databases to local machine
    """
    def write_available_data(bytes_io,server_uid,modified_uid):
        """
        Function that handle zipping files
        """
        server_id_list = []
        modified_id_list = []
        for item in dash.callback_context.states_list[0]:
            server_id_list.append(item['id']['name'])
        for item in dash.callback_context.states_list[1]:
            modified_id_list.append(item['id']['name'])  
        with ZipFile(bytes_io, mode="w") as zf:
            directory = FileSystemStore(cache_dir="./cache/server_cache",
                                        default_timeout=0)
            for item in zip(server_id_list,server_uid):
                data = directory.get(item[1])
                if data is not None:
                    filepath = f"default/{item[0]}.csv"
                    zf.writestr(filepath,
                                data.to_csv(index=False))
            directory = FileSystemStore(cache_dir="./cache/output_cache",
                                        default_timeout=0)    
            for item in zip(modified_id_list,modified_uid):
                data = directory.get(item[1])
                if data is not None:
                    filepath = f"modified/{item[0]}.csv"
                    zf.writestr(filepath,
                                data.to_csv(index=False))
    return [dcc.send_bytes(write_available_data,"processed_data.zip",
                          server_uid=server_uid,modified_uid=modified_uid)
            ]