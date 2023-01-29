import pandas as pd
import numpy as np
import dash
from dash import callback,ctx
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_extensions.enrich import Input,Output,State,ServersideOutput,dcc,\
    html,ALL,MATCH,FileSystemStore
from dash_extensions.enrich import callback as callback_extension
import time,requests,datetime,os
from functools import partial
from multiprocessing import Manager,Pool
from zipfile import ZipFile
import re
import json
import plotly.graph_objects as go
from covid_science import utility_func
from covid_science.collection import browser_request,get_github_time_sery_url
from covid_science.workers import vn_casualty_wrapper_func
#------------------------------------------------------------------

#Tab1: Collecting data 

vn_province_code = pd.read_csv(
    'https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/'+
    'database/Covid-19_raw_data/VietNamData/Ministry_of_Health_data/province_code.csv',
    encoding='utf-8').values.tolist()

tab1_stored_data = html.Div(
    [
    #check criteria,True mean ok
    dcc.Store(id={'code':'p1t1','name':'normal_input',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t1','name':'driver_ready',"data_type":"input_check"},
              data=True),
    dcc.Store(id='tab1_initial_status',data=0),
    
    ##internal usage
    dcc.Store(id={'code':'p1t1','name':'iso_code',
                  "data_type":"internal"}),
    dcc.Store(id={'code':'p1t1','name':'vn_province_code',
                  "data_type":"internal"}),
    
    ##preloaded dataframe
    dcc.Store(id={'code':'p1t1','name':'birth_data',
                  "data_type":"census"}),
    dcc.Store(id={'code':'p1t1','name':'death_data',
                  "data_type":"census"}),
    dcc.Store(id={'code':'p1t1','name':'w_population',
                  "data_type":"census"}),
    dcc.Store(id={'code':'p1t1','name':'vn_population',
                  "data_type":"census"}),
    
    dcc.Store(id={'code':'p1t1','name':'w_covid_data',
                  "data_type":"server"}),
    dcc.Store(id={'code':'p1t1','name':'w_vaccine_volume',
                  "data_type":"server"}),
    dcc.Store(id={'code':'p1t1','name':'w_meta_database',
                  "data_type":"server"}),
    dcc.Store(id={'code':'p1t1','name':'vn_total_case',
                  "data_type":"server"}),
    dcc.Store(id={'code':'p1t1','name':'vn_province_case',
                  "data_type":"server"}),
    dcc.Store(id={'code':'p1t1','name':'vn_death_data',
                  "data_type":"server"}),
    dcc.Store(id={'code':'p1t1','name':'vn_vaccine_vol',
                  "data_type":"server"}),
    dcc.Store(id={'code':'p1t1','name':'vn_vaccine_dist',
                  "data_type":"server"}),
    
    #updated dataframe
    dcc.Store(id={'code':'p1t1','name':'w_covid_data_latest',
                  "data_type":"update"}),
    dcc.Store(id={'code':'p1t1','name':'w_vaccine_volume_latest',
                  "data_type":"update"}),
    dcc.Store(id={'code':'p1t1','name':'w_meta_database_latest',
                  "data_type":"update"}),
    dcc.Store(id={'code':'p1t1','name':'vn_total_case_latest',
                  "data_type":"update"}),
    dcc.Store(id={'code':'p1t1','name':'vn_province_case_latest',
                  "data_type":"update"}),
    dcc.Store(id={'code':'p1t1','name':'vn_death_data_latest',
                  "data_type":"update"}),
    dcc.Store(id={'code':'p1t1','name':'vn_vaccine_vol_latest',
                  "data_type":"update"}),
    dcc.Store(id={'code':'p1t1','name':'vn_vaccine_dist_latest',
                  "data_type":"update"}),
    
    #plot triggering
                    
    dcc.Store(id='tab1_figure_plot_trigger',data=0),
    dcc.Store(id='tab1_figure_dropdown_trigger',data=0),
    
    #store processing parameters
    dcc.Store(id="tab1_process_params"),
    
    #criteria
    dcc.Store(id={'code':'p1t1','name':'buffer_state'},
              data=[False,False,False,False]),
])

tab1_content = [dbc.Form([tab1_stored_data,
    dcc.Interval(id='tab1_data_initiator',max_intervals=1,n_intervals=0,
                  disabled=False),
    dbc.Alert("Please fill in all required parameters.",
              id='tab1_inputs_alert',is_open=False,color='danger',
              class_name="c_alert"),
    
    dbc.Label(html.Li("Color code:"),class_name="ms-1",width='auto',
              id='tab1_color_code_label'),
    dbc.Row([dbc.Col([dbc.Button("Default",size='sm',
                                 style = {'font-size':'0.8em'}),
                      dbc.Button("Attempt to update database not successful",
                                 size='sm',color="warning",
                                 style = {'font-size':'0.8em'}),
                      dbc.Button("Downloaded latest database",size='sm',
                                 color="success",
                                 style = {'font-size':'0.8em'}),
                      ],
                     width='auto')],
            justify='start'),
    
    dbc.Label(html.Li("World covid database:"),class_name="ms-1",width='auto',
              id='tab1_select_w_database_label'),
    html.Div(dbc.RadioItems(
            id='tab1_select_w_database',
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Default", "value": 1},
                {"label": "Latest", "value": 2,"disabled":True},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    dbc.Switch(id="tab1_download_w_database",
               label="Try to update latest covid database",
               value=False,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    
    dbc.Label(html.Li("World vaccine meta database:"),class_name="ms-1",
              width='auto',id='tab1_select_w_meta_database_label'),
    html.Div(dbc.RadioItems(
            id='tab1_select_w_meta_database',
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Default", "value": 1},
                {"label": "Latest", "value": 2,"disabled":True},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    dbc.Switch(id="tab1_download_w_meta_database",
               label="Try to update latest meta database",
               value=False,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    
    dbc.Label(html.Li("Vietnam download setting options:"),
              class_name="ms-1",width='auto',
              id='tab1_vn_download_setting_label'),
    
    dbc.Row([dbc.Label('Maximum processes:',width=7,id='tab1_max_worker_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t1","name":"tab1_max_worker","type":"input"},
                               type='number',size='sm',required=True,
                               min=1,step=1,value=round(os.cpu_count()/2),
                               className="mt-1 mb-1"),
                     width=4)                              
             ]),
    
    dbc.Row([dbc.Label('Maximum wait time:',width=7,
                       id='tab1_max_wait_time_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t1","name":"tab1_max_wait_time","type":"input"},
                               type='number',size='sm',required=True,
                               min=1,step=1,value=30,
                               className="mt-1 mb-1"),
                     width=4)                              
             ]),
    
    dbc.Row([dbc.Label('Maximum attempts:',width=7,
                       id='tab1_max_attempt_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t1","name":"tab1_max_attempt","type":"input"},
                               type='number',size='sm',required=True,
                               min=1,step=1,value=3,
                               className="mt-1 mb-1"),
                     width=4)                              
             ]),
    
    dbc.Row([dbc.Label('Interval time:',width=7,
                       id='tab1_interval_time_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t1","name":"tab1_interval_time","type":"input"},
                               type='number',size='sm',required=True,
                               min=1,step=1,value=10,
                               className="mt-1 mb-1"),
                     width=4)                              
             ]),
    
    dbc.Label("Chrome webdriver path:",class_name="mb-0 mt-2",width='auto',
              style={'font-size':'0.8em'},id='tab1_webdriver_path_options_label'),
    dbc.Row(dbc.RadioItems(options=[
                        {"label": 'Auto', "value": 'auto'},
                        {"label":'Directory path',"value":0},
                        {"label":'File path',"value":1}],
                                    id='tab1_webdriver_path_options',
                                    value='auto',
                                    label_style = {'font-size':'0.8em'},
                            inline=True)
            ),
    dbc.Row([dbc.Col(dbc.Input(id='tab1_webdriver_path',type='text',size='sm',
                               placeholder="Insert a file|directory path",
                               required=True,
                               className="mt-1 mb-1"),
                     width=11)                              
             ]),
    dbc.Alert("Input path doesn't exist.",
              id='tab1_webdriver_alert',is_open=False,color="danger",
              class_name="c_alert"),
    
    dbc.Label(html.Li("Vietnam covid database:"),
              class_name="ms-1",width=12,
              id='tab1_select_vn_database_label'),
    dbc.Button('Choose provinces',id='tab1_select_vn_database_button',
               size='sm',className="mb-1",style = {'font-size':'0.8em'}),
    html.Div(dbc.RadioItems(
            id='tab1_select_vn_database',
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Default", "value": 1},
                {"label": "Latest", "value": 2,"disabled":True},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    dbc.Switch(id="tab1_download_vn_database",
               label="Try to update latest covid database",
               value=False,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    
    dbc.Label(html.Li("Vietnam vaccine distribution database:"),
              class_name="ms-1",width='auto',
              id='tab1_select_vn_dist_database_label'),
    html.Div(dbc.RadioItems(
            id='tab1_select_vn_dist_database',
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Default", "value": 1},
                {"label": "Latest", "value": 2,"disabled":True},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    dbc.Switch(id="tab1_download_vn_dist_database",
               label="Try to update latest distribution database",
               value=False,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    dbc.Collapse([dbc.Alert(id='tab1_processing_alert',className='c_alert')],
                 id='tab1_processing_alert_collapse',
                 is_open=False,),
    dcc.Download(id="tab1_download"),
    
    ],
    className="mt-1 mb-1 bg-secondary",
    style={'color':'white'}),
    dbc.Row([dbc.Col([dbc.Button("Update",id="tab1_model_Update_Button",
                                 n_clicks=0,size='sm'),
                      dbc.Button("Download",id="tab1_model_Download_Button",
                                  n_clicks=0,size='sm'),],
                     width='auto')],
            justify='end'),
    ]

#Tab1 Modals
tab1_modal = html.Div([
    dbc.Modal(
        [dbc.ModalHeader('Choose provinces',close_button=False),
          dbc.ModalBody([dbc.Switch(id="tab1_model_database_all_provinces",
                                    label="Select all",
                                    value=True,
                                    style = {'font-size':'0.8em'},
                                    className="mb-0 mt-2"),
                         dbc.Alert("At least 1 province must be chosen.",
                                   id='tab1_model_database_province_checklist_alert',
                                   is_open=False,color="danger",
                                   class_name="c_alert"),
                        dbc.Checklist(id="tab1_model_database_province_checklist",
                                      options=[{'label':item[0],
                                                'value':item[1]} 
                                                for item in vn_province_code
                                                ],
                                      value=[item[1] for item in vn_province_code],
                                      label_style = {'font-size':'0.8em'})
                        ]),
          dbc.ModalFooter(dbc.Button("Ok", id="tab1_province_option_ok",
                          className="ms-auto",n_clicks=0))
        ],
        id="tab1_province_option",
        is_open=False,
        backdrop='static',
        scrollable=True),
    dbc.Modal(
        [dbc.ModalHeader('Updating status',close_button=False),
          dbc.ModalBody([
                         dbc.Row([dbc.Col(dbc.Spinner(color="primary",size='sm'),
                                           width='auto'),
                                   dbc.Col(html.P(children="Updating...",
                                                  id='tab1_update_process_info',
                                                  style = {'font-size':'0.8em'}),
                                           width='auto')
                                   ]),
                         dbc.Collapse([dbc.Progress(
                                 id='tab1_update_process',
                                 striped=True)],
                             id='tab1_update_process_collapse',
                             is_open=False),
                        ]),
          dbc.ModalFooter(dbc.Button("Cancel", id="tab1_model_Cancel_Button1",
                          className="ms-auto",size='sm',n_clicks=0))
        ],
        id="tab1_processing_status",
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
          dbc.ModalFooter(dbc.Button("Cancel", id="tab1_model_Cancel_Button2",
                          className="ms-auto",size='sm',n_clicks=0))
        ],
        id="tab1_downloading_status",
        is_open=False,
        backdrop='static',
        centered=True),
    
    dbc.Modal(
        [dbc.ModalHeader('Loading status',close_button=False),
          dbc.ModalBody([
                         dbc.Row([dbc.Col(dbc.Spinner(color="primary",size='sm'),
                                           width='auto'),
                                   dbc.Col(html.P(children="Page initial loading...",
                                                  style = {'font-size':'0.8em'}),
                                           width='auto')
                                   ]),
                        ]),
        ],
        id="tab1_initial_loading_status",
        is_open=True,
        backdrop='static',
        centered=True),
    ])

#Tab1 plots

tab1_figure = html.Div([
    dbc.Tabs([
    dbc.Tab(
        utility_func.add_row_choices(
            ['Database:'], 
            [[{"label":"World Demographic","value":"world_demographic"},
              {"label": "Vietnam population by province", "value": "vn_population"}
              ],
             ], 
            ['world_demographic'], 
            [{"tab1_figure_tabs":'t1','type':'select_data',"data_type":"census"}],
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        label="Census",
        id={"tab1_figure_tabs":'t1'}),
    dbc.Tab(
        utility_func.add_row_choices(
            ['Database:'], 
            [[{"label": "World Covid-19 data", "value": "w_covid_data"},
              {"label": "World vaccine volume", "value": "w_vaccine_volume"},
              {"label": "World vaccine metadata", "value": "w_meta_database"},
              {"label": "Vietnam Covid-19 data", "value": "vn_total_case"},
              {"label": "Vietnam case by province", "value": "vn_province_case"},
              {"label": "Vietnam death case by province", "value": "vn_death_data"},
              {"label": "Vietnam vaccine volume", "value": "vn_vaccine_vol"},
              {"label": "Vietnam vaccine distribution", "value": "vn_vaccine_dist"}],
             ], 
            ['w_covid_data'], 
            [{"tab1_figure_tabs":'t2','type':'select_data',"data_type":"server"}],
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        label="Server data",
        id={"tab1_figure_tabs":'t2'}),
    dbc.Tab("No updated data",label="Updated data",
            id={"tab1_figure_tabs":'t3'}),
    ],
    id="tab1_plot_tabs",
    active_tab="tab-0",
    style = {'font-size':'0.7em'},
    class_name='mb-1'
    ),
    html.Div(id='tab1_figure_add_dropdown'),
    html.Div(id='tab1_figure_output'),
])

#Tab1 Tooltips
tab1_tooltip = html.Div([
    dbc.Tooltip(html.Div('''
        Color code to show the status of the relevant databases.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_color_code_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Trigger the button to enable to download databases of Covid-19 case 
        summary & vaccine volume.
        
        If databases are downloaded successfully, user can switch between 
        default & updated database for processing step.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_select_w_database_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Trigger the button to enable to download vaccination metadata of all
        countries.
        
        If database is downloaded successfully, user can switch between 
        default & updated database for processing step.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_select_w_meta_database_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Setting options for Vietnam Covid-19 database update using webdriver 
        since the Ministry of Health website aren't easy to request database 
        directly.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_vn_download_setting_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The maximum number of process which will be used for multiprocessing.
        
        It is recommended not to set this number too high (normally use around 
        half of CPU processors), otherwise it might slow down the update 
        process.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_max_worker_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The maximum waiting time for the website response.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_max_wait_time_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The maximum number of attempts to get website response before stopping.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_max_attempt_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The cummulative added time between each attempt.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_interval_time_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        The settings directory for download Chrome webdriver or path to an
        existing chromedriver.exe. Options are:
        * Auto: auto download to the current working directory
        * Directory path: download to the chosen path
        * File path: retrieve the chromedriver.exe from the path
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_webdriver_path_options_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Trigger the button to enable to download Vietnam Covid-19 databases
        which includes: case, death case, vaccine volume.
        
        The 'Choose provinces' button is used to select which province death 
        cases will be collected. This is normally should be left as default.
        
        If databases are downloaded successfully, user can switch between 
        default & updated database for processing step.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_select_vn_database_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Trigger the button to enable to download Vietnam Covid-19 vaccine
        distribution database
        
        If databases are downloaded successfully, user can switch between 
        default & updated database for processing step, although this is not 
        recommended since database is tend to miss many provinces. The default
        database is enough to estimate the distribution ratio between provinces.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab1_select_vn_dist_database_label',
        delay={'show': 1000}),
])

#Tab1 callbacks

##initial data loading--------------------------------------------------------

census_backend = FileSystemStore(cache_dir="./cache/census_cache",threshold=50,
                             default_timeout=0
                             )

@callback_extension(
    Output({'code':'p1t1','name':'vn_province_code',"data_type":"internal"},'data'),
    Output({'code':'p1t1','name':'iso_code',"data_type":"internal"},'data'),
    Input({'code':'p1t1','name':'w_covid_data',"data_type":"server"},'data'),
    prevent_initial_call=True,
    )
def tab1_load_default_code(w_df):
    """
    Load default databases before any other callback.
    """
    vn_province_code = pd.read_csv(
        'https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/'+
        'database/Covid-19_raw_data/VietNamData/Ministry_of_Health_data/province_code.csv',
        encoding='utf-8').values.tolist()
    check_df = pd.read_csv(
        'https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/'+
        'database/Covid-19_raw_data/census_data/crude_birth_rate.csv',
        encoding='utf-8',
        usecols=['Region, subregion, country or area *',
                 'ISO3 Alpha-code']).drop_duplicates()
    if w_df is None:
        w_df =  pd.read_csv(
            "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
            "database/Covid-19_raw_data/OurWorldinData/covid_data_simplified.csv",
            usecols=['iso_code','location']).drop_duplicates()
    else:
        w_df = w_df[['iso_code','location']].drop_duplicates().copy()
    
    iso_code = pd.merge(w_df,check_df,
                        left_on='iso_code',
                        right_on='ISO3 Alpha-code')[['iso_code','location']].values.tolist()
    
    return vn_province_code,iso_code

@callback_extension(
    [ServersideOutput({'code':'p1t1','name':'birth_data',"data_type":"census"},
                      'data',arg_check=False,session_check=False,
                      backend=census_backend),
    ServersideOutput({'code':'p1t1','name':'death_data',"data_type":"census"},
                     'data',arg_check=False,session_check=False,
                     backend=census_backend),
    ServersideOutput({'code':'p1t1','name':'w_population',"data_type":"census"},
                     'data',arg_check=False,session_check=False,
                     backend=census_backend),
    ServersideOutput({'code':'p1t1','name':'vn_population',"data_type":"census"},
                     'data',arg_check=False,session_check=False,
                     backend=census_backend),
    ],
    Input('tab1_data_initiator','n_intervals'),
    memoize=True,
    prevent_initial_call=True
    )
def tab1_load_default_census_cached_database(click):
    """
    Load serverside census databases as caches for faster data access.
    """
    birth_df = pd.read_csv(
        'https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/'+
        'database/Covid-19_raw_data/census_data/crude_birth_rate.csv',
        encoding='utf-8',
        usecols=['Region, subregion, country or area *',
                 'Location code',
                 'ISO3 Alpha-code',
                 'Year',
                 'Crude Birth Rate (births per 1,000 population)'])  

    death_df = pd.read_csv(
        'https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/'+
        'database/Covid-19_raw_data/census_data/crude_death_rate.csv',
        encoding='utf-8',
        usecols=['Region, subregion, country or area *',
                 'Location code',
                 'ISO3 Alpha-code',
                 'Year',
                 'Crude Death Rate (deaths per 1,000 population)'])

    population_df = pd.read_csv(
        'https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/'+
        'database/Covid-19_raw_data/census_data/population.csv',
        encoding='utf-8',
        usecols=['Region, subregion, country or area *',
                 'Location code',
                 'ISO3 Alpha-code',
                 'Year',
                 'Total Population, as of 1 January (thousands)'])
    
    vn_population_df = pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_raw_data/VietNamData/vietnam_population.csv")
    
    return [birth_df,
            death_df,
            population_df,
            vn_population_df]
    
server_backend = FileSystemStore(cache_dir="./cache/server_cache",threshold=50,
                             default_timeout=0
                             )

@callback_extension(
    [
    ServersideOutput({'code':'p1t1','name':'w_covid_data',"data_type":"server"},
                     'data',arg_check=False,session_check=False,
                     backend=server_backend),
    ServersideOutput({'code':'p1t1','name':'w_vaccine_volume',"data_type":"server"},
                     'data',arg_check=False,session_check=False,
                     backend=server_backend),
    ServersideOutput({'code':'p1t1','name':'w_meta_database',"data_type":"server"},
                     'data',arg_check=False,session_check=False,
                     backend=server_backend),
    ServersideOutput({'code':'p1t1','name':'vn_total_case',"data_type":"server"},
                     'data',arg_check=False,session_check=False,
                     backend=server_backend),
    ServersideOutput({'code':'p1t1','name':'vn_province_case',"data_type":"server"},
                     'data',arg_check=False,session_check=False,
                     backend=server_backend),
    ServersideOutput({'code':'p1t1','name':'vn_death_data',"data_type":"server"},
                     'data',arg_check=False,session_check=False,
                     backend=server_backend),
    ServersideOutput({'code':'p1t1','name':'vn_vaccine_vol',"data_type":"server"},
                     'data',arg_check=False,session_check=False,
                     backend=server_backend),
    ServersideOutput({'code':'p1t1','name':'vn_vaccine_dist',"data_type":"server"},
                     'data',arg_check=False,session_check=False,
                     backend=server_backend)
    ],
    Input('tab1_data_initiator','n_intervals'),
    memoize=True,
    prevent_initial_call=True
    )
def tab1_load_default_cached_database(click):
    """
    Load serverside databases as caches for faster data access.
    """
    w_covid_data = pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_raw_data/OurWorldinData/covid_data_simplified.csv",
        parse_dates=["date"])

    w_vaccine_volume = pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_raw_data/OurWorldinData/vaccine_volume.csv",
        encoding='utf-8')

    w_meta = pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_raw_data/WHO/vaccine_meta.csv",
        encoding='utf-8',parse_dates=['START_DATE'])

    vn_total_case = pd.read_csv(get_github_time_sery_url(
        "https://api.github.com/repos/DeKhaos/covid-modeling/contents/"+
        "database/Covid-19_raw_data/VietNamData/Ministry_of_Health_data/total_case"),                   
        parse_dates=["date"])
    
    vn_province_case = pd.read_csv(get_github_time_sery_url(
        "https://api.github.com/repos/DeKhaos/covid-modeling/contents/"+
        "database/Covid-19_raw_data/VietNamData/Ministry_of_Health_data/case_by_province"))
    
    vn_death_data = pd.read_csv(get_github_time_sery_url(
        "https://api.github.com/repos/DeKhaos/covid-modeling/contents/"+
        "database/Covid-19_raw_data/VietNamData/Ministry_of_Health_data/death_data"),                   
        parse_dates=["date"])

    vn_vaccine_vol = pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_raw_data/OurWorldinData/vietnam_vaccine.csv",
        parse_dates=["date"])
    
    vn_vaccine_dist = pd.read_csv(get_github_time_sery_url(
        "https://api.github.com/repos/DeKhaos/covid-modeling/contents/"+
        "database/Covid-19_raw_data/VietNamData/vaccine_distribution"))
    
    return [w_covid_data,
            w_vaccine_volume,
            w_meta,
            vn_total_case,
            vn_province_case,
            vn_death_data,
            vn_vaccine_vol,
            vn_vaccine_dist]

@callback(
    Output('tab1_figure_dropdown_trigger','data'),
    Input({'code':'p1t1','name':'vn_province_code',"data_type":"internal"},'data'),
    Input({'code':'p1t1','name':'birth_data',"data_type":"census"},'data'),
    Input({'code':'p1t1','name':'w_vaccine_volume',"data_type":"server"},'data'),
    Input('model_param_tabs','active_tab'),
    State('tab1_figure_dropdown_trigger','data'),
    prevent_initial_call=True
    )
def tab1_trigger_plot_process(initial_trigger1,
                              initial_trigger2,
                              initial_trigger3,
                              active_tab,count):
    """
    Synchronize trigger condition of plotting.
    """
    if active_tab=="tab-0":
        return count+1
    else:
        return dash.no_update
    
@callback_extension(
    [ServersideOutput({'code':'p1t1','name':'w_covid_data_latest',
                       "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t1','name':'w_vaccine_volume_latest',
                      "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t1','name':'w_meta_database_latest',
                      "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t1','name':'vn_total_case_latest',
                      "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t1','name':'vn_province_case_latest',
                      "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t1','name':'vn_death_data_latest',
                      "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t1','name':'vn_vaccine_vol_latest',
                      "data_type":"update"},'data'),
    ServersideOutput({'code':'p1t1','name':'vn_vaccine_dist_latest',
                      "data_type":"update"},'data')],
    Input('tab1_data_initiator','n_intervals'),
    prevent_initial_call=True
    )
def tab1_generate_uid(trigger):
    """
    Generate uid for cache access of updated databases.
    """
    outputs = [None for _ in range(len(ctx.outputs_list))]
    return outputs

#handle checking of preloaded data
@callback(
    output = Output('tab1_initial_status','data'),
    inputs = [
        Input('tab1_data_initiator','n_intervals'),
        Input({'code':'p1t1','name':'vn_province_code',"data_type":"internal"},'data'),
        Input({'code':'p1t1','name':'birth_data',"data_type":"census"},'data'),
        Input({'code':'p1t1','name':'w_vaccine_volume',"data_type":"server"},'data'),
        ],
    prevent_initial_call=True
    )
def tab1_initial_data_loaded(trigger1,trigger2,trigger3,trigger4):
    """Use to check if all tab preloaded data is ready"""
    if trigger1==0:
        return dash.no_update
    else:
        return 1

#handle the initial popup modal
@callback(
    output = Output('tab1_initial_loading_status','is_open'),
    inputs = [
        Input('tab1_initial_status','data'),
        Input('tab2_initial_status','data'),
        Input('tab3_initial_status','data'),
        ],
    )
def page1_load_waiting(value1,value2,value3):
    """
    Make sure all tabs are loaded properly before allowing interaction.
    Save the status to cache
    """
    
    if sum([value1,value2,value3])!=3:
        return dash.no_update
    else:
        
        return False

##buttons interactions--------------------------------------------------------

@callback([Output('tab1_province_option','is_open'),
           Output('tab1_model_database_province_checklist_alert','is_open'),
           Output('tab1_model_database_province_checklist','value'),
           Output('tab1_model_database_all_provinces','value')],
          [Input('tab1_select_vn_database_button','n_clicks'),
           Input('tab1_province_option_ok','n_clicks'),
           Input('tab1_model_database_all_provinces','value'),
           Input('tab1_model_database_province_checklist','value')],
          [State('tab1_model_database_province_checklist','value'),
           State({'code':'p1t1','name':'vn_province_code',
                  "data_type":"internal"},'data')],
          prevent_initial_call=True)
def t1_open_province(button1,button2,all_switch,checklist,option_state,
                     vn_province_code):
    '''
    Open and close the modal for choosing Vietnam provinces, can't close if 
    no province was chosen.
    '''
    all_options = [item[1] for item in vn_province_code]
    if ctx.triggered_id =="tab1_select_vn_database_button":
        return [True,dash.no_update,dash.no_update,dash.no_update]
    elif ctx.triggered_id=="tab1_model_database_all_provinces":
        if all_switch:
            return [dash.no_update,dash.no_update,all_options,dash.no_update]
        else:
            return [dash.no_update,dash.no_update,[],dash.no_update]
    elif ctx.triggered_id=="tab1_model_database_province_checklist":
        if np.all(np.isin(all_options,checklist)):
            return [dash.no_update,dash.no_update,dash.no_update,True]
        else:
            return [dash.no_update,dash.no_update,dash.no_update,False]
    else:
        if option_state ==[]:
            return [True,True,dash.no_update,dash.no_update]
        else:
            return [False,False,dash.no_update,dash.no_update]
        
@callback(Output('tab1_webdriver_path','value'),
          Output('tab1_webdriver_alert','is_open'),
          Output('tab1_webdriver_path_options','value'),
          Output('tab1_webdriver_path','invalid'),
          Output({'code':'p1t1','name':'driver_ready',"data_type":"input_check"},'data'),
          Input('tab1_webdriver_path','value'),
          Input('tab1_webdriver_path_options','value'))
def t1_driver_path(path,option):
    """
    Add chromedriver download directory or specific a path to an existing  
    chromedriver.exe file
    """
    if ctx.triggered_id=="tab1_webdriver_path":
       if os.path.isdir(fr"{path}"):
           return dash.no_update,False,0,False,True
       elif os.path.isfile(fr"{path}"):
           return dash.no_update,False,1,False,True
       else:
           return dash.no_update,True,None,True,False
    else:
       if option=='auto':
           auto_path = os.getcwd()
           return auto_path,False,dash.no_update,False,True
       else:
           return None,False,dash.no_update,True,False

##input parameters checking----------------------------------------------------

@callback(
    Output({'code':'p1t1','name':'normal_input',"data_type":"input_check"},'data'),
    Input({"code":"p1t1","name":ALL,"type":"input"},'value'),
    prevent_initial_call=True
    )
def tab1_check_normal_input(values):
    """
    Check input status normal inputs.
    """
    check_array = np.array(values,dtype='float')
    check_value = np.any(np.isnan(check_array))
    return not check_value

@callback(
    Output('tab1_inputs_alert','is_open'),
    Input({'code':'p1t1','name':ALL,"data_type":"input_check"},'data'),
    prevent_initial_call=True
    )
def tab1_check_input_warning(values):
    """
    Check all dcc.Store of input status to print fill in requirement.
    """
    check_value = np.all(values)
    return not check_value

##plot interaction-------------------------------------------------------------

@callback_extension(
    [Output('tab1_figure_output','children'),
    Output('tab1_figure_add_dropdown','children'),
    Output('tab1_figure_plot_trigger','data')],
    Input('tab1_figure_dropdown_trigger','data'), #trigger
    Input('tab1_plot_tabs','active_tab'),
    Input({'tab1_figure_tabs':ALL,'type':'select_data',"data_type":ALL},'value'),
    #census_files
    State({'code':'p1t1','name':'birth_data',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'death_data',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'w_population',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'vn_population',"data_type":"census"},'data'),
    #server_files
    State({'code':'p1t1','name':'w_covid_data',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'w_vaccine_volume',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'w_meta_database',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_total_case',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_province_case',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_death_data',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_vaccine_vol',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_vaccine_dist',"data_type":"server"},'data'),
    #update_files
    State({'code':'p1t1','name':'w_covid_data_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'w_vaccine_volume_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'w_meta_database_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'vn_total_case_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'vn_province_case_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'vn_death_data_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'vn_vaccine_vol_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'vn_vaccine_dist_latest',"data_type":"update"},'data'),
    
    State('tab1_figure_plot_trigger','data'),
    prevent_initial_call=True
    )
def tab1_figure_option(initial_trigger,tab_click,data_click,
                       c1,c2,c3,c4,
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
        if item['id']['tab1_figure_tabs']==f't{idx+1}':
            database_name = item['value']
            break
    wrapper = html.Div(id={'type':'figure_wrapper',
                           'tab1_figure_tabs':f't{idx+1}'})
    if database_name=="world_demographic":
        df = c1
        output = html.Div(utility_func.add_row_choices(
            ['Country & Area:'], 
            [[{"label":item[0],"value":item[1]} for item in 
              df.iloc[:,[0,1]].drop_duplicates().values],
             ], 
            [df.iloc[:,[0,1]].drop_duplicates().values[0][1]], 
            [{'code':'p1t1','tab1_figure_tabs':'t1',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                'plot_arg':None,'layout_arg':None,
                'code_obj':'{}.loc[{}["Location code"]=={}]',
                'obj_type':'expression','format':['df','df','variable']})},
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t1','tab1_figure_tabs':'t1',
            "plot_type":json.dumps(["line","line"]),
            'xaxis_type':json.dumps([["number",],["number",]]),
            'preset':json.dumps([
                {'dataframe':None,
                 'plot':{'x':'Year',
                         'y':['Crude Birth Rate (births per 1,000 population)',
                              'Crude Death Rate (deaths per 1,000 population)']
                         },
                 'layout':{'layout_legend_title':'Criteria'}},
                {'dataframe':{'format':['df','df','df'],
                              'code_obj':'{}.mul(np.where({}.columns!="Total'+
                              ' Population, as of 1 January (thousands)",'+
                              ' np.full({}.columns.size,1), 1000))',
                              'obj_type':'expression'},
                 'plot':{'x':'Year',
                         'y':'Total Population, as of 1 January (thousands)'},
                 'layout':{'layout_yaxis_title':'Total Population'}}
                                 ]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        ),
    elif database_name=="vn_population":
        df = c4
        output = html.Div(utility_func.add_row_choices(
            ['Plot option:','Province:'], 
            [[{"label":i,"value":i} for i in df.iloc[:,1].unique()],
             [{"label":i,"value":i} if i not in ['All']
              else {"label":i,"value":{"All":json.dumps(df.columns[2:]
                                                        .tolist())
                                       }[i]}
              for i in np.append("All",df.columns[2:])]
             ], 
            [df.iloc[:,1].unique()[0],df.columns[2:][0]], 
            [{'code':'p1t1','tab1_figure_tabs':'t1',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                'plot_arg':None,'layout_arg':'layout_yaxis_title_text',
                'code_obj':'{}.loc[{}["criteria"]=={}]',
                'obj_type':'expression',
                'format':['df','df','variable']})},
             {'code':'p1t1','tab1_figure_tabs':'t1',"parameters":1,
              "extra_arg":json.dumps({'target':['plot'],
               'plot_arg':'y','layout_arg':None,
               'code_obj':None,'obj_type':None,'format':None})}],
            persistence=[database_name,database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t1','tab1_figure_tabs':'t1',
            "plot_type":json.dumps(["line"]),
            'xaxis_type':json.dumps([["number",]]),
            'preset':json.dumps([{'dataframe':None,
                                 'plot':{'x':'year'},
                                 'layout':{'layout_legend_title':'Province'}}]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        ),
    elif database_name in ["w_covid_data","w_covid_data_latest"]:
        if database_name=="w_covid_data":
            df = s1
            tab = 't2'
        else:
            df = u1
            tab = 't3'
        output = html.Div(utility_func.add_row_choices(
            ['Country & Area:'], 
            [[{"label":item[1],"value":item[0]} for item in 
              df.iloc[:,[0,2]].drop_duplicates().values],
             ], 
            [df.iloc[:,[0,2]].drop_duplicates().values[0][0]], 
            [{'code':'p1t1','tab1_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                'plot_arg':None,'layout_arg':None,
                'code_obj':'{}.loc[{}["iso_code"]=={}]',
                'obj_type':'expression',
                'format':['df','df','variable']})},
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t1','tab1_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["line","line","line"]),
            'xaxis_type':json.dumps([["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"],
                                     ["date","%Y-%m-%d"]]),
            'preset':json.dumps([
                {'dataframe':None,
                 'plot':{'x':'date','y':'new_cases'},
                 'layout':None},
                {'dataframe':None,
                 'plot':{'x':'date','y':'new_deaths',
                         'color_discrete_sequence':['red']},
                 'layout':None},
                {'dataframe':None,
                 'plot':{'x':'date','y':'new_vaccinations_smoothed',
                         'color_discrete_sequence':['lightgreen']},
                 'layout':None}
                                 ]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        ),
    elif database_name in ['w_vaccine_volume','w_vaccine_volume_latest']:
        if database_name=="w_vaccine_volume":
            df = s2
            tab = 't2'
        else:
            df = u2
            tab = 't3'
        output = html.Div(utility_func.add_row_choices(
            ['Location:'], 
            [[{"label":item,"value":item} for item in 
              df.iloc[:,0].unique()]
             ], 
            [df.iloc[:,0].unique()[0]], 
            [{'code':'p1t1','tab1_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                'plot_arg':None,'layout_arg':None,
                'code_obj':'{}.loc[{}["location"]=={}]',
                'obj_type':'expression',
                'format':['df','df','variable']})}
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t1','tab1_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["line"]),
            'xaxis_type':json.dumps([["date","%Y-%m-%d"]]),
            'preset':json.dumps([
                {'dataframe':None,
                 'plot':{'x':'date','y':'total_vaccinations','color':'vaccine'},
                 'layout':None}
                                 ]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        )
    elif database_name in ['w_meta_database','w_meta_database_latest']:
        iso_dict = s1.set_index('iso_code').to_dict()['location']
        if database_name=="w_meta_database":
            df = s3
            tab = 't2'
        else:
            df = u3
            tab = 't3'
        output = html.Div(utility_func.add_row_choices(
            ['Location:'], 
            [[{"label":iso_dict[item],"value":item} for item in 
              df.iloc[:,0].sort_values().unique() if item in iso_dict.keys()]
             ], 
            [df.iloc[:,0].sort_values().unique()[0]], 
            [{'code':'p1t1','tab1_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                'plot_arg':None,'layout_arg':None,
                'code_obj':'{}.loc[{}["ISO3"]=={}]',
                'obj_type':'expression',
                'format':['df','df','variable']})}
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t1','tab1_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["timeline"]),
            'xaxis_type':json.dumps([["time","%Y-%m-%d"]]),
            'preset':json.dumps([
                {'dataframe':{'format':[],
                              'code_obj':'{}.loc[{}["END_DATE"].isna(),'+
                              '"END_DATE"]=datetime.datetime.now().date()',
                              'obj_type':'statement'},
                 'plot':{'x_start':'START_DATE',
                         'x_end':'END_DATE',
                         'y':'PRODUCT_NAME',
                         'color':'PRODUCT_NAME'},
                 'layout':None}
                                 ]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        )
    elif database_name in ['vn_total_case','vn_total_case_latest']:
        if database_name=="vn_total_case":
            df = s4
            tab = 't2'
        else:
            df = u4
            tab = 't3'
        output = html.Div(utility_func.add_row_choices(
            ['Parameters:'], 
            [[{"label":item,"value":item} for item in 
              df.iloc[:,3].unique()]
             ], 
            [df.iloc[:,3].unique()[0]], 
            [{'code':'p1t1','tab1_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                'plot_arg':None,'layout_arg':None,
                'code_obj':'{}.loc[{}["code"]=={}]',
                'obj_type':'expression',
                'format':['df','df','variable']})}
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t1','tab1_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["line"]),
            'xaxis_type':json.dumps([["time","%Y-%m-%d"]]),
            'preset':json.dumps([
                {'dataframe':None,
                 'plot':{'x':'date',
                         'y':'case'},
                 'layout':None}
                                 ]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        )
        
    elif database_name in ['vn_province_case','vn_province_case_latest']:
        if database_name=="vn_province_case":
            df = s5
            tab = 't2'
        else:
            df = u5
            tab = 't3'
        output = html.Div(utility_func.add_row_choices(
            ['Province:'], 
            [[{"label":i,"value":i} if i not in ['All']
              else {"label":i,"value":{"All":json.dumps(df.iloc[:,1]
                                                        .tolist())
                                             }[i]}
              for i in np.append("All",df.iloc[:,1])]
             ], 
            [df.iloc[:,1].unique()[0]], 
            [{'code':'p1t1','tab1_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['plot'],
                'plot_arg':'y','layout_arg':None,
                'code_obj':None,
                'obj_type':None,'format':None})},
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t1','tab1_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["line"]),
            'xaxis_type':json.dumps([["time","%Y-%m-%d"]]),
            'preset':json.dumps([{'dataframe':None,
                                 'plot':{'x':'date'},
                                 'layout':{'layout_legend_title':'Province',
                                           'layout_xaxis_rangeslider_visible':True}}]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        ),
        
    elif database_name in ['vn_death_data','vn_death_data_latest']:
        if database_name=="vn_death_data":
            df = s6
            tab = 't2'
        else:
            df = u6
            tab = 't3'
        output = html.Div(utility_func.add_row_choices(
            ['Province:'], 
            [[{"label":i,"value":i} if i not in ['All']
              else {"label":i,"value":{"All":df.columns[1]
                                             }[i]}
              for i in np.append("All",df.columns[3:])]
             ], 
            [df.columns[3:][0]], 
            [{'code':'p1t1','tab1_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['plot'],
                'plot_arg':'y','layout_arg':None,
                'code_obj':None,
                'obj_type':None,'format':None})},
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t1','tab1_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["line"]),
            'xaxis_type':json.dumps([["time","%Y-%m-%d"]]),
            'preset':json.dumps([{'dataframe':None,
                                 'plot':{'x':'date'},
                                 'layout':{'layout_legend_title':'Province',
                                           'layout_xaxis_rangeslider_visible':True}}]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        ),
    elif database_name in ['vn_vaccine_vol','vn_vaccine_vol_latest']:
        if database_name=="vn_vaccine_vol":
            df = s7
            tab = 't2'
        else:
            df = u7
            tab = 't3'
        output = html.Div(
        id={'code':'p1t1','tab1_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["line"]),
            'xaxis_type':json.dumps([["time","%Y-%m-%d"]]),
            'preset':json.dumps([{'dataframe':None,
                                 'plot':{'x':'date',
                                         'y':'new_vaccinations_smoothed'},
                                 'layout':{'layout_xaxis_rangeslider_visible':True}}]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        ),
    elif database_name in ['vn_vaccine_dist','vn_vaccine_dist_latest']:
        if database_name=="vn_vaccine_dist":
            df = s8
            tab = 't2'
        else:
            df = u8
            tab = 't3'
        output = html.Div(utility_func.add_row_choices(
            ['Parameter:'], 
            [[{"label":i,"value":i} for i in df.columns[3:6]]
             ], 
            [df.columns[3:6][2]], 
            [{'code':'p1t1','tab1_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['plot'],
                'plot_arg':'y','layout_arg':None,
                'code_obj':None,
                'obj_type':None,'format':None})},
             ],
            persistence=[database_name],
            persistence_type="session",
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        id={'code':'p1t1','tab1_figure_tabs':f'{tab}',
            "plot_type":json.dumps(["bar"]),
            'xaxis_type':json.dumps([["number",]]),
            'preset':json.dumps([{'dataframe':None,
                                 'plot':{'x':'provinceName'},
                                 'layout':{'layout_xaxis_tickangle':45}}]),
            'grid':json.dumps({'grid_arg':None,
                               'col_arg':None})
            }
        ),
    else:
        output = []
    
    return [wrapper,output,trigger_count+1]

@callback(
    Output({"tab1_figure_tabs":'t3'},'children'),
    Input('tab1_processing_alert','children'),
    State({'code':'p1t1','name':ALL,"data_type":"update"},'data'),
    )
def tab1_update_database_dropdown(trigger,updated_databases):
    """
    Update the Update tab if any database is downloaded.
    """
    database_label = {"w_covid_data_latest":"World Covid-19 data",
                      "w_vaccine_volume_latest":"World vaccine volume",
                      "w_meta_database_latest":"World vaccine metadata",
                      "vn_total_case_latest":"Vietnam Covid-19 data",
                      "vn_province_case_latest":"Vietnam case by province",
                      "vn_death_data_latest":"Vietnam death case by province",
                      "vn_vaccine_vol_latest":"Vietnam vaccine volume",
                      "vn_vaccine_dist_latest":"Vietnam vaccine distribution"}
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
            [{"tab1_figure_tabs":'t3','type':'select_data',"data_type":"update"}],
            style={'font-size':'0.9em'},
            class_name='mb-1')[0]
        return output

@callback_extension(
    Output({'type':'figure_wrapper','tab1_figure_tabs':MATCH},'children'),
    Input('tab1_figure_plot_trigger','data'), #trigger
    Input({'code':'p1t1','tab1_figure_tabs':MATCH,"parameters":ALL,
            "extra_arg":ALL},'value'),
    State('tab1_plot_tabs','active_tab'),
    State({'tab1_figure_tabs':MATCH,'type':'select_data',"data_type":ALL},'value'),
    State({'code':'p1t1',"tab1_figure_tabs":MATCH,"plot_type":ALL,'xaxis_type':ALL,
           "preset":ALL,"grid":ALL},'id'),
    #census_files
    State({'code':'p1t1','name':'birth_data',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'death_data',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'w_population',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'vn_population',"data_type":"census"},'data'),
    #server_files
    State({'code':'p1t1','name':'w_covid_data',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'w_vaccine_volume',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'w_meta_database',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_total_case',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_province_case',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_death_data',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_vaccine_vol',"data_type":"server"},'data'),
    State({'code':'p1t1','name':'vn_vaccine_dist',"data_type":"server"},'data'),
    #update_files
    State({'code':'p1t1','name':'w_covid_data_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'w_vaccine_volume_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'w_meta_database_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'vn_total_case_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'vn_province_case_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'vn_death_data_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'vn_vaccine_vol_latest',"data_type":"update"},'data'),
    State({'code':'p1t1','name':'vn_vaccine_dist_latest',"data_type":"update"},'data'),
    prevent_initial_call=True
    )
def tab1_plot(trigger,param_value,active_tab,database_name,item_id,
              c1,c2,c3,c4,
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
    
    if database_name=="world_demographic":
        df = c1.merge(c2).merge(c3) #combine dataframe for easy visualization
    elif database_name=='vn_population':
        df = c4
    elif database_name=='w_covid_data':
        df=s1
    elif database_name=='w_vaccine_volume':
        df=s2
    elif database_name=='w_meta_database':
        df=s3
    elif database_name=='vn_total_case':
        df=s4
    elif database_name=='vn_province_case':
        df=(s5.set_index(['provinceName']).T.drop(['id','population'])
            .reset_index(names='date'))
    elif database_name=='vn_death_data':
        df=s6
    elif database_name=='vn_vaccine_vol':
        df=s7
    elif database_name=='vn_vaccine_dist':
        df=s8
    elif database_name=='w_covid_data_latest':
        df=u1
    elif database_name=='w_vaccine_volume_latest':
        df=u2
    elif database_name=='w_meta_database_latest':
        df=u3
    elif database_name=='vn_total_case_latest':
        df=u4
    elif database_name=='vn_province_case_latest':
        df=(u5.set_index(['provinceName']).T.drop(['id','population'])
            .reset_index(names='date'))
    elif database_name=='vn_death_data_latest':
        df=u6
    elif database_name=='vn_vaccine_vol_latest':
        df=u7
    elif database_name=='vn_vaccine_dist_latest':
        df=u8
    
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
                              'tab1_figure_tabs':item_id[0]['tab1_figure_tabs'],
                              'plot_idx':item[0],
                              'xaxis_type':json.dumps(item[1][2]),
                              'plot_args':None if plot_handler==None 
                              else json.dumps(plot_handler)
                              }))
        else:
            plot_output.append(dmc.Col(children=
                dcc.Graph(figure=fig,
                          id={'code':item_id[0]['code'],
                              'tab1_figure_tabs':item_id[0]['tab1_figure_tabs'],
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
                              'tab1_figure_tabs':item_id[0]['tab1_figure_tabs'],
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

@callback(
    Output({'code':'p1t1','tab1_figure_tabs':MATCH,'plot_idx':MATCH,
            'xaxis_type':ALL,'plot_args':ALL},'figure'),
    Output({'code':'p1t1','tab1_figure_tabs':MATCH,'plot_idx':MATCH,
            'attr_name':ALL},'data'),
    Input({'code':'p1t1','tab1_figure_tabs':MATCH,'plot_idx':MATCH,
            'xaxis_type':ALL,'plot_args':ALL},'relayoutData'),
    State({'code':'p1t1','tab1_figure_tabs':MATCH,'plot_idx':MATCH,
            'xaxis_type':ALL,'plot_args':ALL},'figure'),
    State({'code':'p1t1','tab1_figure_tabs':MATCH,'plot_idx':MATCH,
            'attr_name':ALL},'data'),
    prevent_initial_call=True
)
def tab1_figure_rangeslider_handlding(relay_fig,figs,figs_attr):
    """
    Since the rangeslider of figure doesnt auto scale y axis according to the
    selected x range, this function handle this problem to a certain degree.
    """
    xaxis_type = json.loads(ctx.triggered_id['xaxis_type'])
    #check if data need resize axis range
    if figs_attr==[]:
        fig_output = [dash.no_update for _ in range(len(figs))]
        return fig_output,[]
    
    layout,frames = figs_attr
    fig=go.Figure(data=figs[0]['data'],layout=layout,frames=frames)
    
    #check if x&y axis data type are supported
    check_array = np.array(fig.data[0].y)
    if check_array.dtype.kind=="O":
        try:
            check_array = check_array.astype(np.float64)
        except:
            pass
    if ((np.issubdtype(check_array.dtype.kind, np.number)) 
        and (xaxis_type[0] in ['number','time'])):
        pass
    else:
        fig_output = [dash.no_update for _ in range(len(figs))]
        attr_output = [dash.no_update for _ in range(len(figs_attr))]
        return fig_output,attr_output
    
    for item in fig.data:
        if xaxis_type[0]=='number':
            x_range = pd.Series(item['x'])
        elif xaxis_type[0]=='time':
            x_range = pd.to_datetime(pd.Series(item['x']),format=xaxis_type[1])
        break
        
    if 'xaxis.range' in relay_fig[0].keys():
        if xaxis_type[0]=='number':
            minx_val = relay_fig[0]['xaxis.range'][0]
            maxx_val = relay_fig[0]['xaxis.range'][1]
        elif xaxis_type[0]=='time':
            minx_val = pd.to_datetime(relay_fig[0]['xaxis.range'][0],
                                      format=xaxis_type[1])
            maxx_val = pd.to_datetime(relay_fig[0]['xaxis.range'][1],
                                      format=xaxis_type[1])
    elif (('xaxis.range[0]' in relay_fig[0].keys()) 
          and ('xaxis.range[1]' in relay_fig[0].keys())):
        if xaxis_type[0]=='number':
            minx_val = relay_fig[0]['xaxis.range[0]']
            maxx_val = relay_fig[0]['xaxis.range[1]']
        elif xaxis_type[0]=='time':
            minx_val = pd.to_datetime(relay_fig[0]['xaxis.range[0]'],
                                      format=xaxis_type[1])
            maxx_val = pd.to_datetime(relay_fig[0]['xaxis.range[1]'],
                                      format=xaxis_type[1])
    elif 'xaxis.autorange' in relay_fig[0].keys():
        fig.layout['yaxis'].pop('range')
        fig.layout['xaxis'].pop('range')
        for item in relay_fig[0].items():
            fig.layout[item[0]]=item[1]
        return [fig],[fig.layout,fig.frames]
    else:
        for item in relay_fig[0].items():
            fig.layout[item[0]]=item[1]
        return [fig],[fig.layout,fig.frames]
    
    for key in ['xaxis.autorange','xaxis.showspikes']:
        try:
            fig.layout.pop(key)
        except:
            pass
    
    idx_range = x_range.loc[(x_range>=minx_val) & (x_range<=maxx_val)].index
    
    y_max = []
    y_min = []
    for item in fig.data:
        y_max.append(pd.Series(item['y'])[idx_range].max())
        y_min.append(pd.Series(item['y'])[idx_range].min())
    y_max = np.max(y_max)
    y_min = np.min(y_min)
    
    fig.update_xaxes(range=[minx_val,maxx_val],rangeslider_visible=True)
    fig.update_yaxes(range=[y_min-0.05*y_min,y_max + 0.1*y_max])
    return [fig],[fig.layout,fig.frames]

##update database process----------------------------------------------------

@callback(output=dict(
    update_result=Output('tab1_processing_alert','children'),
    option_colors=(Output('tab1_select_w_database','labelClassName'),
                   Output('tab1_select_w_meta_database','labelClassName'),
                   Output('tab1_select_vn_database','labelClassName'),
                   Output('tab1_select_vn_dist_database','labelClassName')),
    option_disable=(Output('tab1_select_w_database','options'),
                    Output('tab1_select_w_meta_database','options'),
                    Output('tab1_select_vn_database','options'),
                    Output('tab1_select_vn_dist_database','options')),
    buffer_modified = Output({'code':'p1t1','name':'buffer_state'},
                             'data'),
                      ),
          inputs=dict(initiator=Input("tab1_process_params","data"),
                      try_states=(State('tab1_download_w_database','value'),
                                  State('tab1_download_w_meta_database','value'),
                                  State('tab1_download_vn_database','value'),
                                  State('tab1_download_vn_dist_database','value')
                                  ),
                      option_states=(State('tab1_select_w_database',
                                          'options'),
                                    State('tab1_select_w_meta_database',
                                          'options'),
                                    State('tab1_select_vn_database',
                                          'options'),
                                    State('tab1_select_vn_dist_database',
                                          'options')),
                      buffer_state = State({'code':'p1t1','name':'buffer_state'},
                                           'data'),
                      vn_province_code = State({'code':'p1t1','name':'vn_province_code',
                                                "data_type":"internal"},'data'),
                      chosen_province = State('tab1_model_database_province_checklist',
                                              'value')
                      ),
            background=True,
            running=[(Output('tab1_processing_status','is_open'),True,False)],
            progress=[Output('tab1_update_process_collapse','is_open'),
                      Output('tab1_update_process','value'),
                      Output('tab1_update_process_info','children')],
            progress_default=[False,0,'Updating...'],
            cancel=Input('tab1_model_Cancel_Button1','n_clicks'),
            prevent_initial_call=True)
def t1_update_database(
                      progress_status,
                      initiator,
                      try_states,
                      option_states,
                      buffer_state,
                      vn_province_code,
                      chosen_province
                      ):
    
    """
    Attempt to update databases to latest version depending on combination of
    input parameters.
    """
    directory = FileSystemStore(cache_dir="./cache/output_cache",
                                default_timeout=0)
    packed_params = directory.get(initiator)
    (parameter_check,output_uid,driver_ready,webdriver_path,max_processes,
     wait_time,attempts,interval)=packed_params['general_params']
    
    result_color = []
    result_disable = []
    result_buffer_stt = buffer_state.copy()
    if try_states[0]:
        if not buffer_state[0]:
            try:
                time.sleep(0.5)
                progress_status((True,0,'Updating world covid database...'))
                time.sleep(1)
                w_covid_data_latest = pd.read_csv(
                    'https://raw.githubusercontent.com/owid/covid-19-data/'
                    +'master/public/data/owid-covid-data.csv',
                    encoding='utf-8',
                    usecols=['iso_code','continent','location','date',
                              'new_cases','new_deaths',
                              'new_vaccinations_smoothed','population'])
                w_covid_data_latest.date = pd.to_datetime(w_covid_data_latest.date)
                w_vaccine_volume_latest = pd.read_csv(
                    'https://raw.githubusercontent.com/owid/covid-19-data/'
                    +'master/public/data/vaccinations/'
                    +'vaccinations-by-manufacturer.csv',
                    encoding='utf-8')
                result_color.append('btn btn-outline-success')
                option_states[0][1]['disabled']=False
                result_disable.append(option_states[0])
                directory.set(output_uid[0],w_covid_data_latest)
                directory.set(output_uid[1],w_vaccine_volume_latest)
                result_buffer_stt[0]=True
                progress_status((True,100,'Updated world covid database.'))
                time.sleep(1)
            except:
                result_color.append('btn btn-outline-warning')
                option_states[0][1]['disabled']=True
                result_disable.append(option_states[0])
                progress_status((True,100,'Failed to update world covid database.'))
                time.sleep(1)
        else:
            result_color.append(dash.no_update)
            result_disable.append(dash.no_update)
    else:
        result_color.append(dash.no_update)
        result_disable.append(dash.no_update)
    
    if try_states[1]:
        if not buffer_state[1]:
            try:
                progress_status((True,0,'Updating World vaccine meta database...'))
                time.sleep(1)
                w_meta_database_latest = pd.read_csv(
                    'https://covid19.who.int/who-data/vaccination-metadata.csv',
                    encoding='utf-8')
                result_color.append('btn btn-outline-success')
                option_states[1][1]['disabled']=False
                result_disable.append(option_states[1])
                directory.set(output_uid[2],w_meta_database_latest)
                result_buffer_stt[1]=True
                progress_status((True,100,'Updated World vaccine meta database.'))
                time.sleep(1)
            except:
                result_color.append('btn btn-outline-warning')
                option_states[1][1]['disabled']=True
                result_disable.append(option_states[1])
                progress_status((True,100,'Failed to update World vaccine meta database.'))
                time.sleep(1)
        else:
            result_color.append(dash.no_update)
            result_disable.append(dash.no_update)
    else:
        result_color.append(dash.no_update)
        result_disable.append(dash.no_update)
    
    if try_states[2]:
        try:
            progress_status((False,0,'Updating Vietnam covid database...'))
            time.sleep(1)
            request_obj = browser_request(wait_time=wait_time,
                                          attempt=attempts,
                                          interval=interval)
            ##total case
            if directory.get(output_uid[3]) is None:
                progress_status((True,0,'Updating Vietnam total case...'))
                time.sleep(1)
                
                link = "https://ncov.vncdc.gov.vn/"
                api_country_case = ("https://ncov.vncdc.gov.vn/v2/"+
                "vietnam/report-epi?")
                search_scope = ['https://ncov.vncdc.gov.vn/v2/']
                
                json_object = request_obj.get_web_response(website_link=link,
                                                            api=api_country_case,
                                                            search_scope=search_scope,
                                                            driver_path=webdriver_path)
                df_total_case = pd.json_normalize(json_object,
                                                  record_path=['report','data'],
                                                  meta=[['report','name'],
                                                        ['report','code']])
                df_total_case[0] = pd.to_datetime(df_total_case[0],
                                                  unit='ms').dt.date
                df_total_case.rename(columns={0:"date",1:"case",
                                              "report.name":"case_origin",
                                              "report.code":"code"},
                                      inplace=True)
                df_total_case.loc[df_total_case.code=="avg_case",
                                  'code']="7_days_avg_case"
                
                #for some reason the date keep move back 1 day if you request 
                #the data at late night
                if (df_total_case.date.head(1)==datetime.date(2020,1,22)).all():  
                    df_total_case.date=(df_total_case.date
                                        +datetime.timedelta(1))
                df_total_case.date = pd.to_datetime(df_total_case.date)
                directory.set(output_uid[3],df_total_case)
                progress_status((True,100,'Updated Vietnam total case.'))
                time.sleep(1)
            
            ##province_case
            if directory.get(output_uid[4]) is None:
                progress_status((True,0,'Updating Vietnam province case...'))
                time.sleep(1)
                
                link = ("https://ncov.vncdc.gov.vn/viet-nam-full.html?"+
                        "startTime=2020-01-01&endTime=&provinces=&districts=&"+
                        "tabKey=0")
                api_province_case = ("https://ncov.vncdc.gov.vn/v2/"+
                                      "vietnam/by-current?")
                search_scope = ['https://ncov.vncdc.gov.vn/v2/']
                
                json_object = request_obj.get_web_response(website_link=link,
                                                            api=api_province_case,
                                                            search_scope=search_scope,
                                                            driver_path=webdriver_path)  
        
                df_province_case = pd.json_normalize(json_object)
                #remove path of nested reports from column name
                df_province_case.columns = (df_province_case.columns
                                            .str.replace('data.','',
                                                          regex=True))
        
                #Synchronize all province name with other dataframe
                list1 = (df_province_case.tinh
                          .apply(lambda x:str.replace(x,'Thành phố ',''))
                          .apply(lambda x:str.replace(x,'Tỉnh ',''))
                          .apply(lambda x:str.replace(x,'Hồ Chí Minh',
                                                      'TP. Hồ Chí Minh'))
                          .to_list())
                df_province_case.tinh = list1
                df_province_case.loc[df_province_case.tinh=='Hoà Bình',
                                      'tinh']='Hòa Bình'
                df_province_case=df_province_case.loc[~(df_province_case.tinh=="Chưa rõ")]
                df_province_case.sort_values(by='tinh',inplace=True,
                                              ignore_index=True,kind = 'stable')
                #drop some columns to keep data consistently as time sery
                df_province_case.drop(df_province_case.columns[[2,3,4,6,7,8,9,
                                                                10,11]],
                                      axis=1,
                                      inplace=True)
                df_province_case.rename(columns={"tinh":"provinceName"},
                                        inplace=True)
                directory.set(output_uid[4],df_province_case)
                progress_status((True,100,'Updated Vietnam province case.'))
                time.sleep(1)
                
            ##province death case
            progress_status((True,0,'Updating Vietnam total death case...'))
            time.sleep(1)
            
            link = ("https://ncov.vncdc.gov.vn/viet-nam-full.html?start"
                    +"Time=2020-01-01&endTime=&provinces=&districts=&tabKey=0")
            api_total_death = "https://ncov.vncdc.gov.vn/v2/vietnam/report-epi-5?"
            search_scope = ['https://ncov.vncdc.gov.vn/v2/']
            
            json_object = request_obj.get_web_response(website_link=link,
                                                        api=api_total_death,
                                                        search_scope=search_scope,
                                                        driver_path=webdriver_path) 
        
            df1 = pd.json_normalize(json_object['report'][2],
                                    record_path = 'data')
            df1[0] = pd.to_datetime(df1[0],unit='ms').dt.date
            df1.rename(columns={0:"date",1:"VN_death_case"},inplace=True)
    
            df2 = pd.json_normalize(json_object['report'][4],
                                    record_path = 'data')
            df2[0] = pd.to_datetime(df2[0],unit='ms').dt.date
            df2.rename(columns={0:"date",1:"VN_days_death_avg"},inplace=True)

            df_cumulative_death=df1.merge(df2,on='date',how='outer')
            
            progress_status((True,100,'Updated Vietnam total death case.'))
            time.sleep(1)
            #processing
            progress_status((True,0,'Updating Vietnam province death case...'))
            time.sleep(1)
            
            link = ("https://ncov.vncdc.gov.vn/viet-nam-full.html?startTime"
                    +"=2020-01-01&endTime=&provinces={}&districts=&tabKey=0")
            api_province_death = "https://ncov.vncdc.gov.vn/v2/vietnam/report-epi-5?"
            search_scope = ['https://ncov.vncdc.gov.vn/v2/']
            vn_max_worker = max_processes
            
            vn_manager = Manager()
            vn_counter= vn_manager.Array('i',[0,0])
            vn_time_list = vn_manager.list([])
            vn_error_dict = vn_manager.dict()
            vn_shared_output = vn_manager.Namespace()
            vn_shared_output.shared_df = pd.DataFrame({'date':[]})
            vn_lock = vn_manager.Lock()
            
            used_code = [item for item in vn_province_code if item[1] in
                          chosen_province]
            worker_func = partial(vn_casualty_wrapper_func,
                              shared_count=vn_counter,
                              time_list=vn_time_list,
                              error_dict=vn_error_dict,
                              shared_output=vn_shared_output,
                              lock=vn_lock,
                              request_object=request_obj,
                              link = link,
                              api = api_province_death,
                              search_scope = search_scope)
            executor= Pool(processes=vn_max_worker)
            pool_result = executor.map_async(worker_func,used_code)
            
            max_province = len(used_code)
            while vn_counter[0]<max_province:
                time.sleep(1)
                n_done = vn_counter[0]
                n_wait = max_province
                total_time = round(sum(vn_time_list),2)
                if n_done!=0:
                    avg_time = round(total_time/n_done,2)

                if n_done!=0:
                    remain_province = n_wait - n_done
                    remain_time = round(remain_province*avg_time,2)
                    real_remain = round(remain_time/vn_max_worker,2)
                else:
                    real_remain = "-- "
                progress_status((True,100*n_done/max_province,
                                 ('Updating Vietnam province death '
                                 +f'case: {real_remain}s')))
            result_df = vn_shared_output.shared_df.copy()
            sorted_col = np.append(['date'],
                                    np.sort(result_df.columns[1:],
                                            kind='stable'))
            result_df = result_df[sorted_col]
            df_cumulative_death = df_cumulative_death.merge(result_df,
                                                            on="date",
                                                            how='outer')
            executor.close()
            executor.join()
            vn_manager.shutdown()
            
            #format province death case
            df_province_death=df_cumulative_death.fillna(0)
            if (df_province_death.date.head(1)==datetime.date(2020,1,22)).all():
                df_province_death.date=df_province_death.date+datetime.timedelta(1)
            df_province_death[df_province_death.columns[1:]]=\
                df_province_death[df_province_death.columns[1:]].astype(int)
            df_province_death.drop_duplicates(inplace=True,
                                              ignore_index=True,
                                              subset=["date","VN_death_case",
                                                      "VN_days_death_avg"])
            df_province_death.date = pd.to_datetime(df_province_death.date)
            directory.set(output_uid[5],df_province_death)
            progress_status((True,100,'Updated Vietnam province death case.'))
            time.sleep(1)
            ##vaccine volume
            progress_status((True,0,'Updating Vietnam vaccine volume...'))
            time.sleep(1)
            if directory.get(output_uid[6]) is None:
                if (directory.get(output_uid[0]) is not None):
                    df = directory.get(output_uid[0])
                    vn_vaccine_vol_latest = (df
                                .loc[directory.get(output_uid[0]).location=="Vietnam",
                                      ["date","new_vaccinations_smoothed"]]
                                .reset_index(drop=True))
                else:
                    vn_vaccine_vol_latest = pd.read_csv(
                        ('https://raw.githubusercontent.com/owid/covid-19-data/'
                        +'master/public/data/owid-covid-data.csv'),
                        encoding='utf-8',
                        usecols=['location','date',
                                  'new_vaccinations_smoothed'])
                    vn_vaccine_vol_latest = vn_vaccine_vol_latest.loc[
                        vn_vaccine_vol_latest.location=="Vietnam",
                        ["date","new_vaccinations_smoothed"]]
                vn_vaccine_vol_latest.date = pd.to_datetime(vn_vaccine_vol_latest.date)
                directory.set(output_uid[6],vn_vaccine_vol_latest)
                progress_status((True,100,'Updated Vietnam vaccine volume.'))
                time.sleep(1)
                
            result_color.append('btn btn-outline-success')
            option_states[2][1]['disabled']=False
            result_disable.append(option_states[2])
            progress_status((True,100,'Updated Vietnam covid database.'))
            time.sleep(1)
            
        except:
            result_color.append('btn btn-outline-warning')
            option_states[2][1]['disabled']=True
            result_disable.append(option_states[2])
            progress_status((True,100,'Failed to update all Vietnam covid database.'))
            time.sleep(1)
        
    else:
        result_color.append(dash.no_update)
        result_disable.append(dash.no_update)
    if try_states[3]:
        if not buffer_state[3]:
            try:
                progress_status((True,0,'Updating Vietnam vaccine distribution database...'))
                time.sleep(1)
                request_data = requests.get('https://tiemchungcovid19.gov.vn/'+
                              'api/public/dashboard/vaccination-statistics/all')

                #format data
                vn_vaccine_dist_latest=pd.json_normalize(request_data.json())
                vn_vaccine_dist_latest=vn_vaccine_dist_latest[
                    ['provinceName','population','popOverEighteen',
                      'totalOnceInjected','totalTwiceInjected','totalInjected',
                      'totalVaccineAllocated','totalVaccineAllocatedReality']]
                vn_vaccine_dist_latest.loc[
                    vn_vaccine_dist_latest.provinceName=='Hồ Chí Minh',
                    'provinceName']='TP. Hồ Chí Minh'
                vn_vaccine_dist_latest.loc[
                    vn_vaccine_dist_latest.provinceName=='Hoà Bình',
                    'provinceName']='Hòa Bình'
                vn_vaccine_dist_latest.sort_values('provinceName',
                                                    inplace=True,
                                                    ignore_index=True,
                                                    kind = 'stable')
                result_color.append('btn btn-outline-success')
                option_states[3][1]['disabled']=False
                result_disable.append(option_states[3])
                directory.set(output_uid[7],vn_vaccine_dist_latest)
                result_buffer_stt[3]=True
                progress_status((True,100,'Updated Vietnam vaccine distribution database.'))
                time.sleep(1)
            except:
                result_color.append('btn btn-outline-warning')
                option_states[3][1]['disabled']=True
                result_disable.append(option_states[3])
                progress_status((True,100,'Failed to update Vietnam vaccine distribution database.'))
                time.sleep(1)
        else:
            result_color.append(dash.no_update)
            result_disable.append(dash.no_update)
    else:
        result_color.append(dash.no_update)
        result_disable.append(dash.no_update)
    progress_status((False,100,'Attempt to update database completed.'))
    time.sleep(1)
    return dict(update_result='Update process completed.',
                option_colors=tuple(result_color),
                option_disable=tuple(result_disable),
                buffer_modified=(result_buffer_stt 
                if result_buffer_stt!=buffer_state else
                dash.no_update)
                )

@callback_extension(
    output=ServersideOutput('tab1_process_params','data'),
    inputs=dict(
        update_click=Input('tab1_model_Update_Button','n_clicks'),
        parameter_check = State('tab1_inputs_alert','is_open'),
        output_uid = State({'code':'p1t1','name':ALL,"data_type":"update"},'data'),
        driver_ready=State({'code':'p1t1','name':'driver_ready',
                            "data_type":"input_check"},'data'),
        webdriver_path=State('tab1_webdriver_path','value'),
        max_processes = State({"code":"p1t1","name":"tab1_max_worker",
                               "type":"input"},'value'),
        wait_time = State({"code":"p1t1","name":"tab1_max_wait_time",
                           "type":"input"},'value'),
        attempts = State({"code":"p1t1","name":"tab1_max_attempt",
                          "type":"input"},'value'),
        interval = State({"code":"p1t1","name":"tab1_interval_time",
                          "type":"input"},'value')
        ),
    prevent_initial_call=True
    )
def tab1_update_params_prepare(update_click,parameter_check,output_uid,
                               driver_ready,webdriver_path,max_processes,wait_time,
                               attempts,interval):
    """
    Due to some kind of limit to background callback dictionary inputs, if you 
    use many keys, the function 'running' argument won't load quickly.
    This function is used to handle that limitation.
    """
    if parameter_check:
        return dash.no_update
    else:
        outputs = dict(
            general_params=[parameter_check,output_uid,driver_ready,
                            webdriver_path,max_processes,wait_time,
                            attempts,interval])
        return outputs

@callback(
    Output('tab1_processing_alert_collapse','is_open'),
    Input('tab1_processing_alert','children'),
    Input('tab1_model_Update_Button','n_clicks'),
    Input('tab1_model_Download_Button','n_clicks'),
    prevent_initial_call=True
    )
def tab1_display_process_result(trigger1,trigger2,trigger3):
    """
    Show processing result.
    """
    if ctx.triggered_id=="tab1_processing_alert":
        return True
    elif ctx.triggered_id in ["tab1_model_Update_Button",
                              "tab1_model_Download_Button"]:
        return False
    
##download data-------------------------------------------------------------

@callback(
    output=[
        Output('tab1_download','data'),
        ],
    inputs = [
        Input('tab1_model_Download_Button','n_clicks'),
        #census_files
        State({'code':'p1t1','name':ALL,"data_type":"census"},'data'),
        #server_files
        State({'code':'p1t1','name':ALL,"data_type":"server"},'data'),
        #update_files
        State({'code':'p1t1','name':ALL,"data_type":"update"},'data'),
        ],
    background=True,
    running=[(Output('tab1_downloading_status','is_open'),True,False)],
    cancel=Input('tab1_model_Cancel_Button2','n_clicks'),
    prevent_initial_call=True)
def tab1_download_zip(trigger,census_uid,server_uid,updated_uid):
    """
    Download databases to local machine
    """
    def write_available_data(bytes_io,census_uid,server_uid,updated_uid):
        """
        Function that handle zipping files
        """
        census_id_list = []
        server_id_list = []
        updated_id_list = []
        for item in dash.callback_context.states_list[0]:
            census_id_list.append(item['id']['name'])
        for item in dash.callback_context.states_list[1]:
            server_id_list.append(item['id']['name'])  
        for item in dash.callback_context.states_list[2]:
            updated_id_list.append(item['id']['name'])  
        with ZipFile(bytes_io, mode="w") as zf:
            directory = FileSystemStore(cache_dir="./cache/census_cache",
                                        default_timeout=0)
            for item in zip(census_id_list,census_uid):
                data = directory.get(item[1])
                if data is not None:
                    filepath = f"census/{item[0]}.csv"
                    zf.writestr(filepath,
                                data.to_csv(index=False))
            directory = FileSystemStore(cache_dir="./cache/server_cache",
                                        default_timeout=0)
            for item in zip(server_id_list,server_uid):
                data = directory.get(item[1])
                if data is not None:
                    try:
                        filepath = f"default/{item[0]}.csv"
                        zf.writestr(filepath,
                                    data.to_csv(index=False))
                    except:
                        df = pd.DataFrame(data)
                        filepath = f"default/{item[0]}.csv"
                        zf.writestr(filepath,df.to_csv(index=False))
            directory = FileSystemStore(cache_dir="./cache/output_cache",
                                        default_timeout=0)
            for item in zip(updated_id_list,updated_uid):
                data = directory.get(item[1])
                if data is not None:
                    filepath = f"latest/{item[0]}.csv"
                    zf.writestr(filepath,
                                data.to_csv(index=False))
    return [dcc.send_bytes(write_available_data,"raw_data.zip",
                          census_uid=census_uid,server_uid=server_uid,
                          updated_uid=updated_uid)]