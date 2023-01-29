import pandas as pd
import numpy as np
import dash
from dash import callback,ctx
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_extensions.enrich import Input,Output,State,ServersideOutput,dcc,\
    html,ALL,MATCH,FileSystemStore
from dash_extensions.enrich import callback as callback_extension
import time,datetime,json,os
import re
from multiprocessing import Manager,Pool
from functools import partial
from zipfile import ZipFile
from covid_science.workers import wrapper_SVID_predict
from covid_science import utility_func
from covid_science.modeling import get_arima_order

#------------------------------------------------------------------

#Tab3: Modeling data
tab3_stored_data = html.Div([
    #check criteria,True mean ok
    dcc.Store(id={'code':'p1t3','name':'normal_input',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t3','name':'b_d_rate',"data_type":"input_check"},
              data=True),
    dcc.Store(id={'code':'p1t3','name':'tab3_model_number',"data_type":"input_check"},
              data=True),
    dcc.Store(id='tab3_initial_status',data=0),
    
    #preloaded dataframe
    dcc.Store(id={'code':'p1t3','name':'vn_prediction',"data_type":"server"}),
    dcc.Store(id={'code':'p1t3','name':'vn_prediction_by_province',"data_type":"server"}),
    dcc.Store(id={'code':'p1t3','name':'vn_equilibrium',"data_type":"server"}),
    dcc.Store(id={'code':'p1t3','name':'vn_equilibrium_by_province',"data_type":"server"}),
    dcc.Store(id={'code':'p1t3','name':'w_prediction',"data_type":"server"}),
    dcc.Store(id={'code':'p1t3','name':'w_prediction_by_country',"data_type":"server"}),
    dcc.Store(id={'code':'p1t3','name':'w_equilibrium',"data_type":"server"}),
    dcc.Store(id={'code':'p1t3','name':'w_equilibrium_by_country',"data_type":"server"}),
    
    #modified dataframe
    dcc.Store(id={'code':'p1t3','name':'vn_prediction_mod',"data_type":"update"}),
    dcc.Store(id={'code':'p1t3','name':'vn_prediction_by_province_mod',"data_type":"update"}),
    dcc.Store(id={'code':'p1t3','name':'vn_equilibrium_mod',"data_type":"update"}),
    dcc.Store(id={'code':'p1t3','name':'vn_equilibrium_by_province_mod',"data_type":"update"}),
    dcc.Store(id={'code':'p1t3','name':'w_prediction_mod',"data_type":"update"}),
    dcc.Store(id={'code':'p1t3','name':'w_prediction_by_country_mod',"data_type":"update"}),
    dcc.Store(id={'code':'p1t3','name':'w_equilibrium_mod',"data_type":"update"}),
    dcc.Store(id={'code':'p1t3','name':'w_equilibrium_by_country_mod',"data_type":"update"}),
    
    dcc.Store(id={'code':'p1t3','name':'birth_data',"data_type":"census"}),
    dcc.Store(id={'code':'p1t3','name':'death_data',"data_type":"census"}),
    
    #buffer dataframe, stored modified dataframe from tab2 if it's changed
    dcc.Store({'code':'p1t3','name':'vn_modeling_data_mod',"data_type":"buffer"}),
    dcc.Store({'code':'p1t3','name':'vn_modeling_data_by_province_mod',"data_type":"buffer"}),
    dcc.Store({'code':'p1t3','name':'w_modeling_data_mod',"data_type":"buffer"}),
    dcc.Store({'code':'p1t3','name':'w_modeling_data_by_country_mod',"data_type":"buffer"}),
    
    #plot triggering
    dcc.Store(id='tab3_figure_plot_trigger',data=0),
    dcc.Store(id={'tab3_figure_tabs':'t1','name':'tab3_figure_plot_trigger_step1'},
              data=0),
    dcc.Store(id={'tab3_figure_tabs':'t2','name':'tab3_figure_plot_trigger_step1'},
              data=0),
    dcc.Store(id={'tab3_figure_tabs':'t1','name':'tab3_figure_plot_trigger_step2'},
              data=0),
    dcc.Store(id={'tab3_figure_tabs':'t2','name':'tab3_figure_plot_trigger_step2'},
              data=0),
    dcc.Store(id='tab3_figure_dropdown_trigger',data=0),
    
    #store extra forecast argument
    dcc.Store(id='tab3_forecast_extra_args',data=None),
    dcc.Store(id='tab3_preset_orders',data='auto'),
    
    #store processing parameters
    dcc.Store(id="tab3_process_params")
])

tab3_content = [dbc.Form([tab3_stored_data,
    dbc.Alert("Please fill in all required parameters.",
              id='tab3_inputs_alert',is_open=False,color='danger',
              class_name="c_alert"),
    dcc.Interval(id='tab3_data_initiator',max_intervals=1,n_intervals=0,
                 disabled=True),
    dbc.Label(html.Li("Color code:"),class_name="ms-1",width='auto',
              id='tab3_color_code_label'),
    dbc.Row([dbc.Col([dbc.Button("Default",size='sm',
                                 style = {'font-size':'0.8em'}),
                      dbc.Button("Data modeling unsuccessful",
                                 size='sm',color="warning",
                                 style = {'font-size':'0.8em'}),
                      dbc.Button("Data modeling successful",size='sm',
                                 color="success",
                                 style = {'font-size':'0.8em'}),],
                     width='auto')],
            justify='start'),
    dbc.Label(html.Li('Model database result:'),class_name="ms-1",width='auto',
              id='tab3_model_select_label'),
    dbc.Label('Vietnam:',width=7,style={'font-size':'0.8em'}),
    html.Div(dbc.RadioItems(
            id={'code':'p1t3','name':'select_vn_database',"purpose":"database_check"},
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Process status", "value": 1},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    dbc.Label('Each Vietnam\'s province:',width=7,style={'font-size':'0.8em'}),
    html.Div(dbc.RadioItems(
            id={'code':'p1t3','name':'select_vnbp_database',"purpose":"database_check"},
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Process status", "value": 1},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    dbc.Label('World:',width=7,style={'font-size':'0.8em'}),
    html.Div(dbc.RadioItems(
            id={'code':'p1t3','name':'select_w_database',"purpose":"database_check"},
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Process status", "value": 1},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    dbc.Label('Each country:',width=7,style={'font-size':'0.8em'}),
    html.Div(dbc.RadioItems(
            id={'code':'p1t3','name':'select_wbc_database',"purpose":"database_check"},
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Process status", "value": 1},
            ],
            label_style = {'font-size':'0.8em'},
            value=1,
        ),
        className="radio-group"),
    html.Hr(),
    dbc.Label(html.Li('Modeling database:'),class_name="ms-1",width='auto',
              id='tab3_model_database_label'),
    dbc.Switch(id="tab3_model_database_all_switch",
               label_id="tab3_model_database_all_switch_label",
               label="Use all databases",
               value=False,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    
    dbc.Collapse([
    dbc.Row(dbc.RadioItems(options=[{"label": 'VietNam', "value": 'vietnam'},
                                    {"label":'World',"value":'world'}],
                           id='tab3_model_database',
                           value='vietnam',
                           label_style = {'font-size':'0.8em'},
                           inline=True)
            ),
    dbc.Switch(id="tab3_model_database_division_switch",
               label="Use database by provinces",
               value=False,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"),
    dbc.Collapse(dbc.Button("Choose section",size='sm',className="ms-auto",
                            id="tab3_model_database_division_button",
                            style = {'font-size':'0.8em'}),
                 id='tab3_model_database_division_button_collapse',
                 is_open=False)],
        id='tab3_model_database_all_switch_collapse',
        is_open=True),
    
    html.Hr(),
    dbc.Label(html.Li('Census parameters:'),class_name="ms-1",width='auto',
              id='tab3_census_label'),
    
    dbc.Row([dbc.Label('Recruit non-vaccinated ratio:',width=7,
                       id='tab3_p_label',style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t3","name":"tab3_p","type":"input"},
                               type='number',size='sm',required=True,
                               min=0.0,max=1.0,step=0.01,value=1.0,
                               className="mt-1 mb-1",
                               persistence_type="memory",
                               persistence=True),
                     width=4)
             ]),
    
    dbc.Switch(id="tab3_auto_b_d_rate",
               label="Auto calculate birth & death rate from database",
               label_id="tab3_auto_b_d_rate_label",
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
            id="tab3_range_b_d_rate")],
        id='tab3_range_b_d_rate_collapse',
        is_open=True),
    dbc.Collapse(
        [dbc.Row([
            dbc.Label(
                'Birth rate per day:',width=7,
                id='tab3_manual_b_rate_label',
                style={'font-size':'0.8em'}),
            dbc.Col(dbc.Input(
                id={"code":"p1t3","name":"tab3_manual_b_d_rate",
                    "type":"special_input","index":0},
                type='number',required=True,size='sm',
                min=0,max=1,step=0.0000001,value=0.000047,
                className="mt-1 mb-1"),
                    width=4)]),
        dbc.Row([
            dbc.Label(
                'Death rate per day:',width=7,id='tab3_manual_d_rate_label',
                style={'font-size':'0.8em'}),
            dbc.Col(dbc.Input(
                id={"code":"p1t3","name":"tab3_manual_b_d_rate",
                    "type":"special_input","index":1},
                type='number',required=True,size='sm',
                min=0,max=1,step=0.0000001,value=0.0000221,
                className="mt-1 mb-1"),
                    width=4)]),
        ],
        id='tab3_manual_b_d_rate_collapse',
        is_open=False),
    
    html.Hr(),
    dbc.Label(html.Li("Forecasting inputs:"),class_name="ms-1",
              id="tab3_forecast_label",width='12'),
    
    dbc.Label("Start date:",class_name="mb-0",style={'font-size':'0.8em'},
              width='auto',id='tab3_start_date_label'),
    dbc.Row([dbc.Col(dbc.RadioItems(options=[
                        {"label": 'Auto', "value": 'auto'},
                        {"label":'Specific date:',
                         "value":str(datetime.datetime.now().date()-
                                     datetime.timedelta(1))}],#not determined
                                    id='tab3_start_date',
                                    value='auto',
                                    label_style = {'font-size':'0.8em'}),
                    width=7),
            dbc.Col(dmc.DatePicker(
                   id="tab3_start_date_select",
                   minDate=datetime.date(2019, 12, 1),
                   maxDate=datetime.datetime.now().date(),
                   hideOutsideDates=True,
                   clearable=False,
                   value=datetime.datetime.now().date()-datetime.timedelta(1),
                   inputFormat="YYYY-MM-DD",
                   amountOfMonths=1,
                   size='xs'),
                    width=4,align="end")
            
            ]),
    
    dbc.Row([dbc.Label('Number of forecast days:',width=7,id='tab3_days_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t3","name":"tab3_days","type":"input"},
                               type='number',size='sm',
                               min=1,step=1,value=7,required=True,
                               className="mt-1 mb-1"),
                     width=4)
             ]),
    
    dbc.Label('Forecast model:',width='auto',style={'font-size':'0.8em'},
              id='tab3_model_number_label'),
    dbc.Checklist(options=[{"label": "Model 1", "value": 1},
                           {"label": "Model 2", "value": 2},
                           {"label": "Model 3", "value": 3}
                            ],
                  value=[1,2,3],
                  id="tab3_model_number",
                  inline=True,
                  label_style = {'font-size':'0.8em'},
                  class_name='mx-1 border border-1 border-secondary'
                  ),
    dbc.Alert("Must choose at least 1 model.",id='tab3_model_number_alert',
              is_open=False,color="danger",class_name="c_alert"),
    
    dbc.Switch(id="tab3_best_model",
               label_id="tab3_best_model_label",
               label="Choose best forecast model",
               value=True,
               style = {'font-size':'0.8em'},
               className="mb-0 mt-2"
                ),
    
    dbc.Label('Scoring method:',width='auto',style={'font-size':'0.8em'},
              id='tab3_scoring_label'),
    dbc.Col(dbc.Select(id='tab3_scoring',options=
                                       [{'label':'Mean Absolute Percentage Error',
                                         'value':'neg_mean_absolute_percentage_error'},
                                        {'label':'Mean Absolute Error',
                                         'value':'neg_mean_absolute_error'},
                                        {'label':'Mean Squared Log Error',
                                         'value':'neg_mean_squared_log_error'},
                                        {'label':'Root Mean Squared Error',
                                         'value':'neg_root_mean_squared_error'}],
                        value='neg_mean_absolute_percentage_error',
                        style = {'font-size':'0.8em'}),width=9),
    
    dbc.Label("Model scoring criteria:",
              id='tab3_model_criteria_label',
              class_name="mb-0 mt-2",
              style={'font-size':'0.8em'}),
    dbc.Row(dbc.RadioItems(
        options=[{"label": 'Bigger', "value": True},
                 {"label":'Smaller',"value":False}],
        id='tab3_model_criteria',
        value=False,
        label_style = {'font-size':'0.8em'},
        inline=True)),
    
    dbc.Label("Scoring output:",class_name="mb-0 mt-1",width='auto',
              style={'font-size':'0.8em'},id='tab3_multioutput_label'),
    dbc.Row(dbc.RadioItems(options=[
                        {"label": 'Raw values', "value": 'raw_values'},
                        {"label":'Variance weight',"value":'variance_weighted'},
                        {"label":'Uniform average',"value":'uniform_average'}],
                                    id='tab3_multioutput',
                                    value='variance_weighted',
                                    label_style = {'font-size':'0.8em'})
            ),
    
    dbc.Row([dbc.Label('Immunity time:',width=7,id='tab3_r_protect_time_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t3","name":"tab3_r_protect_time","type":"input"},
                               type='number',size='sm',
                               min=1,step=1,value=150,required=True,
                               className="mt-1 mb-1"),
                     width=4)
             ]),
    
    dbc.Row([dbc.Label('Equilibrium plot points:',width=7,
                       id='tab3_n_days_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Input(id={"code":"p1t3","name":"tab3_n_days","type":"input"},
                               type='number',size='sm',
                               min=1,step=1,value=1000,required=True,
                               className="mt-1 mb-1"),
                     width=4)
             ]),

    dbc.Row(dbc.Col(
        dbc.Button('Extra forecast argument',id='tab3_forecast_ex_button',size='sm',
                   className="ms-auto",style = {'font-size':'0.8em'}),
        width=6)),
        
    
    dbc.Label(html.Li('Other parameters:'),class_name="ms-1",width='auto',
              id='tab3_other_param_label'),
    dbc.Row([
        dbc.Label(
            'Maximum processes:',
            width=7,id='tab3_max_worker_label',
            style={'font-size':'0.8em'}),
        dbc.Col(dbc.Input(
            id={"code":"p1t3","name":"tab3_max_worker","type":"input"},
            type='number',required=True,
            size='sm',min=1,step=1,value=round(os.cpu_count()/2)+1,
            className="mt-1 mb-1"),
                width=4)]),
    dbc.Collapse([dbc.Alert(id='tab3_processing_alert',className='c_alert')],
                 id='tab3_processing_alert_collapse',
                 is_open=False,),
    dcc.Download(id="tab3_download"),
    
    ],
    className="mt-1 mb-1 bg-secondary",
    style={'color':'white'}),
    
    dbc.Row([dbc.Col([
        dbc.Button("Apply",id="tab3_model_Apply_Button",
                   n_clicks=0,size='sm'),
        dbc.Button("Download",id="tab3_model_Download_Button",
                   n_clicks=0,size='sm'),
        ],width='auto')],
        justify='end')
    ]

#Tab3 plots

tab3_figure = html.Div([
    dbc.Tabs([
    dbc.Tab(
        [utility_func.add_row_choices(
            ['Database:'], 
            [[{"label":"Vietnam prediction data","value":"vn_prediction"},
              {"label":"Vietnam prediction data by province",
               "value":"vn_prediction_by_province"},
              {"label":"World prediction data","value":"w_prediction"},
              {"label":"World prediction data by country",
               "value":"w_prediction_by_country"}],
             ], 
            ['vn_prediction'], 
            [{"tab3_figure_tabs":'t1','type':'select_data',"data_type":"server"}],
            style={'font-size':'0.9em'},
            class_name='mb-1')[0],
        dbc.Col([dbc.Label('Plot points:',id='tab3_plot_point_label1',
                           style={'font-size':'0.9em'},width='auto'),
                 dbc.Col(dbc.Input(id={"tab3_figure_tabs":'t1','type':'plot_point'},
                                   type='number',size='sm',
                                   min=2,step=1,value=30,required=True,
                                   className="mt-1 mb-1"),
                         width=2)
                 ]),
        ],
        label='Server data',
        id={"tab3_figure_tabs":'t1'},
        ),
    dbc.Tab(["No data",
             #this collapse only for avoid error in callback due to no id exist
             dbc.Collapse(
                 dbc.Input(id={"tab3_figure_tabs":'t2','type':'plot_point'},
                                   type='number',size='sm',
                                   min=2,step=1,value=30,required=True,
                                   className="mt-1 mb-1"),
                 is_open=False
                 )
            ],
            label="Modified data",
            id={"tab3_figure_tabs":'t2'}),
    ],
    id="tab3_plot_tabs",
    active_tab="tab-0",
    style = {'font-size':'0.7em'},
    class_name='mb-1'),
    html.Div(id='tab3_figure_add_dropdown'),
    html.Div(id='tab3_figure_output'),

])

#Tab3 Modals
tab3_modal = html.Div([
    dbc.Modal(
        [dbc.ModalHeader('Choose data sections',close_button=False),
          dbc.ModalBody([dbc.Switch(id="tab3_model_database_all_division",
                                    label="Select all",
                                    value=True,
                                    style = {'font-size':'0.8em'},
                                    className="mb-0 mt-2"),
                         dbc.Alert("At least 1 option must be chosen.",
                                   id='tab3_model_database_division_checklist_alert',
                                   is_open=False,color="danger",
                                   class_name="c_alert"),
                        dbc.Checklist(id="tab3_model_database_division_checklist",
                                      label_style = {'font-size':'0.8em'})
                        ]),
          dbc.ModalFooter(dbc.Button("Ok", id="tab3_division_option_ok",
                          className="ms-auto",n_clicks=0))
        ],
        id="tab3_division_option",
        is_open=False,
        backdrop='static',
        scrollable=True),
    
    dbc.Modal(
        [dbc.ModalHeader('Exra forecast argument',id='tab3_forecast_ex_header',
                         close_button=False),
         dbc.ModalBody([
             dbc.Alert("Please fill in all required parameters.",
                       id='tab3_forecast_ex_alert',
                       is_open=False,color="danger",
                       class_name="c_alert"),
             dbc.Label("Model parameters predict method:",
                       id='tab3_params_model_label',
                       class_name="mb-0 mt-2",
                       style={'font-size':'0.8em'}),
             dbc.Row(dbc.RadioItems(
                 options=[{"label": 'Auto', "value": 'auto'},
                          {"label":'ARIMA',"value":'arima'},
                          {"label":'Curve fit',"value":'curve_fit'}],
                 id='tab3_params_model',
                 value='auto',
                 label_style = {'font-size':'0.8em'},
                 inline=True)),
             dbc.Tooltip(html.Div(
                 '''Choose the method for calculating SVID 
                 model's parameters.
            
                -Auto: Auto choosing the best method
                -ARIMA: Using ARIMA model 
                to calculate the parameters
                -Curve fit: Using regression 
                line to estimate the parameters''',
                style={'white-space':'pre-line',
                       'text-align':'left'}),
                target='tab3_params_model_label',
                delay={'show': 1000},placement='auto-end'),
                        
             dbc.Label('Model forecast scoring method:',
                       id='tab3_params_scoring_label',
                       style={'font-size':'0.8em'}),
             dbc.Col(dbc.Select(
                 id='tab3_params_scoring',
                 options=[{'label':'Mean Absolute Percentage Error',
                           'value':'neg_mean_absolute_percentage_error'
                           },
                          {'label':'Mean Absolute Error',
                           'value':'neg_mean_absolute_error'},
                          {'label':'Mean Squared Log Error',
                           'value':'neg_mean_squared_log_error'},
                          {'label':'Root Mean Squared Error',
                           'value':'neg_root_mean_squared_error'}],
                 value='neg_mean_absolute_percentage_error',
                 style = {'font-size':'0.8em'},size='sm'),width=9),
             dbc.Tooltip(html.Div(
                 '''The scoring method for both comparing 
                 predict methods and estimating accurary.''',
                 style={'white-space':'normal',
                        'text-align':'left'}),
                 target='tab3_params_scoring_label',
                 delay={'show': 1000},placement='auto-end'),
            
             dbc.Collapse([
                 dbc.Label("Parameter forecast scoring criteria:",
                           id='tab3_params_model_criteria_label',
                           class_name="mb-0 mt-2",
                           style={'font-size':'0.8em'}),
                 dbc.Row(dbc.RadioItems(
                     options=[{"label": 'Bigger', "value": True},
                              {"label":'Smaller',"value":False}],
                     id='tab3_params_model_criteria',
                     value=False,
                     label_style = {'font-size':'0.8em'},
                     inline=True)),
                 dbc.Tooltip(html.Div(
                     '''Choose whether predict method with bigger 
                     or smaller score value is better.''',
                     style={'white-space':'normal',
                            'text-align':'left'}),
                     target='tab3_params_model_criteria_label',
                     delay={'show': 1000},placement='auto-end'),
            
                dbc.Row([dbc.Label(
                    '''Model forecast score difference allowed:''',
                    id='tab3_allowed_dif_label',width=7,
                    style = {'font-size':'0.8em'}),
                dbc.Col(dbc.Input(
                    id={'code':'p1t3','name':'tab3_allowed_dif',
                        "data_type":"special_input"},
                    type='number',size='sm',required=True,
                    min=0,step=0.001,value=0.05,
                    className="mt-1 mb-1"),
                    width=5)]),
                dbc.Tooltip(dcc.Markdown(
                    '''Different scoring distance allowed between 
                    ARIMA and curve fit method, if threshold 
                    is passed, ARIMA will always be chosen.''',
                    style={'white-space':'normal',
                           'text-align':'left'}),
                    target='tab3_allowed_dif_label',
                    delay={'show': 1000},placement='auto-end')
                ],
                 id='tab3_params_model_collapse',
                 is_open=True),
             
             dbc.Row([dbc.Label(
                 "In-sample training points:",width=7,
                 id='tab3_train_points_label',
                 style = {'font-size':'0.8em'}),
                 dbc.Col(dbc.Input(
                     id={'code':'p1t3','name':'tab3_train_points',
                         "data_type":"modal_input_check"},
                     type='number',size='sm',required=True,
                     min=1,step=1,value=28,
                     className="mt-1 mb-1"),
                     width=5)]),
             dbc.Tooltip(dcc.Markdown(
                 '''The number of training points 
                 when estimate ARIMA orders|curve 
                 fit from database. 
                 Note: The total data points of 
                 both training and testing is 
                 the number of data points 
                 used to fit the output ARIMA 
                 & curve fit model.''',
                 style={'white-space':'normal',
                        'text-align':'left'}),
                 target='tab3_train_points_label',
                 delay={'show': 1000},placement='auto'),
       
             dbc.Row([dbc.Label(
                 "In-sample testing points:",
                 width=7,id='tab3_test_points_label',
                 style = {'font-size':'0.8em'}),
                 dbc.Col(dbc.Input(
                     id={'code':'p1t3','name':'tab3_test_points',
                         "data_type":"modal_input_check"},
                     type='number',size='sm',required=True,
                     min=1,step=1,value=7,
                     className="mt-1 mb-1"),
                     width=5)]),
             dbc.Tooltip(html.Div(
                 '''The number of testing points 
                 when estimate ARIMA orders|curve 
                 fit from database. 
                       
                 Note: The total data points of 
                 both training and testing is 
                 the number of data points 
                 used to fit the output ARIMA 
                 & curve fit model.''',
                 style={'white-space':'normal',
                        'text-align':'left'}),
                 target='tab3_test_points_label',
                 delay={'show': 1000},placement='auto'),
             
             dbc.Collapse([
                 html.Hr(),
                 dbc.Label(html.Li("ARIMA method:"),
                           class_name="ms-1",width='12'),
                 dbc.Label(
                     "ARIMA max order (p,d,q):",
                     width=7,id='tab3_arima_max_order_label',
                     style = {'font-size':'0.8em'}),
                 dbc.Tooltip(html.Div(
                     '''The maximum allowed values for grid 
                     searching the best ARIMA orders of all
                     SVID model parameters.''',
                     style={'white-space':'normal',
                            'text-align':'left'}),
                     target='tab3_arima_max_order_label',
                     delay={'show': 1000},placement='auto'),
                 dmc.Grid(
                     [dmc.Col(dbc.Label(
                         'Order of AR:',
                         width='auto',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(dbc.Label(
                         'Order of differencing:',
                         width='auto',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(dbc.Label(
                         'Order of MA:',
                         width='auto',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(span=3),
                     dmc.Col(dbc.Input(
                         id={"code":"p1t3","name":"tab3_arima_max_order",
                             "type":"special_input","index":0,"params":0},
                         step=1,min=1,value=5,required=True,
                         type='number',size='sm',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(dbc.Input(
                         id={"code":"p1t3","name":"tab3_arima_max_order",
                             "type":"special_input","index":0,"params":1},
                         step=1,min=1,value=2,required=True,
                         type='number',size='sm',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(dbc.Input(
                         id={"code":"p1t3","name":"tab3_arima_max_order",
                             "type":"special_input","index":0,"params":2},
                         step=1,min=1,value=5,required=True,
                         type='number',size='sm',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(span=3),
                     ],
                     gutter=3,
                     align='center',
                     id='tab3_arima_max_order'),
                 dbc.Label(
                     "ARIMA max seasonal order (P,D,Q):",
                     width=7,id='tab3_arima_max_s_order_label',
                     style = {'font-size':'0.8em'}),
                 dbc.Tooltip(html.Div(
                     '''The maximum allowed values for grid 
                     searching the best ARIMA seasonal orders of 
                     all SVID model parameters.''',
                     style={'white-space':'normal',
                            'text-align':'left'}),
                     target='tab3_arima_max_s_order_label',
                     delay={'show': 1000},placement='auto'),
                 dmc.Grid(
                     [dmc.Col(dbc.Label(
                         'Order of SAR:',
                         width='auto',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(dbc.Label(
                         'Order of seasonal differencing:',
                         width='auto',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(dbc.Label(
                         'Order of SMA:',
                         width='auto',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(span=3),
                     dmc.Col(dbc.Input(
                         id={"code":"p1t3","name":"tab3_arima_max_s_order",
                             "type":"special_input","index":0,"params":0},
                         step=1,min=1,value=2,required=True,
                         type='number',size='sm',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(dbc.Input(
                         id={"code":"p1t3","name":"tab3_arima_max_s_order",
                             "type":"special_input","index":0,"params":1},
                         step=1,min=1,value=1,required=True,
                         type='number',size='sm',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(dbc.Input(
                         id={"code":"p1t3","name":"tab3_arima_max_s_order",
                             "type":"special_input","index":0,"params":2},
                         step=1,min=1,value=2,required=True,
                         type='number',size='sm',style={'font-size':'0.7em'}),
                         span=3),
                     dmc.Col(span=3),
                     ],
                     gutter=3,
                     align='center',
                     id='tab3_arima_max_s_order'),
                 
                 dbc.Label("Preset ARIMA model orders:",
                      id='tab3_arima_option_label',
                      style = {'font-size':'0.8em'}),
                 dbc.Row(dbc.RadioItems(
                     options=[{"label": 'None', "value": None},
                              {"label":'Auto',"value":'auto'},
                              {"label":'Manual',"value":'manual'}],
                     id='tab3_arima_option',
                     value=None,
                     label_style = {'font-size':'0.8em'})),
                 dbc.Tooltip(html.Div(
                     '''Choose whether to preset the 
                     ARIMA orders for fast model 
                     calculation or not.
                           
                     -None: No presetting order
                     -Auto: Auto choosing the best 
                     orders base on predefined method.
                     -Manual: User choose their 
                     own ARIMA orders
                     
                     NOTE: This input is only applied for 
                     individual country and VietNam's province 
                     databases only.''',
                     style={'white-space':'pre-line',
                            'text-align':'left'}),
                     target='tab3_arima_option_label',
                     delay={'show': 1000},placement='auto-end'),
                 
                dbc.Collapse([
                    dbc.Label("Customize ARIMA order (p,d,q):",width=7,
                                       id='tab3_arima_order_label',
                                       style = {'font-size':'0.8em'}),
                    dbc.Tooltip(html.Div(
                        '''The customized values for ARIMA orders of 
                        all SVID model parameters.''',
                            style={'white-space':'normal',
                                   'text-align':'left'}),
                            target='tab3_arima_order_label',
                            delay={'show': 1000},placement='auto'),
                    dmc.Grid(
                        [dmc.Col(dbc.Label(
                            'Model parameter:',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Label(
                            'Order of AR:',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Label(
                            'Order of differencing:',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Label(
                            'Order of MA:',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=4),
                        
                        dmc.Col(dbc.Label(
                            'beta',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":0,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":0,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":0,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=4),
                        
                        dmc.Col(dbc.Label(
                            'beta_v',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":1,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":1,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":1,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=4),
                        
                        dmc.Col(dbc.Label(
                            'gamma',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":2,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":2,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":2,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=4),
                        
                        dmc.Col(dbc.Label(
                            'theta',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":3,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":3,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":3,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=4),
                        
                        dmc.Col(dbc.Label(
                            'alpha',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":4,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":4,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":4,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=4),
                        
                        dmc.Col(dbc.Label(
                            'alpha0',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":5,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":5,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":5,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=4),
                        
                        dmc.Col(dbc.Label(
                            'pi',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":6,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":6,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_order",
                                "type":"special_input","index":6,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=4),
                        ],
                        gutter=3,
                        align='center',
                        id='tab3_arima_order'),
                    
                    dbc.Label(
                        "Customize ARIMA seasonal order (P,D,Q,s):",
                        id='tab3_arima_s_order_label',
                        width=7,
                        style = {'font-size':'0.8em'}),
                    dbc.Tooltip(html.Div(
                        '''The customized values for ARIMA seasonal 
                        orders of all SVID model parameters.''',
                            style={'white-space':'normal',
                                   'text-align':'left'}),
                            target='tab3_arima_s_order_label',
                            delay={'show': 1000},placement='auto'),
                    dmc.Grid(
                        [dmc.Col(dbc.Label(
                            'Model parameter:',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Label(
                            'Order of AR:',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Label(
                            'Order of differencing:',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Label(
                            'Order of MA:',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Label(
                            'Frequency cycle:',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=2),
                        
                        dmc.Col(dbc.Label(
                            'beta',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":0,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":0,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":0,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":0,"params":3},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=2),
                        
                        dmc.Col(dbc.Label(
                            'beta_v',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":1,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":1,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":1,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":1,"params":3},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=2),
                        
                        dmc.Col(dbc.Label(
                            'gamma',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":2,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":2,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":2,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":2,"params":3},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=2),
                        
                        dmc.Col(dbc.Label(
                            'theta',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":3,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":3,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":3,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":3,"params":3},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=2),
                        
                        dmc.Col(dbc.Label(
                            'alpha',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":4,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":4,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":4,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":4,"params":3},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=2),
                        
                        dmc.Col(dbc.Label(
                            'alpha0',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":5,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":5,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":5,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":5,"params":3},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=2),
                        
                        dmc.Col(dbc.Label(
                            'pi',
                            width='auto',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":6,"params":0},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":6,"params":1},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":6,"params":2},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(dbc.Input(
                            id={"code":"p1t3","name":"tab3_arima_s_order",
                                "type":"special_input","index":6,"params":3},
                            step=1,min=0,value=0,required=True,
                            type='number',size='sm',style={'font-size':'0.7em'}),
                            span=2),
                        dmc.Col(span=2),
                        ],
                        gutter=3,
                        align='center',
                        id='tab3_arima_s_order'),
                    ],
                    id='tab3_arima_order_collapse',
                    is_open=False),
           
                dbc.Row([dbc.Label(
                    "Information Criterion:",width=7,
                    id='tab3_arima_criterion_label',
                    style = {'font-size':'0.8em'}),
                    dbc.Col(dbc.Select(options=
                        [{"label": "aic", "value": "aic"},
                         {"label": "bic", "value": "bic"},
                         {"label": "hqic", "value": "hqic"},
                         {"label": "oob", "value": "oob"}],
                        id="tab3_arima_criterion",
                        value="oob",
                        size='sm',
                        className="mt-1 mb-1"),
                        width=5)]),
                dbc.Tooltip(html.Div(
                    '''The information criterion used to select 
                    the best ARIMA model.''',
                    style={'white-space':'normal',
                           'text-align':'left'}),
                    target='tab3_arima_criterion_label',
                    delay={'show': 1000},placement='auto'),
           
                ],
                id='tab3_arima_collapse',
                is_open=True),
           
           dbc.Collapse([
               html.Hr(),
               dbc.Label(html.Li("Curve fit method:"),
                         class_name="ms-1",width='12'),
               dbc.Row([
                   dbc.Label(
                       "Curve fit max degree:",
                       id='tab3_curve_degree_label',
                       width=7,
                       style = {'font-size':'0.8em'}),
                   dbc.Col(dbc.Input(
                       id={'code':'p1t3','name':'tab3_curve_degree',
                           "data_type":"special_input"},
                       type='number',size='sm',required=True,
                       min=1,step=1,value=5,
                       className="mt-1 mb-1"),
                       width=5)]),
               dbc.Tooltip(html.Div(
                   '''The maximum degree of the fitting 
                   polynomial.''',
                   style={'white-space':'normal',
                          'text-align':'left'}),
                   target='tab3_curve_degree_label',
                   delay={'show': 1000},placement='auto'),
           
                dbc.Label("Curve fit method:",
                    id='tab3_curve_fit_method_label',
                    class_name="mb-0 mt-2",
                    style={'font-size':'0.8em'}),
                dbc.Row(dbc.RadioItems(
                    options=[{"label": 'Auto', "value": 'auto'},
                             {"label":'Polynomial',"value":'poly_fit'},
                             {"label":'Exponential function with base <1',
                              "value":'exponential_base[0,1]'},
                             {"label":'Exponential function with base >1',
                              "value":'exponential_base[1,inf]'},
                             {"label":'Sigmoid',"value":'sigmoid'}],
                    id='tab3_curve_fit_method',
                    value='auto',
                    label_style = {'font-size':'0.8em'})),
                dbc.Tooltip(html.Div(
                    '''Select a specific curve fit method or 
                    automatically choose the fittest regression 
                    line.''',
                    style={'white-space':'normal',
                           'text-align':'left'}),
                    target='tab3_curve_fit_method_label',
                    delay={'show': 1000},placement='auto'),
           
                dbc.Label('Curve fit scoring method:',
                    id='tab3_curve_fit_scoring_label',
                    style={'font-size':'0.8em'}),
                dbc.Col(dbc.Select(id='tab3_curve_fit_scoring',
                    options=[{'label':'Mean Absolute Percentage Error',
                              'value':'neg_mean_absolute_percentage_error'},
                             {'label':'Mean Absolute Error',
                              'value':'neg_mean_absolute_error'},
                             {'label':'Mean Squared Log Error',
                              'value':'neg_mean_squared_log_error'},
                             {'label':'Root Mean Squared Error',
                              'value':'neg_root_mean_squared_error'}],
                    value='neg_mean_absolute_percentage_error',
                    style = {'font-size':'0.8em'},size='sm'),width=9),
                dbc.Tooltip(html.Div(
                    '''The scoring method for both comparing 
                    the fittest regression line and estimating 
                    accurary.''',
                    style={'white-space':'normal',
                           'text-align':'left'}),
                    target='tab3_curve_fit_scoring_label',
                    delay={'show': 1000},placement='auto-end'),
           
                dbc.Collapse([
                    dbc.Label("Curve fit scoring criteria:",
                        id='tab3_curve_fit_criteria_label',
                        class_name="mb-0 mt-2",
                        style={'font-size':'0.8em'}),
                    dbc.Row(dbc.RadioItems(
                        options=[{"label": 'Bigger', "value": True},
                        {"label":'Smaller',"value":False}],
                        id='tab3_curve_fit_criteria',
                        value=False,
                        label_style = {'font-size':'0.8em'},
                        inline=True)),
                dbc.Tooltip(html.Div(
                    '''Choose whether bigger or smaller score value
                    is better.''',
                    style={'white-space':'normal',
                           'text-align':'left'}),
                    target='tab3_curve_fit_criteria_label',
                    delay={'show': 1000},placement='auto-end')
                    ],
                    id='tab3_curve_fit_collapse2',
                    is_open=True),
                ],
               id='tab3_curve_fit_collapse1',
               is_open=True)
           ]),
         dbc.ModalFooter(dbc.Row(dbc.Col([
             dbc.Button("Apply",id="tab3_forecast_ex_apply",
                        size='sm',n_clicks=0,
                        className="ms-auto"),
             dbc.Button("Cancel",id="tab3_forecast_ex_cancel",
                        size='sm',n_clicks=0,
                        className="ms-auto")]),
             justify='end')
             )
        ],
        id="tab3_forecast_ex",
        is_open=False,
        backdrop='static'),
    
    dbc.Modal(
        [dbc.ModalHeader('Processing status',close_button=False),
          dbc.ModalBody([
                         dbc.Row([dbc.Col(dbc.Spinner(color="primary",size='sm'),
                                           width='auto'),
                                   dbc.Col(html.P(children="Processing...",
                                                  id='tab3_update_process_info',
                                                  style = {'font-size':'0.8em'}),
                                           width='auto')
                                   ]),
                         dbc.Collapse([dbc.Progress(
                                 id='tab3_update_process',
                                 striped=True)],
                             id='tab3_update_process_collapse',
                             is_open=False),
                        ]),
          dbc.ModalFooter(dbc.Button("Cancel", id="tab3_model_Cancel_Button1",
                          className="ms-auto",size='sm',n_clicks=0))
        ],
        id="tab3_processing_status",
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
          dbc.ModalFooter(dbc.Button("Cancel", id="tab3_model_Cancel_Button2",
                          className="ms-auto",size='sm',n_clicks=0))
        ],
        id="tab3_downloading_status",
        is_open=False,
        backdrop='static',
        centered=True),

])

#Tab3 Tooltips

tab3_tooltip = html.Div([
    dbc.Tooltip(html.Div('''
        Color code to show the status of the relevant databases.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab3_color_code_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Show the modeling process status of relevant databases.
        If a database fails to process, it won't override the old modified
        database and change the color code.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab3_model_select_label',
        delay={'show': 1000}),
    dbc.Tooltip(html.Div('''
                Choose between Vietnam database and world database.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_model_database_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''
                Switch on to choose which database sections to process.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_model_database_division_switch',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''
                By default, 'birth rate' and 'death rate' of main database will
                be used (Vietnam data and world data).
                For each Vietnam province, average birth rate and death rate 
                will be applied by default, but can take customized values 
                applying to all provinces.
                For each country of world data, their own birth rate and death 
                rate will be applied by default, but can take customized values
                applying to all countries.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_census_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''p: The fraction of birth population that is is not 
                         vaccinated.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_p_label',
                delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Apply the manually input birth rate per day for processing all selected 
        databases.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab3_manual_b_rate_label',
        delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''
        Apply the manually input death rate per day for processing all selected 
        databases.
        ''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab3_manual_d_rate_label',
        delay={'show': 1000}),
    dbc.Tooltip(html.Div('''Setting inputs for forecast models.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_forecast_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''Select a starting date for model forecasting:
                         -auto: select the lastest date in each database division
                         -specific date: choose a date from the calendar, any
                         database division that is out of range will not be 
                         included in the forecast result.
                         Input format is: YYYY-MM-DD''',
                style={'white-space':'pre-line',
                       'text-align':'left'}),
                target='tab3_start_date_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''Choose the number of days for forecasting. The more 
                         days chosen, the less accurate the furthest day will 
                         be. Recommended range is 7-14 days.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_days_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''Choose the forecasting model:
                         -Model 1: delta method, in which X_next_day = 
                         X_previous_day + f()
                         -Model 2: ODEs which use all forecasted model 
                         parameters to forecast SVID
                         -Model 3: ODEs which use  model parameters of 
                         'start_date' to forecast SVID
                         Note: Model 3 is the fastest of the 3 models, since
                         no ARIMA or curve fit method is used for estimating
                         the model parameters.''',
                style={'white-space':'pre-line',
                       'text-align':'left'}),
                target='tab3_model_number_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''Return only the best|all models.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_best_model_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''The scoring method for both comparing 
                         the fittest model and estimating accurary.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_scoring_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''Choose whether predict method with bigger 
                         or smaller score value is better.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_model_criteria_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''Choose how the output scores will be displayed:
                         -Raw values: Raw scores of each S,V,I,D will be 
                         displayed
                         -Variance weight: Raw scores will be averaged based
                         on the weights of each S,V,I,D variance
                         -Uniform average: Return the mean of all raw scores
                         ''',
                style={'white-space':'pre-line',
                       'text-align':'left'}),
                target='tab3_multioutput_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''Average protected (immunity time) of post-covid 
                         patient (in days).''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_r_protect_time_label',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''The number of data points prior to 'start date'
                         that will be plotted.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_plot_point_label1',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''The number of data points prior to 'start date'
                         that will be plotted.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_plot_point_label2',
                delay={'show': 1000}),
    dbc.Tooltip(html.Div('''The number of days to use in equilibrium plot. 
                         Remember, the equilibrium plot use the SVID model
                         initial state at 'start date'.''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_n_days_label',
                delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown('''Extra arguments for forecasting, if 'Cancel' is
                             selected then no extra arguments will be applied''',
                style={'white-space':'normal',
                       'text-align':'left'}),
                target='tab3_forecast_ex_button',
                delay={'show': 1000}),
    dbc.Tooltip(dcc.Markdown(
        '''The maximum number of process which will be used for multiprocessing.
        It is recommended not to set this number too high (normally use around 
        half of CPU processors), otherwise it might slow down the modifying 
        process.''',
        style={'white-space':'normal',
               'text-align':'left'}),
        target='tab3_max_worker_label',
        delay={'show': 1000}),
    ])        
                        
#Tab3 callbacks

##initial data loading--------------------------------------------------------

@callback(
    Output('tab3_data_initiator','disabled'),
    Input('tab1_initial_status','data'),
    Input('tab2_initial_status','data'),
    State('tab3_data_initiator','n_intervals'),
    prevent_initial_call=True
)
def tab3_trigger_data_process(trigger1,trigger2,count):
    """
    Triggered only the first time.
    """
    if count>=1:
        return dash.no_update
    else:
        if sum([trigger1,trigger2])!=2:
            return dash.no_update
        else:
            return False
    
@callback_extension(
    ServersideOutput({'code':'p1t3','name':'birth_data',"data_type":"census"},'data'),
    ServersideOutput({'code':'p1t3','name':'death_data',"data_type":"census"},'data'),
    Input('tab3_data_initiator','n_intervals'),
    Input('tab3_range_b_d_rate','value'),
    State({'code':'p1t1','name':'birth_data',"data_type":"census"},'data'),
    State({'code':'p1t1','name':'death_data',"data_type":"census"},'data'),
    prevent_initial_call=True
)
def tab3_process_b_d_rate(trigger,year_range,birth_df,death_df):
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

server_backend = FileSystemStore(cache_dir="./cache/server_cache",threshold=50,
                             default_timeout=0
                             )
@callback_extension(
    [
     ServersideOutput({'code':'p1t3','name':'vn_prediction',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t3','name':'vn_prediction_by_province',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t3','name':'vn_equilibrium',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t3','name':'vn_equilibrium_by_province',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t3','name':'w_prediction',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t3','name':'w_prediction_by_country',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t3','name':'w_equilibrium',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
     ServersideOutput({'code':'p1t3','name':'w_equilibrium_by_country',"data_type":"server"},
                       'data',arg_check=False,session_check=False,
                       backend=server_backend),
    ],
    Input('tab3_data_initiator','n_intervals'),
    memoize=True,
    prevent_initial_call=True
    )
def tab3_load_default_cached_database(click):
    """
    Save databases cache for faster data loading.
    """
    vn_prediction=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_output/VietNamData/vn_prediction.csv",
        parse_dates=['date'])
    vn_prediction_by_province=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_output/VietNamData/vn_prediction_by_province.csv",
        parse_dates=['date'])
    vn_equilibrium = pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_output/VietNamData/vn_equilibrium.csv",
        parse_dates=['date'])
    vn_equilibrium_by_province = pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_output/VietNamData/vn_equilibrium_by_province.csv",
        parse_dates=['date'])
    w_prediction=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_output/WorldData/w_prediction.csv",
        parse_dates=['date'])
    w_prediction_by_country=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_output/WorldData/w_prediction_by_country.csv",
        parse_dates=['date'])
    w_equilibrium=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_output/WorldData/w_equilibrium.csv",
        parse_dates=['date'])
    w_equilibrium_by_country=pd.read_csv(
        "https://raw.githubusercontent.com/DeKhaos/covid-modeling/master/"+
        "database/Covid-19_modeling_output/WorldData/w_equilibrium_by_country.csv",
        parse_dates=['date'])
    
    return [
        vn_prediction,
        vn_prediction_by_province,
        vn_equilibrium,
        vn_equilibrium_by_province,
        w_prediction,
        w_prediction_by_country,
        w_equilibrium,
        w_equilibrium_by_country
        ]

@callback(
    Output('tab3_figure_dropdown_trigger','data'),
    Input('tab3_data_initiator','n_intervals'),
    Input({'code':'p1t3','name':'vn_prediction',"data_type":"server"},'data'),
    State('tab3_figure_dropdown_trigger','data'),
    prevent_initial_call=True
    )
def tab3_trigger_plot_process(initial_trigger1,initial_trigger2,
                              count):
    """
    Synchronize initial trigger condition of plotting.
    """
    return count+1

@callback_extension(
    [
     ServersideOutput({'code':'p1t3','name':'vn_prediction_mod',
                       "data_type":"update"},'data'),
     ServersideOutput({'code':'p1t3','name':'vn_prediction_by_province_mod',
                       "data_type":"update"},'data'),
     ServersideOutput({'code':'p1t3','name':'vn_equilibrium_mod',
                       "data_type":"update"},'data'),
     ServersideOutput({'code':'p1t3','name':'vn_equilibrium_by_province_mod',
                       "data_type":"update"},'data'),
     ServersideOutput({'code':'p1t3','name':'w_prediction_mod',
                       "data_type":"update"},'data'),
     ServersideOutput({'code':'p1t3','name':'w_prediction_by_country_mod',
                       "data_type":"update"},'data'),
     ServersideOutput({'code':'p1t3','name':'w_equilibrium_mod',
                       "data_type":"update"},'data'),
     ServersideOutput({'code':'p1t3','name':'w_equilibrium_by_country_mod',
                       "data_type":"update"},'data'),
     
     #buffer dataframe
     ServersideOutput({'code':'p1t3','name':'vn_modeling_data_mod',
                       "data_type":"buffer"},'data'),
     ServersideOutput({'code':'p1t3','name':'vn_modeling_data_by_province_mod',
                       "data_type":"buffer"},'data'),
     ServersideOutput({'code':'p1t3','name':'w_modeling_data_mod',
                       "data_type":"buffer"},'data'),
     ServersideOutput({'code':'p1t3','name':'w_modeling_data_by_country_mod',
                       "data_type":"buffer"},'data'),
    ],
    Input('tab3_data_initiator','n_intervals'),
    prevent_initial_call=True
    )
def tab3_generate_uid(update_state):
    """
    Generate uid for cache access of modified databases.
    """
    outputs = [None for _ in range(len(ctx.outputs_list))]
    return outputs

#handle checking of preloaded data
@callback(
    output = Output('tab3_initial_status','data'),
    inputs = [
        Input('tab3_data_initiator','n_intervals'),
        Input({'code':'p1t3','name':'birth_data',"data_type":"census"},'data'),
        Input({'code':'p1t3','name':'vn_prediction',"data_type":"server"},'data')
        ],
    prevent_initial_call=True
    )
def tab3_initial_data_loaded(trigger1,trigger2,trigger3):
    """Use to check if all tab preloaded data is ready"""
    return 1

##buttons interactions--------------------------------------------------------

@callback(
    Output('tab3_model_database_all_switch_collapse','is_open'),
    Input('tab3_model_database_all_switch','value'),
    prevent_initial_call=True
    )
def tab3_choose_all_database(switch):
    """
    Switch between choosing all database or Vietnam|World.
    """
    if switch:
        return False
    else:
        return True

@callback_extension(
    Output('tab3_model_database_division_switch','label'),
    Output('tab3_model_database_division_switch','value'),
    Output('tab3_model_database_division_checklist','options'),
    Output('tab3_model_database_division_checklist','value'),
    Output('tab3_model_database_all_division','value'),
    Input('model_param_tabs','active_tab'),
    Input('tab3_model_database','value'),
    Input('tab3_model_database_all_division','value'),
    Input('tab3_model_database_division_checklist','value'),
    State('tab3_model_database','value'),
    State({'code':'p1t2','name':'select_wbc_database',"purpose":"database_check"},'value'),
    State({'code':'p1t2','name':'w_modeling_data_by_country',"data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_modeling_data_by_country_modified',"data_type":"update"},'data'),
    State({'code':'p1t2','name':'select_vnbp_database',"purpose":"database_check"},'value'),
    State({'code':'p1t1','name':'vn_province_code',"data_type":"internal"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data_by_province',"data_type":"server"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data_by_province_modified',"data_type":"update"},'data'),
    prevent_initial_call=True
    )
def tab3_change_database(
        tab3_trigger,
        database_switch,
        all_switch,
        checklist,
        current_database,
        w_switch,w_df,w_modified_df,
        vn_switch,province_code,vn_df,vn_modified_df):
    """
    Change between Vietnam and world modeling database and synchronize all 
    relevant buttons and modal.
    """
    province_code = dict(province_code)
    if current_database=="vietnam":
        if vn_switch==1:
            all_options = vn_df['id'].unique().tolist()
        else:
            all_options = vn_modified_df['id'].unique().tolist()
    else:
        if w_switch==1:
            all_options = w_df['iso_code'].unique().tolist()
        else:
            all_options = w_modified_df['iso_code'].unique().tolist()
    if ctx.triggered_id in ["tab3_model_database",
                            "model_param_tabs"]:
        if database_switch=="vietnam":
            if vn_switch==1:
                options = [{'label':item,'value':province_code[item]} for 
                           item in vn_df['provinceName'].unique().tolist()]
                values = all_options
            else:
                options = [{'label':item,'value':province_code[item]} for 
                           item in vn_modified_df['provinceName'].unique().tolist()]
                values = all_options
            return "Use database by each province",False,options,values,True
        else:
            if w_switch==1:
                options = [{'label':item[0],'value':item[1]} for item in 
                           w_df[['location','iso_code']].drop_duplicates().values]
                values = all_options
            else:
                options = [{'label':item[0],'value':item[1]} for item in 
                           w_modified_df[['location','iso_code']].drop_duplicates().values]
                values = all_options
            return "Use database by each country",False,options,values,True
    elif ctx.triggered_id =="tab3_model_database_all_division":
        if all_switch:
            return dash.no_update,dash.no_update,dash.no_update,all_options,dash.no_update
        else:
            return dash.no_update,dash.no_update,dash.no_update,[],dash.no_update
    elif ctx.triggered_id=="tab3_model_database_division_checklist":
        if np.all(np.isin(all_options,checklist)):
            return dash.no_update,dash.no_update,dash.no_update,dash.no_update,True
        else:
            return dash.no_update,dash.no_update,dash.no_update,dash.no_update,False

@callback_extension(
    Output({"code":"p1t3","name":"tab3_manual_b_d_rate",
        "type":"special_input","index":0},'value'),
    Output({"code":"p1t3","name":"tab3_manual_b_d_rate",
        "type":"special_input","index":1},'value'),
    Input('tab3_model_database_all_switch','value'),
    Input('tab3_model_database','value'),
    Input('tab3_auto_b_d_rate','value'),
    State({'code':'p1t3','name':'birth_data',"data_type":"census"},'data'),
    State({'code':'p1t3','name':'death_data',"data_type":"census"},'data'),
    prevent_initial_call=True
    )
def tab3_change_initial_b_d_rate_manual(trigger1,trigger2,trigger3,birth_df,death_df):
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

@callback(
    Output({"code":"p1t3","name":"tab3_arima_order",
        "type":"special_input","index":ALL,"params":ALL},'value'),
    Input('tab3_model_database_all_switch','value'),
    Input('tab3_model_database','value'),
    )
def tab3_change_initial_customized_ARIMA_order(trigger1,trigger2):
    """
    Update initial ARIMA orders in extra argument modal
    """
    #TODO: Update this function everytime server data is updated
    w_SARIMA = [((2, 1, 1), (0, 0, 0, 0)),
                 ((0, 1, 4), (0, 0, 0, 0)),
                 ((0, 1, 0), (0, 0, 0, 0)),
                 ((0, 1, 0), (0, 0, 0, 0)),
                 ((0, 1, 0), (0, 0, 0, 0)),
                 ((0, 1, 0), (0, 0, 0, 0)),
                 ((4, 1, 1), (0, 0, 0, 0))]
    
    vn_SARIMA = [((1, 1, 0), (0, 0, 0, 0)),
                 ((1, 1, 0), (0, 0, 0, 0)),
                 ((0, 0, 0), (0, 0, 0, 0)),
                 ((0, 2, 1), (0, 0, 0, 0)),
                 ((0, 1, 0), (0, 0, 0, 0)),
                 ((0, 1, 3), (0, 0, 0, 0)),
                 ((0, 1, 0), (0, 0, 0, 0))]
    
    if trigger1:
        SARIMA_list = w_SARIMA
    else:
        if trigger2=="vietnam":
            SARIMA_list = vn_SARIMA
        else: 
            SARIMA_list = w_SARIMA
    dummy_list = []
    for item in SARIMA_list:
        dummy_list.extend(list(item[0]))
    
    return dummy_list

@callback(Output('tab3_model_database_division_button_collapse','is_open'),
          Input('tab3_model_database_division_switch','value'),
          prevent_initial_call=True)
def tab3_open_section_button(switch):
    '''
    Hide/unhide the sections button.
    '''
    if switch:
        return True
    else:
        return False
    
@callback([Output('tab3_division_option','is_open'),
           Output('tab3_model_database_division_checklist_alert','is_open')],
          [Input('tab3_model_database_division_button','n_clicks'),
           Input('tab3_division_option_ok','n_clicks')],
          State('tab3_model_database_division_checklist','value'),
          prevent_initial_call=True)
def tab3_open_section(button1,button2,option_state):
    '''
    Open and close the data section modal, can't close if no data section was 
    chosen
    '''
    if dash.callback_context.triggered_id =="tab3_model_database_division_button":
        return [True,dash.no_update]
    else:
        if option_state ==[]:
            return [True,True]
        else:
            return [False,False]
        
@callback(
    Output('tab3_range_b_d_rate_collapse','is_open'),
    Output('tab3_manual_b_d_rate_collapse','is_open'),
    Input('tab3_auto_b_d_rate','value'),
    prevent_initial_call=True
    )
def tab3_switch_auto_OR_manual_b_and_d_rate(trigger):
    """
    Switch between auto generate birth & death rate or not.
    """
    if trigger:
        return True,False
    else:
        return False,True
    
@callback(
    Output({'code':'p1t3','name':'b_d_rate',"data_type":"input_check"},'data'),
    Input('tab3_auto_b_d_rate','value'),
    Input({"code":"p1t3","name":"tab3_manual_b_d_rate","type":"special_input",
           "index":ALL},'value'),
    prevent_initial_call=True
    )
def tab3_check_b_d_rate(switch,values):
    """
    Check input status of death & birth rate manual input.
    """
    if switch:
        return True
    else:
        check_array = np.array(values,dtype='float')
        check_value = np.any(np.isnan(check_array))
        return not check_value

@callback([Output('tab3_start_date','options'),
           Output('tab3_start_date','value')],
          Input('tab3_start_date_select','value'),
          State('tab3_start_date','options'),
          prevent_initial_call=True)
def tab3_optional_forecast_start_date(input_date,options):
    """
    Connect DatePickerSingle (id=tab3_start_date_select) value with custom option in 
    RadioItem (id=tab3_start_date)
    """
    options[-1]['value']=input_date
    return [options,input_date]
    
@callback([Output('tab3_model_number_alert','is_open'),
           Output('tab3_model_number','class_name'),
           Output({'code':'p1t3','name':'tab3_model_number',
                   "data_type":"input_check"},'data')],
          Input('tab3_model_number','value'))
def tab3_alert_no_model(value):
    if value==[]:
        return [True,'mx-1 border border-1 border-danger',False]
    else:
        return [False,'mx-1 border border-1 border-secondary',True]
  
@callback([Output('tab3_params_model_collapse','is_open'),
           Output('tab3_arima_collapse','is_open'),
           Output('tab3_curve_fit_collapse1','is_open')],
          [Input('tab3_params_model','value')])
def tab3_extra_forecast_argument_predict_method(model_option):
    '''
    Choose the SVID model parameters predicting method
    '''
    if model_option == 'auto':
        return [True,True,True]
    elif model_option == 'arima':
        return [False,True,False]
    else:
        return [False,False,True]
    
@callback([Output('tab3_arima_order_collapse','is_open')],
          [Input('tab3_arima_option','value')])
def tab3_extra_forecast_argument_customize_arima_order(order_option):
    '''
    Choose the method to find the ARIMA orders.
    '''
    if order_option == 'manual':
        return [True]
    else:
        return [False]

@callback(Output('tab3_curve_fit_collapse2','is_open'),
          Input('tab3_curve_fit_method','value'))
def tab3_extra_forecast_argument_curve_fit(option):
    '''
    Open criteria for selecting if auto choosing the best curve_fit method.
    '''
    if option == 'auto':
        return True
    else:
        return False

@callback(
    output = Output('tab3_forecast_ex_alert','is_open'),
    inputs = dict(
        normal_inputs = Input({'code':'p1t3','name':ALL,
                               "data_type":"modal_input_check"},'value'),
        model_condition = Input('tab3_params_model','value'),
        dif_score = Input({'code':'p1t3','name':'tab3_allowed_dif',
            "data_type":"special_input"},'value'),
        curve_degree = Input({'code':'p1t3','name':'tab3_curve_degree',
            "data_type":"special_input"},'value'),
        max_order = Input({"code":"p1t3","name":"tab3_arima_max_order",
            "type":"special_input","index":ALL,"params":ALL},'value'),
        max_s_order=Input({"code":"p1t3","name":"tab3_arima_max_s_order",
            "type":"special_input","index":ALL,"params":ALL},'value'),
        preset=Input('tab3_arima_option','value'),
        custom_order = Input({"code":"p1t3","name":"tab3_arima_order",
            "type":"special_input","index":ALL,"params":ALL},'value'),
        custom_s_order = Input({"code":"p1t3","name":"tab3_arima_s_order",
            "type":"special_input","index":ALL,"params":ALL},'value'),
        ),
    prevent_initial_call=True)
def tab3_alert_extra_forecast_arguments(normal_inputs,model_condition,dif_score,
                                        curve_degree,max_order,max_s_order,preset,
                                        custom_order,custom_s_order):
    """
    Check for invalid inputs in extra forecast argument modal.
    """
    check_list = []
    check_list.extend(normal_inputs)
    if model_condition=='auto':
        check_list.extend([dif_score,curve_degree])
        check_list.extend(max_order)
        check_list.extend(max_s_order)
    elif model_condition=="arima":
        check_list.extend(max_order)
        check_list.extend(max_s_order)
    elif model_condition=='curve_fit':
        check_list.append(curve_degree)
    if (model_condition in ['auto','arima']) and preset=='manual':
        check_list.extend(custom_order)
        check_list.extend(custom_s_order)
    check_array = np.array(check_list,dtype='float')
    check_value = np.any(np.isnan(check_array))
    return check_value

@callback(
    output=[Output('tab3_forecast_ex','is_open'),
            Output('tab3_forecast_extra_args','data'),
            Output('tab3_preset_orders','data')],
    inputs = dict(
        buttons=(Input('tab3_forecast_ex_button','n_clicks'),
                 Input('tab3_forecast_ex_apply','n_clicks'),
                 Input('tab3_forecast_ex_cancel','n_clicks')),
        warning=State('tab3_forecast_ex_alert','is_open'),
        normal_inputs = State({'code':'p1t3','name':ALL,
                               "data_type":"modal_input_check"},'value'),
        model_condition = State('tab3_params_model','value'),
        score_method = State('tab3_params_scoring','value'),
        score_criteria = State('tab3_params_model_criteria','value'),
        dif_score = State({'code':'p1t3','name':'tab3_allowed_dif',
            "data_type":"special_input"},'value'),
        curve_degree = State({'code':'p1t3','name':'tab3_curve_degree',
            "data_type":"special_input"},'value'),
        max_order = State({"code":"p1t3","name":"tab3_arima_max_order",
            "type":"special_input","index":ALL,"params":ALL},'value'),
        max_s_order=State({"code":"p1t3","name":"tab3_arima_max_s_order",
            "type":"special_input","index":ALL,"params":ALL},'value'),
        preset=State('tab3_arima_option','value'),
        custom_order = State({"code":"p1t3","name":"tab3_arima_order",
            "type":"special_input","index":ALL,"params":ALL},'value'),
        custom_s_order = State({"code":"p1t3","name":"tab3_arima_s_order",
            "type":"special_input","index":ALL,"params":ALL},'value'),
        arima_criteria = State('tab3_arima_criterion','value'),
        curve_method = State('tab3_curve_fit_method','value'),
        curve_score = State('tab3_curve_fit_scoring','value'),
        curve_criteria = State('tab3_curve_fit_criteria','value')
                  ),
    prevent_initial_call=True)
def tab3_process_extra_forecast_arguments(buttons,warning,
                                          normal_inputs,model_condition,score_method,
                                          score_criteria,dif_score,curve_degree,
                                          max_order,max_s_order,preset,
                                          custom_order,custom_s_order,arima_criteria,
                                          curve_method,curve_score,curve_criteria):
    """
    Handling extra forecast arguments.
    """
    
    if ctx.triggered_id=="tab3_forecast_ex_button":
        return [True,dash.no_update,dash.no_update]
    elif ctx.triggered_id=="tab3_forecast_ex_apply":
        output_dict= {}
        preset_order=None
        if warning:
            return [dash.no_update,dash.no_update,dash.no_update]
        output_dict['train_p']=normal_inputs[0]
        output_dict['test_p']=normal_inputs[1]
        output_dict['method']=model_condition
        output_dict['scoring']=score_method
        if model_condition=='auto':
            output_dict['greater_better']=score_criteria
            output_dict['allowed_dif']=dif_score
        if model_condition in ['auto','arima']:
            max_order.extend(max_s_order)
            buffer_dict = dict(
                zip(['max_p','max_d','max_q','max_P','max_D','max_Q'],max_order)
                )
            buffer_dict['information_criterion']=arima_criteria
            output_dict['auto_arima_kwargs']=buffer_dict
            
            if preset=='auto':
                preset_order='auto'
            elif preset=='manual':
                buffer_order = []
                for i in range(len(custom_order)//3):
                    tup1 = tuple(custom_order[i*3:(i+1)*3])
                    tup2 = tuple(custom_s_order[i*4:(i+1)*4])
                    buffer_order.append((tup1,tup2))
                preset_order=buffer_order
        if model_condition in ['auto','curve_fit']:
            buffer_dict = dict(degree=curve_degree,method=curve_method,
                                scoring=curve_score)
            if curve_method=='auto':
                buffer_dict['greater_better']=curve_criteria
            output_dict['curve_kwargs']=buffer_dict
        return [False,output_dict,preset_order]
    else:
        return [False,None,'auto']
    
##input parameters checking----------------------------------------------------

@callback(
    Output({'code':'p1t3','name':'normal_input',"data_type":"input_check"},'data'),
    Input({"code":"p1t3","name":ALL,"type":"input"},'value'),
    prevent_initial_call=True
    )
def tab3_check_normal_input(values):
    """
    Check input status normal inputs.
    """
    check_array = np.array(values,dtype='float')
    check_value = np.any(np.isnan(check_array))
    return not check_value

@callback(
    Output('tab3_inputs_alert','is_open'),
    Input({'code':'p1t3','name':ALL,"data_type":"input_check"},'data'),
    prevent_initial_call=True
    )
def tab3_check_input_warning(values):
    """
    Check all dcc.Store of input status to print fill in requirement.
    """
    check_value = np.all(values)
    return not check_value

##plot interaction-------------------------------------------------------------

@callback_extension(
    [Output('tab3_figure_output','children'),
    Output('tab3_figure_add_dropdown','children'),
    Output('tab3_figure_plot_trigger','data')],
    Input('tab3_figure_dropdown_trigger','data'), #trigger
    Input('tab3_plot_tabs','active_tab'),
    Input({'tab3_figure_tabs':ALL,'type':'select_data',"data_type":ALL},'value'),
    
    #server_files
    State({'code':'p1t3','name':'vn_prediction',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'vn_prediction_by_province',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'w_prediction',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'w_prediction_by_country',"data_type":"server"},'data'),
    #modified_files
    
    State({'code':'p1t3','name':'vn_prediction_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'vn_prediction_by_province_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'w_prediction_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'w_prediction_by_country_mod',"data_type":"update"},'data'),
    
    State({'code':'p1t1','name':'iso_code',"data_type":"internal"},'data'),
    State({'code':'p1t1','name':'vn_province_code',"data_type":"internal"},'data'),
    State('tab3_figure_plot_trigger','data'),
    prevent_initial_call=True
    )
def tab3_figure_option(initial_trigger,tab_click,data_click,
                       s1,s2,s3,s4,
                       u1,u2,u3,u4,
                       iso_code,vn_province_code,
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
                     "plot","layout"}
                with preset_layout: user can use keyword 'special_layout'
                to apply an evaluation to a layout parameter.
                E.g: {"preset":json.dumps([
                    {"layout":{"special_layout":[('layout_title_x','a+0.5')]}}
                                           ])
                      }
                    mean figure layout title will be string of (variable a +5)
                    where a is local variable in covid_science.utility_func.condition_plot
            'grid':default setting for grid usage {"grid_arg","col_arg"}
            }
    
    """
    idx = int(re.search(r"(?<=-).*",tab_click)[0])
    iso_code_dict = dict(iso_code)
    province_array = np.array(vn_province_code)
    province_array = province_array[:,[1,0]]
    province_code = dict(province_array.tolist())
    database_name = None
    for item in ctx.inputs_list[2]:
        if item['id']['tab3_figure_tabs']==f't{idx+1}':
            database_name = item['value']
            break
    wrapper = html.Div(id={'type':'figure_wrapper',
                           'tab3_figure_tabs':f't{idx+1}'})
    
    if database_name in ["vn_prediction",
                         "vn_prediction_mod"]:
        if database_name=="vn_prediction":
            df = s1
            tab = 't1'
        else:
            df = u1
            tab = 't2'
        
        titles = ['Model:','Best model:']
        no_best  = np.all(np.isnan(df['best_method'].unique()))
        option_groups = []
        select_options = []
        group_ids = []
        extra_args1 = {
            'target':['dataframe'],
            'plot_arg':None,'layout_arg':None,
            'code_obj':'{}.loc[[{}["method"].isin([np.NaN,{}]) if "method" in {}.columns else {}.index][0]]',
            'obj_type':'expression',
            'format':['df','df','variable','df','df']}
        extra_args2 = {
            'target':['dataframe'],
            'plot_arg':None,'layout_arg':None,
            'code_obj':'{}.loc[[{}["best_method"].isin([np.NaN,{}]) if "best_method" in {}.columns else {}.index][0]]',
            'obj_type':'expression',
            'format':['df','df','variable','df','df']}
        if no_best:
            option_groups.append([
                {"label":item,"value":item} 
                for item in df['method'].unique()
                ])
            select_options.append(df['method'].unique()[0])
            arg1 = extra_args1.copy()
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":0,
                 'extra_arg':json.dumps(arg1)}
                )
            
            option_groups.append([{"label":'None',"value":'None'}])
            select_options.append('None')
            arg2 = extra_args2.copy()
            arg2['target']=[None]
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":1,
                 'extra_arg':json.dumps(arg2)}
                )
            
        else:
            option_groups.append([{"label":'None',"value":'None'}])
            select_options.append('None')
            arg1 = extra_args1.copy()
            arg1['target']=[None]
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":0,
                 'extra_arg':json.dumps(arg1)}
                )
            
            option_groups.append([
                {"label":item,"value":item} 
                for item in df['best_method'].unique()
                ])
            select_options.append(df['best_method'].unique()[0])
            arg2 = extra_args2.copy()
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":1,
                 'extra_arg':json.dumps(arg2)}
                )
            
        output = html.Div(
            utility_func.add_row_choices(
                titles,
                option_groups,
                select_options,
                group_ids,
                persistence=[database_name,database_name],
                persistence_type="session",
                style={'font-size':'0.9em'},
                class_name='mb-1')[0],
            id={'code':'p1t3','tab3_figure_tabs':f'{tab}',
                "plot_type":json.dumps(["line","line","line","line","line",
                                        "line","line","line","line","line",
                                        "line"]),
                'xaxis_type':json.dumps([["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"]]),
                'preset':json.dumps([
                    {'dataframe':None,
                     'plot':{'x':'date','y':['S0','S0_forecast']},
                     'layout':{'layout_title_text':"Forecast S",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast S (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][0],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['V0','V0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast V (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['I0','I0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast I (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][2],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['D0','D0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast D (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][3],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_case_non_vaccinated',
                                             'daily_case_non_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast none-vaccinated daily case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_case_vaccinated',
                                             'daily_case_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast vaccinated daily case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['death_avg','death_avg_forecast']},
                     'layout':{'layout_title_text':"Forecast daily death case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['recovery_case','recovery_case_forecast']},
                     'layout':{'layout_title_text':"Forecast daily recovery case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_full_vaccinated',
                                             'new_full_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast daily new full vaccinated",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_boost_req',
                                             'new_boost_req_forecast']},
                     'layout':{'layout_title_text':"Forecast daily booster required",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['S','V','I']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Equilibrium plot (daily_R0:{} - average_R0:{})'"
                                +".format(*np.round(df[['daily_R0','avg_R0']].head(1)"+
                                ".to_numpy().squeeze(),2))")]}},
                                     ]),
                'grid':json.dumps({'grid_arg':{'gutter':0},
                                   'col_arg':[{'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':12}]
                                   })
                }
        ),
    elif database_name in ["vn_prediction_by_province",
                           "vn_prediction_by_province_mod"]:
        if database_name=="vn_prediction_by_province":
            df = s2
            tab = 't1'
        else:
            df = u2
            tab = 't2'
        titles = ['Province:','Model:','Best model:']
        option_groups = [
            [{"label":province_code[str(item)],"value":item} 
             for item in df['id'].unique()]
            ]
        select_options = [df['id'].unique()[0]]
        group_ids = [
            {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                                      'plot_arg':None,'layout_arg':None,
                                      'code_obj':'{}.loc[{}["id"]=={}]',
                                      'obj_type':'expression',
                                      'format':['df','df','variable']})},
            ]
        extra_args1 = {
            'target':['dataframe'],
            'plot_arg':None,'layout_arg':None,
            'code_obj':'{}.loc[[{}["method"].isin([np.NaN,{}]) if "method" in {}.columns else {}.index][0]] if {} !="None" else {}',
            'obj_type':'expression',
            'format':['df','df','variable','df','df','variable','df']}
        extra_args2 = {
            'target':['dataframe'],
            'plot_arg':None,'layout_arg':None,
            'code_obj':'{}.loc[[{}["best_method"].isin([np.NaN,{}]) if "best_method" in {}.columns else {}.index][0]] if {} !="None" else {}',
            'obj_type':'expression',
            'format':['df','df','variable','df','df','variable','df']}
        
        check_df = df.loc[df['id'].isin([df['id'].unique()[0]])].copy()
        no_best  = np.all(np.isnan(check_df['best_method'].unique()))
        
        if no_best:
            option_groups.append([
                {"label":item,"value":item} 
                for item in check_df['method'].unique()
                ])
            select_options.append(check_df['method'].unique()[0])
            arg1 = extra_args1.copy()
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":1,
                 'extra_arg':json.dumps(arg1)}
                )
       
            option_groups.append([{"label":'None',"value":'None'}])
            select_options.append('None')
            arg2 = extra_args2.copy()
            # arg2['target']=[None]
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":2,
                 'extra_arg':json.dumps(arg2)}
                )
        else:
            option_groups.append([{"label":'None',"value":'None'}])
            select_options.append('None')
            arg1 = extra_args1.copy()
            # arg1['target']=[None]
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":1,
                 'extra_arg':json.dumps(arg1)}
                )
            
            option_groups.append([
                {"label":item,"value":item} 
                for item in check_df['best_method'].unique()
                ])
            select_options.append(check_df['best_method'].unique()[0])
            arg2 = extra_args2.copy()
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":2,
                 'extra_arg':json.dumps(arg2)}
                )
        output = html.Div(
            utility_func.add_row_choices(
                titles,
                option_groups,
                select_options,
                group_ids,
                persistence=[database_name,database_name,database_name],
                persistence_type="session",
                style={'font-size':'0.9em'},
                class_name='mb-1')[0],
            id={'code':'p1t3','tab3_figure_tabs':f'{tab}',
                "plot_type":json.dumps(["line","line","line","line","line",
                                        "line","line","line","line","line",
                                        "line"]),
                'xaxis_type':json.dumps([["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"]]),
                'preset':json.dumps([
                    {'dataframe':None,
                     'plot':{'x':'date','y':['S0','S0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast S (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][0],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['V0','V0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast V (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['I0','I0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast I (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][2],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['D0','D0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast D (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][3],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_case_non_vaccinated',
                                             'daily_case_non_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast none-vaccinated daily case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_case_vaccinated',
                                             'daily_case_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast vaccinated daily case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['death_avg','death_avg_forecast']},
                     'layout':{'layout_title_text':"Forecast daily death case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['recovery_case','recovery_case_forecast']},
                     'layout':{'layout_title_text':"Forecast daily recovery case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_full_vaccinated',
                                             'new_full_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast daily new full vaccinated",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_boost_req',
                                             'new_boost_req_forecast']},
                     'layout':{'layout_title_text':"Forecast daily booster required",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['S','V','I']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Equilibrium plot (daily_R0:{} - average_R0:{})'"
                                +".format(*np.round(df[['daily_R0','avg_R0']].head(1)"+
                                ".to_numpy().squeeze(),2))")]}},
                                     ]),
                'grid':json.dumps({'grid_arg':{'gutter':0},
                                   'col_arg':[{'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':12}]
                                   })
                }
        ),
    elif database_name in ["w_prediction","w_prediction_mod"]:
        if database_name=="w_prediction":
            df = s3
            tab = 't1'
        else:
            df = u3
            tab = 't2'
         
        titles = ['Model:','Best model:']
        no_best  = np.all(np.isnan(df['best_method'].unique()))
        option_groups = []
        select_options = []
        group_ids = []
        extra_args1 = {
            'target':['dataframe'],
            'plot_arg':None,'layout_arg':None,
            'code_obj':'{}.loc[[{}["method"].isin([np.NaN,{}]) if "method" in {}.columns else {}.index][0]]',
            'obj_type':'expression',
            'format':['df','df','variable','df','df']}
        extra_args2 = {
            'target':['dataframe'],
            'plot_arg':None,'layout_arg':None,
            'code_obj':'{}.loc[[{}["best_method"].isin([np.NaN,{}]) if "best_method" in {}.columns else {}.index][0]]',
            'obj_type':'expression',
            'format':['df','df','variable','df','df']}
        if no_best:
            option_groups.append([
                {"label":item,"value":item} 
                for item in df['method'].unique()
                ])
            select_options.append(df['method'].unique()[0])
            arg1 = extra_args1.copy()
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":0,
                 'extra_arg':json.dumps(arg1)}
                )
            
            option_groups.append([{"label":'None',"value":'None'}])
            select_options.append('None')
            arg2 = extra_args2.copy()
            arg2['target']=[None]
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":1,
                 'extra_arg':json.dumps(arg2)}
                )
            
        else:
            option_groups.append([{"label":'None',"value":'None'}])
            select_options.append('None')
            arg1 = extra_args1.copy()
            arg1['target']=[None]
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":0,
                 'extra_arg':json.dumps(arg1)}
                )
            
            option_groups.append([
                {"label":item,"value":item} 
                for item in df['best_method'].unique()
                ])
            select_options.append(df['best_method'].unique()[0])
            arg2 = extra_args2.copy()
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":1,
                 'extra_arg':json.dumps(arg2)}
                )

        output = html.Div(
            utility_func.add_row_choices(
                titles,
                option_groups,
                select_options,
                group_ids,
                persistence=[database_name,database_name],
                persistence_type="session",
                style={'font-size':'0.9em'},
                class_name='mb-1')[0],
            id={'code':'p1t3','tab3_figure_tabs':f'{tab}',
                "plot_type":json.dumps(["line","line","line","line","line",
                                        "line","line","line","line","line",
                                        "line"]),
                'xaxis_type':json.dumps([["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"]]),
                'preset':json.dumps([
                    {'dataframe':None,
                     'plot':{'x':'date','y':['S0','S0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast S (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][0],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['V0','V0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast V (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['I0','I0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast I (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][2],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['D0','D0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast D (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][3],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_case_non_vaccinated',
                                             'daily_case_non_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast none-vaccinated daily case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_case_vaccinated',
                                             'daily_case_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast vaccinated daily case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['death_avg','death_avg_forecast']},
                     'layout':{'layout_title_text':"Forecast daily death case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['recovery_case','recovery_case_forecast']},
                     'layout':{'layout_title_text':"Forecast daily recovery case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_full_vaccinated',
                                             'new_full_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast daily new full vaccinated",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_boost_req',
                                             'new_boost_req_forecast']},
                     'layout':{'layout_title_text':"Forecast daily booster required",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['S','V','I']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Equilibrium plot (daily_R0:{} - average_R0:{})'"
                                +".format(*np.round(df[['daily_R0','avg_R0']].head(1)"+
                                ".to_numpy().squeeze(),2))")]}},
                                     ]),
                'grid':json.dumps({'grid_arg':{'gutter':0},
                                   'col_arg':[{'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':12}]
                                   })
                }
        ),
    elif database_name in ["w_prediction_by_country",
                           "w_prediction_by_country_mod"]:
        if database_name=="w_prediction_by_country":
            df = s4
            tab = 't1'
        else:
            df = u4
            tab = 't2'
            
        titles = ['Country & Area:','Model:','Best model:']
        option_groups = [
            [{"label":iso_code_dict[item],"value":item} 
             for item in df['iso_code'].unique()]
            ]
        select_options = [df['iso_code'].unique()[0]]
        group_ids = [
            {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":0,
              'extra_arg':json.dumps({'target':['dataframe'],
                                      'plot_arg':None,'layout_arg':None,
                                      'code_obj':'{}.loc[{}["iso_code"]=={}]',
                                      'obj_type':'expression',
                                      'format':['df','df','variable']})},
            ]
        extra_args1 = {
            'target':['dataframe'],
            'plot_arg':None,'layout_arg':None,
            'code_obj':'{}.loc[[{}["method"].isin([np.NaN,{}]) if "method" in {}.columns else {}.index][0]] if {} !="None" else {}',
            'obj_type':'expression',
            'format':['df','df','variable','df','df','variable','df']}
        extra_args2 = {
            'target':['dataframe'],
            'plot_arg':None,'layout_arg':None,
            'code_obj':'{}.loc[[{}["best_method"].isin([np.NaN,{}]) if "best_method" in {}.columns else {}.index][0]] if {} !="None" else {}',
            'obj_type':'expression',
            'format':['df','df','variable','df','df','variable','df']}
        
        check_df = df.loc[df['iso_code'].isin([df['iso_code'].unique()[0]])].copy()
        no_best  = np.all(np.isnan(check_df['best_method'].unique()))
        
        if no_best:
            option_groups.append([
                {"label":item,"value":item} 
                for item in check_df['method'].unique()
                ])
            select_options.append(check_df['method'].unique()[0])
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":1,
                 'extra_arg':json.dumps(extra_args1)}
                )
       
            option_groups.append([{"label":'None',"value":'None'}])
            select_options.append('None')
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":2,
                 'extra_arg':json.dumps(extra_args2)}
                )
        else:
            option_groups.append([{"label":'None',"value":'None'}])
            select_options.append('None')
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":1,
                 'extra_arg':json.dumps(extra_args2)}
                )
            
            option_groups.append([
                {"label":item,"value":item} 
                for item in check_df['best_method'].unique()
                ])
            select_options.append(check_df['best_method'].unique()[0])
            group_ids.append(
                {'code':'p1t3','tab3_figure_tabs':f'{tab}',"parameters":2,
                 'extra_arg':json.dumps(extra_args2)}
                )
            
        output = html.Div(
            utility_func.add_row_choices(
                titles,
                option_groups,
                select_options,
                group_ids,
                persistence=[database_name,database_name,database_name],
                persistence_type="session",
                style={'font-size':'0.9em'},
                class_name='mb-1')[0],
            id={'code':'p1t3','tab3_figure_tabs':f'{tab}',
                "plot_type":json.dumps(["line","line","line","line","line",
                                        "line","line","line","line","line",
                                        "line"]),
                'xaxis_type':json.dumps([["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"],["date","%Y-%m-%d"],
                                         ["date","%Y-%m-%d"]]),
                'preset':json.dumps([
                    {'dataframe':None,
                     'plot':{'x':'date','y':['S0','S0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast S (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][0],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['V0','V0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast V (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['I0','I0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast I (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][2],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['D0','D0_forecast']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Forecast D (score:{})'"+
                                ".format(ffs(df['score'].values[-1],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.float64]"+
                                "else (ffs(df['score'].values[-1][3],precision=3) "+
                                "if type(df['score'].values[-1]) in [np.ndarray]"+
                                "else 'N/A'))")]}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_case_non_vaccinated',
                                             'daily_case_non_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast none-vaccinated daily case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['daily_case_vaccinated',
                                             'daily_case_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast vaccinated daily case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['death_avg','death_avg_forecast']},
                     'layout':{'layout_title_text':"Forecast daily death case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['recovery_case','recovery_case_forecast']},
                     'layout':{'layout_title_text':"Forecast daily recovery case",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_full_vaccinated',
                                             'new_full_vaccinated_forecast']},
                     'layout':{'layout_title_text':"Forecast daily new full vaccinated",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['new_boost_req',
                                             'new_boost_req_forecast']},
                     'layout':{'layout_title_text':"Forecast daily booster required",
                               'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10}},
                    {'dataframe':None,
                     'plot':{'x':'date','y':['S','V','I']},
                     'layout':{'layout_title_x':0.5,'layout_legend_y':-0.5,
                               'layout_legend_x':0,'layout_margin_r':10,
                               'special_layout':[('layout_title_text',
                                "'Equilibrium plot (daily_R0:{} - average_R0:{})'"
                                +".format(*np.round(df[['daily_R0','avg_R0']].head(1)"+
                                ".to_numpy().squeeze(),2))")]}},
                                     ]),
                'grid':json.dumps({'grid_arg':{'gutter':0},
                                   'col_arg':[{'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':6},{'span':6},
                                              {'span':6},{'span':12}]
                                   })
                }
        ),
    else:
        output = []
    return [wrapper,output,trigger_count+1]

@callback(
    Output({"tab3_figure_tabs":'t2'},'children'),
    Input('tab3_processing_alert','children'),
    State({'code':'p1t3','name':ALL,"data_type":"update"},'data'),
    )
def tab3_update_database_dropdown(trigger,processed_databases):
    """
    Update the Modified tab if  new database is processed.
    """
    database_label = {'vn_prediction_mod':"Vietnam prediction data",
                      'vn_prediction_by_province_mod':'Vietnam prediction data by province',
                      'w_prediction_mod':'World prediction data',
                      'w_prediction_by_country_mod':'World prediction data by country'}
    
    option_lists={}
    data_list = np.array(ctx.states_list[0],dtype='O')
    data_list = data_list[[0,1,4,5]].tolist()
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
            [{"tab3_figure_tabs":'t2','type':'select_data',"data_type":"update"}],
            style={'font-size':'0.9em'},
            class_name='mb-1')[0]
        output = [output,
                  dbc.Col([dbc.Label('Plot points:',id='tab3_plot_point_label2',
                                     style={'font-size':'0.9em'},width='auto'),
                           dbc.Col(dbc.Input(id={"tab3_figure_tabs":'t2','type':'plot_point'},
                                             type='number',size='sm',
                                             min=2,step=1,value=30,required=True,
                                             className="mt-1 mb-1"),
                                   width=2),
                           dbc.Tooltip(dcc.Markdown(
                               '''The number of data points prior to 
                               'start date' that will be plotted.''',
                               style={'white-space':'normal',
                                      'text-align':'left'}),
                                   target='tab3_plot_point_label2',
                                   delay={'show': 1000}),
                           ])
                  ]
        return output

@callback(
      [Output({'tab3_figure_tabs':MATCH,'name':'tab3_figure_plot_trigger_step1'},
            'data'),
      Output({'code':'p1t3','tab3_figure_tabs':MATCH,"parameters":1,
            "extra_arg":ALL},'options'),
      Output({'code':'p1t3','tab3_figure_tabs':MATCH,"parameters":2,
              "extra_arg":ALL},'options'),
      ],
     
    Input('tab3_figure_plot_trigger','data'),
    Input({'code':'p1t3','tab3_figure_tabs':MATCH,"parameters":0,
            "extra_arg":ALL},'value'),
    State({'tab3_figure_tabs':MATCH,'type':'select_data',"data_type":ALL},'value'),
    #server_files
    State({'code':'p1t3','name':'vn_prediction_by_province',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'w_prediction_by_country',"data_type":"server"},'data'),
    #modified_files
    State({'code':'p1t3','name':'vn_prediction_by_province_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'w_prediction_by_country_mod',"data_type":"update"},'data'),
    
    State({'tab3_figure_tabs':MATCH,'name':'tab3_figure_plot_trigger_step1'},
            'data'),
    prevent_initial_call=True
    )
def tab3_modify_available_methods(trigger,param1,database_name,
                                  s1,s2,
                                  u1,u2,
                                  count):
    output_list=[]
    for item in ctx.outputs_list:
        if type(item)!=list:
            output_list.append(dash.no_update)
        else:
            output_list.append([dash.no_update for _ in range(len(item))])
    
    if database_name==[]:
        return output_list
    else:
        database_name = database_name[0]
    if param1==[]:
        return output_list
    
    if database_name in ['vn_prediction','w_prediction','vn_prediction_mod',
                          'w_prediction_mod']:
        triggered = [t["prop_id"] for t in ctx.triggered]
        output_list[0] = count + 1
        return output_list
    else:
        triggered = [t["prop_id"] for t in ctx.triggered]
        code = param1[0]
        directory = FileSystemStore(cache_dir="./cache/output_cache",
                                    default_timeout=0)
        server_directory = FileSystemStore(cache_dir="./cache/server_cache",
                                    default_timeout=0)
        if database_name=='vn_prediction_by_province':
            df = server_directory.get(s1)
        elif database_name=='w_prediction_by_country':
            df = server_directory.get(s2)
        elif database_name=='vn_prediction_by_province_mod':
            df = directory.get(u1)
        elif database_name=='w_prediction_by_country_mod':
            df = directory.get(u2)
        if 'id' in df.columns:
            check_array = df.loc[df['id']==int(code),'method'].unique()
            method_array = df.loc[df['id']==int(code),'best_method'].unique()
        else:
            check_array = df.loc[df['iso_code']==code,'method'].unique()
            method_array = df.loc[df['iso_code']==code,'best_method'].unique()
        if np.all(np.isnan(check_array)):
            output_list[1]=[[{"label":'None',"value":'None'}]]
            output_list[2]=[[{"label":method_array[0],"value":method_array[0]}]]
        else:
            output_list[1]=[[
                {"label":item,"value":item} 
                for item in check_array]]
            output_list[2]=[[{"label":'None',"value":'None'}]]
            
        output_list[0] = count + 1
        return output_list

@callback(
    [Output({'tab3_figure_tabs':MATCH,'name':'tab3_figure_plot_trigger_step2'},
          'data'),
     Output({'code':'p1t3','tab3_figure_tabs':MATCH,"parameters":1,
             "extra_arg":ALL},'value'),
     Output({'code':'p1t3','tab3_figure_tabs':MATCH,"parameters":2,
             "extra_arg":ALL},'value'),
     ],
    Input({'tab3_figure_tabs':MATCH,'name':'tab3_figure_plot_trigger_step1'},
          'data'),
    Input({'code':'p1t3','tab3_figure_tabs':MATCH,"parameters":1,
            "extra_arg":ALL},'value'),
    Input({'code':'p1t3','tab3_figure_tabs':MATCH,"parameters":2,
            "extra_arg":ALL},'value'),
    State({'code':'p1t3','tab3_figure_tabs':MATCH,"parameters":1,
            "extra_arg":ALL},'options'),
    State({'code':'p1t3','tab3_figure_tabs':MATCH,"parameters":2,
            "extra_arg":ALL},'options'),
    State({'tab3_figure_tabs':MATCH,'type':'select_data',"data_type":ALL},'value'),
    State({'tab3_figure_tabs':MATCH,'name':'tab3_figure_plot_trigger_step2'},
          'data'),
    prevent_initial_call=True
    )
def tab3_assign_option(trigger,param2,param3,option2,option3,database_name,count):
    triggered = [t["prop_id"] for t in ctx.triggered]
    
    output_list=[]
    for item in ctx.outputs_list:
        if type(item)!=list:
            output_list.append(dash.no_update)
        else:
            output_list.append([dash.no_update for _ in range(len(item))])
    
    if database_name==[]:
        return output_list
    else:
        database_name = database_name[0]
    trigger_check = any([True for item in triggered 
                         if "tab3_figure_plot_trigger_step1" in item])
    if (trigger_check and database_name in ['vn_prediction_by_province',
                                           'w_prediction_by_country',
                                           'vn_prediction_by_province_mod',
                                           'w_prediction_by_country_mod']):
        output_list[1]=[option2[0][0]['value']]
        output_list[2]=[option3[0][0]['value']]
    output_list[0] = count + 1
    return output_list
    
@callback_extension(
    Output({'type':'figure_wrapper','tab3_figure_tabs':MATCH},'children'),
    Input({'tab3_figure_tabs':MATCH,'name':'tab3_figure_plot_trigger_step2'},'data'), #trigger
    Input({"tab3_figure_tabs":MATCH,'type':'plot_point'},'value'),
    State({'code':'p1t3','tab3_figure_tabs':MATCH,"parameters":ALL,
            "extra_arg":ALL},'value'),
    State('tab3_start_date','value'),
    State({"code":"p1t3","name":"tab3_days","type":"input"},'value'),
    State('tab3_plot_tabs','active_tab'),
    State({'tab3_figure_tabs':MATCH,'type':'select_data',"data_type":ALL},'value'),
    State({'code':'p1t3',"tab3_figure_tabs":MATCH,"plot_type":ALL,'xaxis_type':ALL,
           "preset":ALL,'grid':ALL},'id'),
    #server_files
    State({'code':'p1t2','name':'vn_modeling_data',"data_type":"server"},'data'),
    State({'code':'p1t2','name':'vn_modeling_data_by_province',"data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_modeling_data',"data_type":"server"},'data'),
    State({'code':'p1t2','name':'w_modeling_data_by_country',"data_type":"server"},'data'),
    
    State({'code':'p1t3','name':'vn_prediction',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'vn_prediction_by_province',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'vn_equilibrium',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'vn_equilibrium_by_province',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'w_prediction',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'w_prediction_by_country',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'w_equilibrium',"data_type":"server"},'data'),
    State({'code':'p1t3','name':'w_equilibrium_by_country',"data_type":"server"},'data'),
    #modified_files
    State({'code':'p1t3','name':'vn_modeling_data_mod',"data_type":"buffer"},'data'),
    State({'code':'p1t3','name':'vn_modeling_data_by_province_mod',
           "data_type":"buffer"},'data'),
    State({'code':'p1t3','name':'w_modeling_data_mod',"data_type":"buffer"},'data'),
    State({'code':'p1t3','name':'w_modeling_data_by_country_mod',
           "data_type":"buffer"},'data'),
    
    State({'code':'p1t3','name':'vn_prediction_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'vn_prediction_by_province_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'vn_equilibrium_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'vn_equilibrium_by_province_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'w_prediction_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'w_prediction_by_country_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'w_equilibrium_mod',"data_type":"update"},'data'),
    State({'code':'p1t3','name':'w_equilibrium_by_country_mod',"data_type":"update"},'data'),
    prevent_initial_call=True
    )
def tab3_plot(trigger,plot_point,param_value,start_date,forecast_points,
              active_tab,database_name,item_id,
              t2s1,t2s2,t2s3,t2s4,
              s1,s2,s3,s4,s5,s6,s7,s8,
              t2b1,t2b2,t2b3,t2b4,
              u1,u2,u3,u4,u5,u6,u7,u8):
    """
    Plot databases based on input format from 'id' and select components
    If user want the function to handle multiple dataframe for plotting 
    in a single page. Define 'df' in a tuple variable, with 1st item is list
    of usage dataframe, 2nd item is index list which dataframe is used.
    """
    if param_value==[]:
        return []
    
    if database_name==[]:
        return []
    else:
        database_name = database_name[0]
    extra_input = ctx.states_list[0]
    extra_plot_arg = [json.loads(item['id']['extra_arg']) for item in extra_input]
    extra_def_value = [item for item in param_value]
    preset = json.loads(item_id[0]['preset'])
    plot_type = json.loads(item_id[0]['plot_type'])
    xaxis_type = json.loads(item_id[0]['xaxis_type'])
    grid = json.loads(item_id[0]['grid'])
    
    if plot_point is None:
        return dash.no_update
    rename_group = {
        'S0':'S0_forecast','V0':'V0_forecast','I0':'I0_forecast','D0':'D0_forecast',
        'daily_case_non_vaccinated':'daily_case_non_vaccinated_forecast',
        'daily_case_vaccinated':'daily_case_vaccinated_forecast',
        'death_avg':'death_avg_forecast','recovery_case':'recovery_case_forecast',
        'new_full_vaccinated':'new_full_vaccinated_forecast',
        'new_boost_req':'new_boost_req_forecast'}
    if database_name=="vn_prediction":
        df1 = t2s1.tail(plot_point)
        df1.insert(1,'iso_code','VNM')
        df2 = s1
        df2.rename(columns=rename_group,inplace=True)
        df1 = df1.set_index(['date','iso_code'])
        df2 = df2.set_index(['date','iso_code'])
        join_df = pd.concat([df1,df2])
        df = join_df.reset_index()
        equil_df = s3
        df = ([df,equil_df],[0,0,0,0,0,0,0,0,0,0,1])
    elif database_name=="vn_prediction_by_province":
        df1 = t2s2.groupby(['id']).tail(plot_point)
        df2 = s2
        df2.rename(columns=rename_group,inplace=True)
        df1 = df1.set_index(['date','id'])
        df2 = df2.set_index(['date','id'])
        join_df = pd.concat([df1,df2])
        df = join_df.reset_index()
        equil_df = s4
        df = ([df,equil_df],[0,0,0,0,0,0,0,0,0,0,1])
    elif database_name=="w_prediction":
        df1 = t2s3.tail(plot_point)
        df2 = s5
        df2.rename(columns=rename_group,inplace=True)
        df1 = df1.set_index(['date','iso_code'])
        df2 = df2.set_index(['date','iso_code'])
        join_df = pd.concat([df1,df2])
        df = join_df.reset_index()
        equil_df = s7
        df = ([df,equil_df],[0,0,0,0,0,0,0,0,0,0,1])
    elif database_name=="w_prediction_by_country":
        df1 = t2s4.groupby(['iso_code']).tail(plot_point)
        df2 = s6
        df2.rename(columns=rename_group,inplace=True)
        df1 = df1.set_index(['date','iso_code'])
        df2 = df2.set_index(['date','iso_code'])
        join_df = pd.concat([df1,df2])
        df = join_df.reset_index()
        equil_df = s8
        df = ([df,equil_df],[0,0,0,0,0,0,0,0,0,0,1])
    elif database_name=="vn_prediction_mod":
        if start_date=='auto':
            df1 = t2b1.tail(plot_point)
        else:
            date1 = pd.to_datetime(start_date)-datetime.timedelta(plot_point)
            date2 = pd.to_datetime(start_date)+datetime.timedelta(forecast_points)
            df1 = t2b1.loc[(t2b1.date>date1)&(t2b1.date<=date2)]
        df1.insert(1,'iso_code','VNM')
        df2 = u1
        df2.rename(columns=rename_group,inplace=True)
        df1 = df1.set_index(['date','iso_code'])
        df2 = df2.set_index(['date','iso_code'])
        join_df = pd.concat([df1,df2])
        df = join_df.reset_index()
        equil_df = u3
        df = ([df,equil_df],[0,0,0,0,0,0,0,0,0,0,1])
    elif database_name=="vn_prediction_by_province_mod":
        if start_date=='auto':
            df1 = t2b2.groupby(['id']).tail(plot_point)
        else:
            date1 = pd.to_datetime(start_date)-datetime.timedelta(plot_point)
            date2 = pd.to_datetime(start_date)+datetime.timedelta(forecast_points)
            df1 = t2b2.groupby(['id'], group_keys=False).apply(
                lambda x:x.loc[(x.date>date1)&(x.date<=date2)])
        df2 = u2
        df2.rename(columns=rename_group,inplace=True)
        df1 = df1.set_index(['date','id'])
        df2 = df2.set_index(['date','id'])
        join_df = pd.concat([df1,df2])
        df = join_df.reset_index()
        equil_df = u4
        df = ([df,equil_df],[0,0,0,0,0,0,0,0,0,0,1])
    elif database_name=="w_prediction_mod":
        if start_date=='auto':
            df1 = t2b3.tail(plot_point)
        else:
            date1 = pd.to_datetime(start_date)-datetime.timedelta(plot_point)
            date2 = pd.to_datetime(start_date)+datetime.timedelta(forecast_points)
            df1 = t2b3.loc[(t2b3.date>date1)&(t2b3.date<=date2)]
        df2 = u5
        df2.rename(columns=rename_group,inplace=True)
        df1 = df1.set_index(['date','iso_code'])
        df2 = df2.set_index(['date','iso_code'])
        join_df = pd.concat([df1,df2])
        df = join_df.reset_index()
        equil_df = u7
        df = ([df,equil_df],[0,0,0,0,0,0,0,0,0,0,1])
    elif database_name=="w_prediction_by_country_mod":
        if start_date=='auto':
            df1 = t2b4.groupby(['iso_code']).tail(plot_point)
        else:
            date1 = pd.to_datetime(start_date)-datetime.timedelta(plot_point)
            date2 = pd.to_datetime(start_date)+datetime.timedelta(forecast_points)
            df1 = t2b4.groupby(['iso_code'], group_keys=False).apply(
                lambda x:x.loc[(x.date>date1)&(x.date<=date2)])
        df2 = u6
        df2.rename(columns=rename_group,inplace=True)
        df1 = df1.set_index(['date','iso_code'])
        df2 = df2.set_index(['date','iso_code'])
        join_df = pd.concat([df1,df2])
        df = join_df.reset_index()
        equil_df = u8
        df = ([df,equil_df],[0,0,0,0,0,0,0,0,0,0,1])
    output = []
    plot_output = []
    store_output = []
    for item in enumerate(zip(plot_type,preset,xaxis_type)):
        if type(df)!=tuple:
            fig = utility_func.condition_plot(item[1][0],
                                              df,extra_plot_arg,extra_def_value,
                                              item[1][1])
        else:
            df_idx = df[1][item[0]]
            fig = utility_func.condition_plot(item[1][0],
                                              df[0][df_idx],
                                              extra_plot_arg,extra_def_value,
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
                              'tab3_figure_tabs':item_id[0]['tab3_figure_tabs'],
                              'plot_idx':item[0],
                              'xaxis_type':json.dumps(item[1][2]),
                              'plot_args':None if plot_handler==None 
                              else json.dumps(plot_handler)
                              }))
        else:
            plot_output.append(dmc.Col(children=
                dcc.Graph(figure=fig,
                          id={'code':item_id[0]['code'],
                              'tab3_figure_tabs':item_id[0]['tab3_figure_tabs'],
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
                              'tab3_figure_tabs':item_id[0]['tab3_figure_tabs'],
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

##modeling data-------------------------------------------------------------

@callback_extension(
    output=ServersideOutput('tab3_process_params','data'),
    inputs=dict(
        process_click=Input('tab3_model_Apply_Button','n_clicks'),
        parameter_check = State('tab3_inputs_alert','is_open'),
        ##DATABASE PARAMS
        #output uids
        output_uid = State({'code':'p1t3','name':ALL,"data_type":"update"},'data'),
        buffer_uid = State({'code':'p1t3','name':ALL,"data_type":"buffer"},'data'),
        #chose all database
        chosen_all = State('tab3_model_database_all_switch','value'),
        #chose Vietnam|world database
        chosen_database = State('tab3_model_database','value'),
        #data by province|countries
        data_separation = State('tab3_model_database_division_switch','value'),
        #chosen provinces/countries code
        chosen_sections = State('tab3_model_database_division_checklist','value'),
        #default|latest processed database
        w_default = State({'code':'p1t2','name':'select_w_database',
                            "purpose":"database_check"},'value'),
        w_country_default = State({'code':'p1t2','name':'select_wbc_database',
                                    "purpose":"database_check"},'value'),
        vn_default = State({'code':'p1t2','name':'select_vn_database',
                            "purpose":"database_check"},'value'),
        vn_province_default = State({'code':'p1t2','name':'select_vnbp_database',
                                      "purpose":"database_check"},'value'),
        #census manual setting
        p = State({"code":"p1t3","name":"tab3_p","type":"input"},'value'),
        auto_b_d_rate = State('tab3_auto_b_d_rate','value'),
        manual_b_rate = State({"code":"p1t3","name":"tab3_manual_b_d_rate",
            "type":"special_input","index":0},'value'),
        manual_d_rate = State({"code":"p1t3","name":"tab3_manual_b_d_rate",
            "type":"special_input","index":1},'value'),
        
        #modeling inputs
        start_date = State('tab3_start_date','value'),
        predict_days = State({"code":"p1t3","name":"tab3_days","type":"input"},'value'),
        model_number = State('tab3_model_number','value'),
        best_model = State('tab3_best_model','value'),
        model_score = State('tab3_scoring','value'),
        model_score_criteria = State('tab3_model_criteria','value'),
        model_score_output = State('tab3_multioutput','value'),
        r_protect_time = State({"code":"p1t3","name":"tab3_r_protect_time",
                                  "type":"input"},'value'),
        equil_days = State({"code":"p1t3","name":"tab3_n_days",
                                  "type":"input"},'value'),
        #GENERAL INPUTS
        max_worker=State({"code":"p1t3","name":"tab3_max_worker","type":"input"},'value'),
        ),
    prevent_initial_call=True
    )
def tab3_model_params_prepare(
        process_click,
        parameter_check,
        #output uids
        output_uid,buffer_uid,
        #database params
        chosen_all,chosen_database,data_separation,
        chosen_sections,
        w_default,w_country_default,vn_default,vn_province_default,
        p,auto_b_d_rate,manual_b_rate,manual_d_rate,
        #modeling inputs
        start_date,predict_days,model_number,best_model,model_score,
        model_score_criteria,model_score_output,r_protect_time,equil_days,
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
        outputs = dict(
            database_params=[
                output_uid,buffer_uid,chosen_all,chosen_database,data_separation,
                chosen_sections,w_default,w_country_default,
                vn_default,vn_province_default,p,auto_b_d_rate,manual_b_rate,
                manual_d_rate],
            model_params=[
                start_date,predict_days,model_number,best_model,model_score,
                model_score_criteria,model_score_output,r_protect_time,
                equil_days],
            general_params=max_worker
                       )
        return outputs

@callback(
    output=dict(
        process_result=Output('tab3_processing_alert','children'),
        option_colors=Output({'code':'p1t3','name':ALL,
                              "purpose":"database_check"},'labelClassName'),
    ),
    inputs = dict(
        initiator=Input('tab3_process_params','data'),
        
        #census input
        birth_uid=State({'code':'p1t3','name':'birth_data',
                          "data_type":"census"},'data'),
        death_uid = State({'code':'p1t3','name':'death_data',
                            "data_type":"census"},'data'),
        #data input
        vn_uid = State({'code':'p1t2','name':'vn_modeling_data',
                        "data_type":"server"},'data'),
        vn_mod_uid = State({'code':'p1t2','name':'vn_modeling_data_modified',
                            "data_type":"update"},'data'),
        vnbp_uid = State({'code':'p1t2','name':'vn_modeling_data_by_province',
                          "data_type":"server"},'data'),
        vnbp_mod_uid = State({'code':'p1t2','name':'vn_modeling_data_by_province_modified',
                              "data_type":"update"},'data'),
        w_uid = State({'code':'p1t2','name':'w_modeling_data',
                       "data_type":"server"},'data'),
        w_mod_uid = State({'code':'p1t2','name':'w_modeling_data_modified',
                           "data_type":"update"},'data'),
        bc_uid= State({'code':'p1t2','name':'w_modeling_data_by_country',
                            "data_type":"server"},'data'),
        bc_mod_uid = State({'code':'p1t2','name':'w_modeling_data_by_country_modified',
                            "data_type":"update"},'data'),
        extra_args = State('tab3_forecast_extra_args','data'),
        preset_orders = State('tab3_preset_orders','data'),
    ),
    background=True,
    running=[(Output('tab3_processing_status','is_open'),True,False)],
    progress=[Output('tab3_update_process_collapse','is_open'),
              Output('tab3_update_process','value'),
              Output('tab3_update_process_info','children')],
    progress_default=[False,0,'Processing...'],
    cancel=Input('tab3_model_Cancel_Button1','n_clicks'),
    prevent_initial_call=True)
def tab3_data_modeling(
    progress_status,initiator,
    birth_uid,death_uid,vn_uid,vn_mod_uid,vnbp_uid,vnbp_mod_uid,
    w_uid,w_mod_uid,bc_uid,bc_mod_uid,extra_args,preset_orders
    ):
    n_outputs = len(ctx.outputs_list[1])
    color_outputs = [dash.no_update for _ in range(n_outputs)]
    
    directory = FileSystemStore(cache_dir="./cache/output_cache",
                                default_timeout=0)
    server_directory = FileSystemStore(cache_dir="./cache/server_cache",
                                default_timeout=0)
    packed_params = directory.get(initiator)
    
    (output_uid,buffer_uid,chosen_all,chosen_database,data_separation,
    chosen_sections,w_default,w_country_default,
    vn_default,vn_province_default,p,auto_b_d_rate,manual_b_rate,
    manual_d_rate)=packed_params['database_params']
    
    (start_date,predict_days,model_number,best_model,model_score,
    model_score_criteria,model_score_output,r_protect_time,
    equil_days)=packed_params['model_params']
    
    (max_worker)=packed_params['general_params']
    
    #get data from uid
    birth_df = directory.get(birth_uid)
    death_df = directory.get(death_uid)
    
    #check start date
    if start_date=='auto':
        start_date=None
    
    #iterator
    options_index=[]
    storage_index=[]
    title_str1=[]
    title_str2=[]
    model_worker_range=[]
    model_database = []
    birth_input=[]
    death_input=[]
    unsucessful_process = 0
    if chosen_database=="vietnam" or chosen_all:
        if (not data_separation) or chosen_all:
            options_index.append(0)
            storage_index.append(0)
            title_str1.append("Vietnam")
            title_str2.append("")
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
            model_worker_range.append(['VNM'])
            if vn_default==1:
                vn_input_df = server_directory.get(vn_uid)
            else:
                vn_input_df = directory.get(vn_mod_uid)
            model_database.append((vn_input_df,'iso_code'))
        if data_separation or chosen_all:
            options_index.append(1)
            storage_index.append(1)
            title_str1.append("Vietnam")
            title_str2.append("by provinces")
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
            if vn_province_default==1:
                vn_bp_input_df = server_directory.get(vnbp_uid)
            else:
                vn_bp_input_df = directory.get(vnbp_mod_uid)
            
            if chosen_all:
                model_range = vn_bp_input_df['id'].unique().tolist()
            else:
                model_range = chosen_sections
            model_worker_range.append(model_range)
            model_database.append((vn_bp_input_df,'id'))
    if chosen_database=="world" or chosen_all:
        if (not data_separation) or chosen_all: #process total data
            options_index.append(2)
            storage_index.append(4)
            title_str1.append("World")
            title_str2.append("")
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
            model_worker_range.append(['OWID_WRL'])
            if w_default==1:
                w_input_df = server_directory.get(w_uid)
            else:
                w_input_df = directory.get(w_mod_uid)
            model_database.append((w_input_df,'iso_code'))
        if data_separation or chosen_all: #process by country
            options_index.append(3)
            storage_index.append(5)
            title_str1.append("World")
            title_str2.append("by countries")
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
            if w_country_default==1:
                bc_input_df = server_directory.get(bc_uid)
            else:
                bc_input_df = directory.get(bc_mod_uid)
            if chosen_all:
                model_range = bc_input_df['iso_code'].unique().tolist()
            else:
                model_range = chosen_sections
            model_worker_range.append(model_range)
            model_database.append((bc_input_df,'iso_code'))
    for input_group in zip(options_index,storage_index,title_str1,title_str2,
                           model_worker_range,model_database,birth_input,
                           death_input):
        (idx,store_idx,str1,str2,worker_range,model_input,birth_data,
         death_data)=input_group
        try:
            #the main code goes here
            progress_status((False,0,f'Proccessing {str1} modeling database {str2}...'))
            time.sleep(1)
            
            model_manager = Manager()
            model_counter= model_manager.Array('i',[0,0])
            model_time_list = model_manager.list([])
            model_error_dict = model_manager.dict()
            model_shared_output = model_manager.Namespace()
            model_shared_output.shared_df = pd.DataFrame()
            model_shared_output.equil_df = pd.DataFrame()
            model_lock = model_manager.Lock()
            input_orders = preset_orders
            if (preset_orders=='auto') and (idx in [1,3]):
                if idx==1:
                    if w_default==1:
                        refer_df = server_directory.get(w_uid)
                    else:
                        refer_df = directory.get(w_mod_uid)
                else:
                    if vn_default==1:
                        refer_df = server_directory.get(vn_uid)
                    else:
                        refer_df = directory.get(vn_mod_uid)
                if start_date is None:
                    get_date = refer_df.iloc[-1].date
                else:
                    get_date = start_date
                if extra_args is None:
                    input_orders=get_arima_order(
                        input_df=refer_df,
                        cols=['beta','beta_v','gamma','theta','alpha','alpha0',
                              'pi'],
                        start=refer_df.loc[refer_df.date==get_date].index[0])
                else:
                    n_slice = extra_args['train_p'] + extra_args['test_p']
                    test_p = extra_args['test_p']
                    information_criterion = extra_args['auto_arima_kwargs']['information_criterion']
                    
                    preset_args = {'n_slice':n_slice,
                                   'test_p':test_p,
                                   'information_criterion':information_criterion}
                    
                    input_orders=get_arima_order(
                        input_df=refer_df,
                        cols=['beta','beta_v','gamma','theta','alpha','alpha0',
                              'pi'],
                        start=refer_df.loc[refer_df.date==get_date].index[0],
                        **preset_args)
            elif (preset_orders=='auto') and (idx in [0,2]):
                input_orders = None
            SVID_extra_arg = {
                'predict_days':predict_days,
                'date_col':'date',
                'initial_state':['S0','V0','I0','D0'],
                'model_param':['beta','beta_v','gamma','theta','alpha','alpha0','pi'],
                'model_number':model_number,
                'best_model':best_model,
                'scoring':model_score,
                'multioutput':model_score_output,
                'greater_better':model_score_criteria,
                'preset_arima_order': input_orders, 
                'param_kwargs':extra_args
                }
            worker_func = partial(wrapper_SVID_predict,
                      shared_count=model_counter,
                      time_list = model_time_list,
                      error_dict=model_error_dict,
                      shared_output=model_shared_output,
                      lock=model_lock,
                      input_data=model_input,
                      start_date=start_date,
                      birth_data=birth_data,
                      death_data=death_data,
                      p=p,
                      r_protect_time=r_protect_time,
                      n_days = equil_days,
                      **SVID_extra_arg)
            if worker_range not in [['VNM'],['OWID_WRL']]:
                executor= Pool(processes=max_worker)
                pool_result = executor.map_async(worker_func,worker_range)
            else:
                worker_func(worker_range[0])
            max_count = len(worker_range)
            while model_counter[0]<max_count:
                time.sleep(1)
                n_done = model_counter[0]
                n_wait = max_count
                total_time = round(sum(model_time_list),2)
                if n_done!=0:
                    avg_time = round(total_time/n_done,2)
    
                if n_done!=0:
                    remain_section = n_wait - n_done
                    remain_time = round(remain_section*avg_time,2)
                    real_remain = round(remain_time/max_worker,2)
                else:
                    real_remain = "-- "
                progress_status((True,100*n_done/max_count,
                                  (f'Processing {str1} modeling data {str2}:'+
                                    f' {real_remain}s')
                                  ))
            #for debug logging if necessary
            model_error_result = dict(model_error_dict)
            model_time = list(model_time_list)
            model_df = model_shared_output.shared_df.copy()
            equil_df = model_shared_output.equil_df.copy()
            if model_df.shape[0]==0 or equil_df.shape[0]==0:
                raise Exception("Modeling result is empty.")
            if idx==1:
                model_df = model_df.sort_values(
                    ['id','method','date']).reset_index(drop=True)
                equil_df = equil_df.sort_values(
                    ['id','date']).reset_index(drop=True)
            elif idx==3:
                model_df = model_df.sort_values(
                    ['iso_code','method','date']).reset_index(drop=True)
                equil_df = equil_df.sort_values(
                    ['iso_code','date']).reset_index(drop=True)
                
            if worker_range not in [['VNM'],['OWID_WRL']]:
                executor.close()
                executor.join()
            model_manager.shutdown()
            
            time.sleep(1)
            directory.set(output_uid[store_idx],model_df)
            directory.set(output_uid[store_idx+2],equil_df)
            
            check_array = [vn_default,vn_province_default,
                           w_default,w_country_default]
            use_array = [(vn_uid,vn_mod_uid),(vnbp_uid,vnbp_mod_uid),
                         (w_uid,w_mod_uid),(bc_uid,bc_mod_uid)]
            if check_array[idx]==1:
                buffer_df = server_directory.get(use_array[idx][0])
            else:
                buffer_df = directory.get(use_array[idx][1])
            directory.set(buffer_uid[idx],buffer_df)
            
            progress_status((False,100,
                              (f'Processed {str1} modeling data {str2}')
                              ))
            time.sleep(1)
            color_outputs[idx]='btn btn-outline-success'
        except Exception as error:
            color_outputs[idx]='btn btn-outline-warning'
            unsucessful_process +=1
    if unsucessful_process==0:
        process_result="All databases processed successfully."
    else:
        process_result=f"{unsucessful_process} database(s) processed unsuccessfully."
    return dict(process_result=process_result,
                option_colors=color_outputs)

@callback(
    Output('tab3_processing_alert_collapse','is_open'),
    Input('tab3_processing_alert','children'),
    Input('tab3_model_Apply_Button','n_clicks'),
    Input('tab3_model_Download_Button','n_clicks'),
    prevent_initial_call=True
    )
def tab3_display_process_result(trigger1,trigger2,trigger3):
    """
    Show processing result.
    """
    if ctx.triggered_id=="tab3_processing_alert":
        return True
    elif ctx.triggered_id in ["tab3_model_Apply_Button",
                              "tab3_model_Download_Button"]:
        return False

##download data-------------------------------------------------------------
@callback(
    output=[
        Output('tab3_download','data')
        ],
    inputs = [
        Input('tab3_model_Download_Button','n_clicks'),
        #server_files
        State({'code':'p1t3','name':ALL,"data_type":"server"},'data'),
        #update_files
        State({'code':'p1t3','name':ALL,"data_type":"update"},'data'),
        ],
    background=True,
    running=[(Output('tab3_downloading_status','is_open'),True,False)],
    cancel=Input('tab3_model_Cancel_Button2','n_clicks'),
    prevent_initial_call=True)
def tab3_download_zip(trigger,server_uid,modified_uid):
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
    return [dcc.send_bytes(write_available_data,"modeling_data.zip",
                          server_uid=server_uid,modified_uid=modified_uid)
            ]