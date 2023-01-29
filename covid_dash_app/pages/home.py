import dash
from dash import dcc, html


dash.register_page(__name__,path='/')

layout = html.Div([dcc.Markdown(
    """
    ### Basic guideline

    Currently, this applications allow download latest database from sources, processing raw data, and modeling processed data.

    **Note:** This application might use lots of cache storage on local machine, it is suggested to store the application on drives with good amount of space (1-2Gb).

    ---    
    #### Modeling page:
        
    - Preparation tab:

    By default, when starting the page, all server databases will be downloaded to `cache` folder and ready for viewing. If users need the latest data, they can attempt to download database with `Download` button.
    
    Before process to download, click the switch of databases which you want to update. After processing, the button color will display the process result.
    
    For Vietnam Covid-19 database, if you need just the total information, it's ok to just use the database from World Covid-19 database since it's included already. If you are interested in a certain Vietnam's province, you can select that province before download. Note that if you try to download all province databases, it'll be quite slow. Another note is downloading Vietnam's province databases require webdriver, so it will be installed to `.wdm` folder.
    
    - Processing tab:
    
    Chosen databases on `Preparation` tab will be used for processing, `Default` databases will be selected if not specified. Same as `Preparation` tab, you can select to process all databases or just choose some specific countries|Vietnam provinces.
    
    This step helps remove outlier, average the time sery from data and make it easier to visualize and find the pattern.
    
    Some assumptions worth mentions are:

    1. All vaccine types is considered to be fully vaccinated 2 doses, extra doses are all considered to be booster shot and will wear off after a period of time.
    2. Birth rate, death are are considered to be constant through all year long.
    3. No recovery case database is available. Therefore, it is all calculated and is affected by processing input parameters.
    4. Recovery & original fatality case distribibution calculation are only nearly random, since calculation will be very slow if fully randomness is taken into account. For more informaion, users can read functions inside the `covid_science.preparation` for more information.
    
    - Modeling tab:
    
    Chosen databases on `Processing` tab will be used for processing, `Default` databases will be selected if not specified. Same as `Processing` tab, you can select to process all databases or just choose some specific countries|Vietnam provinces.
    
    Modeling is calculated base on input model parameters and initial SVID state of the model. The aspect that affects the model the most is how model input parameters is defined. Currently 2 methods are used: SARIMA and regression line. It is calculated based on previous model parameter data points from `Processing` step then apply to the model.
    
    If users see the score values beside the figure title, although it's not possible to do scoring if predict values are out of available data range, that score is for reference. The score value is calculated by retract a number of day equal to `Number of forecast days`, apply model for prediction and compare with real data points. If forecast is made inside the range of database, then the score is comparing to real values.
    
    Equilibrium plot use the initial state of the `Start date` from `Forecasting inputs` to calculate the EE/DFE state of the model after a set period of time.
    """,
    style={'white-space':'normal',
           'text-align':'left'}
    )
                   
])