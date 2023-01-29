from dash import DiskcacheManager
from dash_extensions.enrich import DashProxy,FileSystemStore,ServersideOutputTransform
import dash_bootstrap_components as dbc
import diskcache
import pandas as pd
import os

#by default this will be used as backend for ServersideOutputTransform, modeling 
#output & user interaction files will be cached in this directory
my_backend = FileSystemStore(cache_dir="./cache/output_cache",threshold=300,
                             default_timeout=0
                             )

#server data will be cached here
server_backend = FileSystemStore(cache_dir="./cache/server_cache",threshold=100,
                             default_timeout=0
                             )

#census data will be cached here
census_backend = FileSystemStore(cache_dir="./cache/census_cache",threshold=50,
                             default_timeout=0
                             )

cache = diskcache.Cache("./cache/dash_cache")
background_callback_manager = DiskcacheManager(cache)

#added a date cache to see if data cache need to be update
last_cache = diskcache.Cache("./cache/update_date")
cache_keys = list(last_cache.iterkeys())
update_model_date = pd.read_csv("https://raw.githubusercontent.com/DeKhaos/"+
                           "covid-modeling/master/database/"+
                           "modified_model_data_time.csv").last_modified.to_list()[0]

update_census_date = pd.read_csv("https://raw.githubusercontent.com/DeKhaos/"+
                           "covid-modeling/master/database/"+
                           "modified_census_data_time.csv").last_modified.to_list()[0]

if cache_keys==[]:
    last_cache["last_model_modified"]=update_model_date
    last_cache["last_census_modified"]=update_census_date
else:
    #if new server data available, delete old one and read to download new one
    if update_model_date>last_cache["last_model_modified"]:
        last_cache["last_model_modified"]=update_model_date
        path = 'cache/server_cache'
        cache_base = server_backend._get_filename("__wz_cache_count").split(os.sep)[-1]
        list_dir = os.listdir(path)
        for file in list_dir:
            if file !=cache_base:
                os.remove(os.path.join(path, file))
    if update_census_date>last_cache["last_census_modified"]:
        last_cache["last_census_modified"]=update_census_date
        path = 'cache/census_cache'
        cache_base = census_backend._get_filename("__wz_cache_count").split(os.sep)[-1]
        list_dir = os.listdir(path)
        for file in list_dir:
            if file !=cache_base:
                os.remove(os.path.join(path, file))
        
app = DashProxy(__name__,external_stylesheets=[dbc.themes.CYBORG,
                                               dbc.icons.BOOTSTRAP],
                suppress_callback_exceptions=True,use_pages=True,
                background_callback_manager=background_callback_manager,
                transforms=[ServersideOutputTransform(backend=my_backend),
                            ],
                assets_folder='assets',
                pages_folder='pages')