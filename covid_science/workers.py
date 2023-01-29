"""Worker functions for multiprocessing and threading."""

import time
import numpy as np
import pandas as pd
from multiprocessing.managers import AcquirerProxy,BaseListProxy,DictProxy,\
    NamespaceProxy,ArrayProxy
from scipy.integrate import odeint
import warnings
from covid_science.modeling import SVID_modeling,convert_to_daily_data
from covid_science.preparation import process_raw_infect_data,vaccinated_case
from covid_science.utility_func import SVID_ODE

#------------------------------------------------------------------
def vn_casualty_wrapper_func(province,
                             shared_count,
                             time_list,
                             error_dict,
                             shared_output,
                             lock,
                             request_object,
                             link,
                             api,
                             **kwargs):
    """
    The purpose of this function is for wrapping Vietnam covid-19 casualty data
    processing into one function to use for multiprocessing.Pool workers.

    Parameters
    ----------
    province : list
        ['province_name','code'].
        
    shared_count : multiprocessing.managers.ArrayProxy
        An array that counts [total processed provinces,
                              provinces that results in error when processing]
        
    time_list : multiprocessing.managers.BaseListProxy
        A shared variable to store time spent processing data.
        
    error_dict : multiprocessing.managers.DictProxy
        Return dict if errors found in workers.
        
    shared_output : multiprocessing.managers.NamespaceProxy
        Use to concatenate DataFrames.
        Have 1 output attribute, ['shared_df']
        
    lock : multiprocessing.managers.AcquirerProxy
        Lock for synchronizing.
        
    request_object : browser_request class
        Request object for data collection.
        
    link : str
        The website URL.
        
    api : str
        The API which to gather data.
        
    **kwargs :
        Extra keyword arguments for browser_request.get_web_response function.

    Returns
    -------
    None.
        The DataFrame is returned in 'shared_output' instead.
    """
    start_time = time.perf_counter()
    
    try:
        
        if not isinstance(shared_count,ArrayProxy):
            raise TypeError("'shared_count' type is not correct.")
        if not isinstance(time_list,BaseListProxy):
            raise TypeError("'time_list' type is not correct.")
        if not isinstance(error_dict,DictProxy):
            raise TypeError("'error_dict' type is not correct.")
        if not isinstance(shared_output,NamespaceProxy):
            raise TypeError("'shared_output' type is not correct.")
        if not isinstance(lock,AcquirerProxy):
            raise TypeError("'lock' type is not correct.")
    
        province_name=province[0]
        province_id=province[1]
        worker_link = link.format(province_id)
    
        json_object = request_object.get_web_response(website_link=worker_link,
                                                      api=api,
                                                      **kwargs)
        if json_object is None:
            raise ValueError(f"No data collected after {request_object.attempt}"
                             " attempt(s).")
        else:
            df = pd.json_normalize(json_object['report'][2],record_path = 'data')
            df[0] = pd.to_datetime(df[0],unit='ms').dt.date
            df.rename(columns={0:"date",1:province_name},inplace=True)
        
        with lock:
            shared_output.shared_df = shared_output.shared_df.merge(
                df,on="date",how='outer')
            shared_count[0] +=1
            end_time=time.perf_counter()
            time_list.append(end_time-start_time)
            return
        
    except Exception as error:
        with lock:
            if len(error.args)==0:
                string = str(error)
            else:
                string = (type(error)).__name__ + ": " + error.args[0]
            error_dict[province_name]=string
            shared_count[0] += 1
            shared_count[1] += 1
            end_time = time.perf_counter()
            time_list.append(end_time-start_time)
            return
        
#------------------------------------------------------------------

def temperature_data_wrapper_func(country,
                                  shared_count,
                                  time_list,
                                  error_dict,
                                  shared_output,
                                  lock,
                                  link):
    """
    The purpose of this function is for wrapping processing average temperature 
    of the world into one function to use for multiprocessing.Pool workers.

    Parameters
    ----------
    country : list
        ['country_code','country_name']
        
    shared_count : multiprocessing.managers.ArrayProxy
        An array that counts [total processed countries,
                              countries that results in error when processing]
        
    time_list : multiprocessing.managers.BaseListProxy
        A shared variable to store time spent processing data.
        
    error_dict : multiprocessing.managers.DictProxy
        Return dict if errors found in workers.
        
    shared_output : multiprocessing.managers.NamespaceProxy
        Use to concatenate DataFrames.
        Have 1 output attribute, ['shared_df']
        
    lock : multiprocessing.managers.AcquirerProxy
        Lock for synchronizing.
        
    link : str
        The website URL.

    Returns
    -------
    None.
        The DataFrame is returned in 'shared_output' instead.
    """
    start_time = time.perf_counter()
    
    try:
        if not isinstance(shared_count,ArrayProxy):
            raise TypeError("'shared_count' type is not correct.")
        if not isinstance(time_list,BaseListProxy):
            raise TypeError("'time_list' type is not correct.")
        if not isinstance(error_dict,DictProxy):
            raise TypeError("'error_dict' type is not correct.")
        if not isinstance(shared_output,NamespaceProxy):
            raise TypeError("'shared_output' type is not correct.")
        if not isinstance(lock,AcquirerProxy):
            raise TypeError("'lock' type is not correct.")
            
        c_code = country[0]
        c_name = country[1]
        api_link = link.format(c_code)
    
        df_temp = pd.read_csv(api_link,skiprows=[0],header=1)
        df_temp.rename(columns={"Unnamed: 0":"Year"},inplace=True)
        df_temp = df_temp.tail(3).copy()
        df_temp.insert(1,'Country',c_name)
        df_temp.insert(1,'iso_code',c_code)
        with lock:
            shared_output.shared_df = pd.concat([shared_output.shared_df,
                                                df_temp],ignore_index=True)
            shared_count[0] +=1
            end_time=time.perf_counter()
            time_list.append(end_time-start_time)
            return
    except Exception as error:
        with lock:
            if len(error.args)==0:
                string = str(error)
            else:
                string = (type(error)).__name__ + ": " + error.args[0]
            error_dict[c_code]=string
            shared_count[0] += 1
            shared_count[1] += 1
            end_time = time.perf_counter()
            time_list.append(end_time-start_time)
            return

#------------------------------------------------------------------

def raw_wrapper_func(code,
                     shared_count,
                     time_list,
                     error_dict,
                     shared_output,
                     lock,
                     input_data,
                     birth_data,
                     death_data,
                     population_data,
                     raw_rolling,
                     trim_percent,
                     death_distribution,
                     death_after,
                     death_rolling,
                     use_origin,
                     recovery_distribution,
                     recovery_after,
                     raw_method_limit=30,
                     raw_seed=None,
                     auto_fill_col=None):
    """
    The purpose of this function is for wrapping the raw data processing 
    into one function to use for multiprocessing.Pool workers.
    
    Parameters
    ----------
    code : str|int|float
        Section code for filtering DataFrame.
        
    shared_count : multiprocessing.managers.ArrayProxy
        An array that counts [total processed sections,
                              sections without population data,
                              sections with empty DataFrame data,
                              sections that results in error when processing]
        
    time_list : multiprocessing.managers.BaseListProxy
        A shared variable to store time spent processing data.
        
    error_dict : multiprocessing.managers.DictProxy
        Return dict if errors found in workers.
        
    shared_output : multiprocessing.managers.NamespaceProxy
        Use to concatenate DataFrames.
        Have 1 output attribute, ['shared_df']
        
    lock : multiprocessing.managers.AcquirerProxy
        Lock for synchronizing.
        
    input_data : tuple|DataFrame
        Tuple format: (DataFrame,search_col)
        Contain all necessary data that is accessed from 'search_col' using
        'code'.
        If input type is 'DataFrame','code' filtering will be ignored.
        
    birth_data : int|float|tuple
        Tuple format: (DataFrame,search_col,result_col)
        Contain birth rate of all sections and access by 'search_col' using 
        'code' or simply input a direct birth rate.
        
        Note: DataFrame must contain birth rate per 1000.If type int|float,
        data must be %/day.
        
    death_data : int|float|tuple
        Tuple format: (DataFrame,search_col,result_col)
        Contain death rate of all sections and access by 'search_col' using 
        'code' or simply input a direct death rate.
        
        Note: DataFrame must contain death rate per 1000.If type int|float,
        data must be %/day.
        
    population_data : int|float|tuple
        Tuple format: (DataFrame,search_col,result_col)
        Contain population of all sections and access by 'search_col' using 
        'code' or simply input a direct population.
        
    raw_rolling : int
        Input for covid_science.prepration.process_raw_infect_data function.
        
    trim_percent : int/float
        Input for covid_science.prepration.process_raw_infect_data function.
        
    death_distribution : list
        Input for covid_science.prepration.process_raw_infect_data function.
        
    death_after : int
        Input for covid_science.prepration.process_raw_infect_data function.
        
    death_rolling : int
        Input for covid_science.prepration.process_raw_infect_data function.
        
    use_origin : boolean
        Input for covid_science.prepration.process_raw_infect_data function.
        
    recovery_distribution : list
        Input for covid_science.prepration.process_raw_infect_data function.
        
    recovery_after : int
        Input for covid_science.prepration.process_raw_infect_data function.
        
    raw_method_limit : int, optional
        Input for covid_science.prepration.process_raw_infect_data function.
        
    raw_seed : int, optional
        Input for covid_science.prepration.process_raw_infect_data function.
        
    auto_fill_col: list, optional
        List of columns that need to auto fill with previous value.
    
    Returns
    -------
    None
        DataFrame is returned in 'shared_output' instead.
    """
    start_time=time.perf_counter()
    
    try:
        if not isinstance(shared_count,ArrayProxy):
            raise TypeError("'shared_count' type is not correct.")
        if not isinstance(time_list,BaseListProxy):
            raise TypeError("'time_list' type is not correct.")
        if not isinstance(error_dict,DictProxy):
            raise TypeError("'error_dict' type is not correct.")
        if not isinstance(shared_output,NamespaceProxy):
            raise TypeError("'shared_output' type is not correct.")
        if not isinstance(lock,AcquirerProxy):
            raise TypeError("'lock' type is not correct.")
        if type(input_data) not in [tuple,pd.DataFrame]:
            raise TypeError("'input_data' type is not correct.")
        if type(birth_data) not in [int,float,tuple,np.float64,np.int64]:
            raise TypeError("'birth_data' type is not correct.")
        if type(death_data) not in [int,float,tuple,np.float64,np.int64]:
            raise TypeError("'death_data' type is not correct.")
        if type(population_data) not in [int,float,tuple,np.float64,np.int64]:
            raise TypeError("'population_data' type is not correct.")
            
            
        #Groups of function input parameters
        
        col = ['new_cases','new_deaths']
        window_trim = [('case_avg',raw_rolling,trim_percent),
                          ('death_avg',raw_rolling,trim_percent)]
        date = ['date',None]
        death_input = ('death_avg','case_avg','origin_case',
                          death_distribution,
                          {'death_after':death_after})
        death_per = ('origin_case',
                     'case_avg',
                     'death_percent',
                     death_rolling,
                     trim_percent)
        recov_input = ('case_avg',
                       'recovery_case',
                       use_origin,
                       recovery_distribution,
                       {'recovery_after':recovery_after})
    
        #birth rate data
        if type(birth_data) in [int,float,np.float64,np.int64]:
            birth_rate = birth_data
        else:
            birth_df = birth_data[0]
            birth_rate = birth_df.loc[birth_df[birth_data[1]]==code,
                                      birth_data[2]]
        birth_rate = np.array(birth_rate)
        
        #death rate data
        if type(death_data) in [int,float,np.float64,np.int64]:
            death_rate = death_data
        else:
            death_df = death_data[0]
            death_rate = death_df.loc[death_df[death_data[1]]==code,
                                      death_data[2]]
        death_rate = np.array(death_rate)
        
        #population data
        if type(population_data) in [int,float,np.float64,np.int64]:
            population = population_data
        else:
            population_df = population_data[0]
            population = population_df.loc[population_df[population_data[1]]
                                           ==code,population_data[2]]
        population = np.array(population)
            
        #if no population data
        if ((birth_rate.size==0 or np.all(np.isnan(birth_rate))) or 
            (death_rate.size==0 or np.all(np.isnan(death_rate))) or 
            (population.size==0 or np.all(np.isnan(population)))):
            with lock:
                shared_count[0] +=1
                shared_count[1] +=1
                end_time = time.perf_counter()
                time_list.append(end_time-start_time)
                return
            
        if type(input_data)==tuple:
            input_df = input_data[0]
            raw_df = (input_df.loc[input_df[input_data[1]]==code]
                      .reset_index(drop=True))
        else:
            raw_df = input_data
                         
        #if dataframe contain no data
        if np.all(np.isnan(
                raw_df[['new_cases','new_deaths','new_vaccinations_smoothed']]
                )):
            with lock:
                shared_count[0] +=1
                shared_count[2] +=1
                end_time = time.perf_counter()
                time_list.append(end_time-start_time)
                return
        
        #convert to %/day
        if type(birth_data) not in  [int,float,np.float64,np.int64]:
            birth_rate = (1+(float(birth_rate)/1000))**(1/365)-1 
        else:
            birth_rate = float(birth_rate)
            
        #convert to %/day
        if type(death_data) not in  [int,float,np.float64,np.int64]:
            death_rate = (1+(float(death_rate)/1000))**(1/365)-1
        else:
            death_rate = float(death_rate)
            
        #living population which affected by the disease
        total_N = (int(population),birth_rate,death_rate,'current_pol_N')
        add_df = process_raw_infect_data(raw_df,col,
                                         window_trim,
                                         date,
                                         death_input,
                                         death_per,
                                         recov_input,
                                         total_N,
                                         method_limit=raw_method_limit,
                                         seed=raw_seed)
        
        #drop ['case_avg','death_avg'] NaN rows
        add_df = add_df.iloc[raw_rolling-1:].reset_index(drop=True).copy() 
    
        add_df['new_vaccinations_smoothed']=(add_df['new_vaccinations_smoothed']
                        .interpolate(method='linear',limit_direction='forward')
                        .fillna(0)
                        .astype(np.int64))
        if auto_fill_col is None:
            pass
        else:
            add_df[auto_fill_col]=add_df[auto_fill_col].ffill().bfill()
            
        with lock:
            shared_output.shared_df = pd.concat([shared_output.shared_df,
                                                 add_df],ignore_index=True)
            shared_count[0] += 1
            end_time = time.perf_counter()
            time_list.append(end_time-start_time)
            return
    except Exception as error:
        with lock:
            if len(error.args)==0:
                string = str(error)
            else:
                string = (type(error)).__name__ + ": " + error.args[0]
            error_dict[code]=string
            shared_count[0] += 1
            shared_count[3] += 1
            end_time = time.perf_counter()
            time_list.append(end_time-start_time)
            return
    
#------------------------------------------------------------------

def vaccine_wrapper_func(code,
                         shared_count,
                         time_list,
                         error_dict,
                         shared_output,
                         lock,
                         input_data,
                         vaccine_info,
                         base_vaccine_percent,
                         target_priority=None,
                         vac_meta_data=None,
                         vac_weight=None,
                         other_vaccine_weight=None,
                         vac_method_limit=30,
                         vac_seed=None
                        ):
    """
    The purpose of this function is for wrapping the vaccine data processing 
    into one function to use for multiprocessing.Pool workers.

    Parameters
    ----------
    code : str|int|float
        Section code for filtering DataFrame.
        
    shared_count : multiprocessing.managers.ArrayProxy
        An array that counts [total processed sections,
                              sections that results in error when processing]
        
    time_list : multiprocessing.managers.BaseListProxy
        A shared variable to store time spent processing data.
        
    error_dict : multiprocessing.managers.DictProxy
        Return dict if errors found in workers.
        
    shared_output : multiprocessing.managers.NamespaceProxy
        Use to add new data to a shared DataFrame.
        Have 1 output attribute, ['shared_df']
        
    lock : multiprocessing.managers.AcquirerProxy
        Lock for synchronizing.
        
    input_data : tuple|DataFrame
        Tuple format: (DataFrame,search_col)
        Contain all necessary data that is accessed from 'search_col' using
        'code'.
        If input type is 'DataFrame','code' filtering will be ignored.
        
    vaccine_info : tuple
        Input for covid_science.prepration.vaccinated_case class.
        
    base_vaccine_percent : 2-D array|list|tuple
        Input for covid_science.prepration.vaccinated_case class.
        If tuple, the input format is: 
            ([[vaccine_ratio1],start_date1,end_date1],...),*end_date can be None
        
    target_priority : 2-D array|list of percentage ratio|tuple, optional
        Input for covid_science.prepration.vaccinated_case class.
        If tuple, the input format is: 
            ([[target_priority1],start_date1,end_date1],...),*end_date can be None
            
    vac_meta_data : pd.DataFrame|tuple, optional
        Contains vaccination meta data of all countries for filtering, or
        choose a single country using tuple input.
        Tuple format: (pd.DataFrame,iso_code)
        
    vac_weight : dict, optional
        Input for covid_science.prepration.vaccinated_case.estimate_vac_percent 
        function.
        
    other_vaccine_weight : tuple, optional
        Input for covid_science.prepration.vaccinated_case.estimate_vac_percent 
        function.
 
    vac_method_limit : int, optional
        Input for covid_science.prepration.vaccinated_case.calculate_vaccine_
        distribution function.
        
    vac_seed : int, optional
        Input for covid_science.prepration.vaccinated_case.calculate_vaccine_
        distribution function.

    Returns
    -------
    None
        Data is added to 'shared_output' DataFrame instead.

    """
    start_time = time.perf_counter()
    
    try:
        if not isinstance(shared_count,ArrayProxy):
            raise TypeError("'shared_count' type is not correct.")
        if not isinstance(time_list,BaseListProxy):
            raise TypeError("'time_list' type is not correct.")
        if not isinstance(error_dict,DictProxy):
            raise TypeError("'error_dict' type is not correct.")
        if not isinstance(shared_output,NamespaceProxy):
            raise TypeError("'shared_output' type is not correct.")
        if not isinstance(lock,AcquirerProxy):
            raise TypeError("'lock' type is not correct.")
        
        if type(input_data)==tuple:
            input_df = input_data[0]
            dummy_df = (input_df.loc[input_df[input_data[1]]==code]
                                .reset_index(drop=True).copy())
        else:
            dummy_df = input_data
        
        if type(base_vaccine_percent) is tuple:
            dummy_percent = np.zeros((dummy_df.shape[0],
                                      len(vaccine_info)))
            date_range = pd.date_range(dummy_df.date.min(),
                                       dummy_df.date.max())
            date_min = date_range[0]
            date_max = date_range[-1]
            for item in base_vaccine_percent:
                try:
                    if item[2] is None:
                        dummy_percent[date_range.get_loc(item[1]):] = item[0]
                    else:
                        dummy_percent[date_range.get_loc(item[1]):
                                      (date_range.get_loc(item[2])+1)] = item[0]
                except:
                    if item[2] is None:
                        fill_start = pd.to_datetime(item[1])
                        if fill_start<date_min:
                            dummy_percent[:]=item[0]
                    else:
                        fill_start = pd.to_datetime(item[1])
                        fill_end = pd.to_datetime(item[2])
                        if fill_start in date_range:
                            dummy_percent[date_range.get_loc(fill_start):] = item[0]
                        elif fill_end in date_range:
                            dummy_percent[:date_range.get_loc(fill_end)+1] = item[0]
                        elif (fill_start<date_min) and (fill_end>date_max):
                            dummy_percent[:]=item[0]
                        
            base_vaccine_percent = dummy_percent
            
        if type(target_priority) is tuple:
            dummy_priority = np.zeros((dummy_df.shape[0],3))
            date_range = pd.date_range(dummy_df.date.min(),
                                       dummy_df.date.max())
            date_min = date_range[0]
            date_max = date_range[-1]
            for item in target_priority:
                try:
                    if item[2] is None:
                        dummy_priority[date_range.get_loc(item[1]):] = item[0]
                    else:
                        dummy_priority[date_range.get_loc(item[1]):
                                      (date_range.get_loc(item[2])+1)] = item[0]
                except:
                    if item[2] is None:
                        fill_start = pd.to_datetime(item[1])
                        if fill_start<date_min:
                            dummy_priority[:]=item[0]
                    else:
                        fill_start = pd.to_datetime(item[1])
                        fill_end = pd.to_datetime(item[2])
                        if fill_start in date_range:
                            dummy_priority[date_range.get_loc(fill_start):] = item[0]
                        elif fill_end in date_range:
                            dummy_priority[:date_range.get_loc(fill_end)+1] = item[0]
                        elif (fill_start<date_min) and (fill_end>date_max):
                            dummy_priority[:]=item[0]
                        
            target_priority = dummy_priority
            
        vac_distribute = vaccinated_case(base_vaccine_percent,
                                         vaccine_info,
                                         dummy_df.shape[0],
                                         target_priority)
        
        if ((vac_weight is not None) and 
            (other_vaccine_weight is not None) and 
             (vac_meta_data is not None)):
            start_date = dummy_df.head(1).date.to_list()[0]
            if type(vac_meta_data)==tuple:
                check_dict = (vac_meta_data[0]
                              .loc[vac_meta_data[0].ISO3==vac_meta_data[1]]
                              .set_index("PRODUCT_NAME")[['START_DATE']]
                              .to_dict()['START_DATE'])
            else:
                check_dict = (vac_meta_data.loc[vac_meta_data.ISO3==code]
                              .set_index("PRODUCT_NAME")[['START_DATE']]
                              .to_dict()['START_DATE'])
            
            vac_distribute.estimate_vac_percent(start_date,
                                                vac_weight,
                                                check_dict,
                                                other_vaccine_weight)
        
        vac_distribute.calculate_vaccine_distribution(dummy_df,
                                            'new_vaccinations_smoothed',
                                            method_limit=vac_method_limit,
                                            seed=vac_seed
                                                     )
        dummy_df['curr_full_vaccinated']=(vac_distribute
                                          .full_immu_matrix.sum(axis=1))
        dummy_df['new_full_vaccinated']=(vac_distribute
                                         .new_full_matrix.sum(axis=1))
        dummy_df['new_boost_req']=(vac_distribute
                                   .new_req_boost_matrix.sum(axis=1))
        with lock:
            shared_output.shared_df = pd.concat([shared_output.shared_df,
                                                 dummy_df],ignore_index=True)
            shared_count[0] +=1
            end_time=time.perf_counter()
            time_list.append(end_time-start_time)
            return
    except Exception as error:
        with lock:
            if len(error.args)==0:
                string = str(error)
            else:
                string = (type(error)).__name__ + ": " + error.args[0]
            error_dict[code]=string
            shared_count[0] += 1
            shared_count[1] += 1
            end_time = time.perf_counter()
            time_list.append(end_time-start_time)
            return
        
#------------------------------------------------------------------
    
def wrapper_SVID_predict(code,
                         shared_count,
                         time_list,
                         error_dict,
                         shared_output,
                         lock,
                         input_data,
                         start_date,
                         birth_data,
                         death_data,
                         p,
                         r_protect_time,
                         n_days,
                         **kwargs):
    """
    The purpose of this function is for wrapping the modeling processing into
    one function to use for multiprocessing.Pool workers.

    Parameters
    ----------
         
    code : str
        Iso code of processing country.
        
    shared_count : multiprocessing.managers.ArrayProxy
        An array that counts [total processed countries,
                              country that results in error when processing]
        
    time_list : multiprocessing.managers.BaseListProxy
        A shared variable to store time spent processing data.
        
    error_dict : multiprocessing.managers.DictProxy
        Return dict if errors found in workers.
        
    shared_output : multiprocessing.managers.NamespaceProxy
        Use to add new data to shared DataFrames.
        Have 2 output attribute, ['census_df','shared_df']
        
    lock : multiprocessing.managers.AcquirerProxy
        Lock for synchronizing.
        
    input_data : tuple
        Tuple format: (DataFrame,search_col)
        Contain all necessary data that is accessed from 'search_col' using
        'code'.
        if 'search_col' doesn't exist, new 'search_col' will be inserted at 
        loc 1.
        
    start_date: pd.Timestamp|str|None
        Desired starting date of Input DataFrame, data won't be return for a 
        specific country if 'start_date' is out of range of DataFrame slice.
        If string, format should be "YYYY-MM-DD".
                
    birth_data : int|float|tuple
        Tuple format: (DataFrame,search_col,result_col)
        Contain birth rate of all sections and access by 'search_col' using 
        'code' or simply input a direct birth rate.
        
        Note: DataFrame must contain birth rate per 1000. If type int|float,
        data must be %/day.
        
    death_data : int|float|tuple
        Tuple format: (DataFrame,search_col,result_col)
        Contain death rate of all sections and access by 'search_col' using 
        'code' or simply input a direct death rate.
        
        Note: DataFrame must contain death rate per 1000.If type int|float,
        data must be %/day.
        
    p : float
        Non-vaccinated percent of recruitment rate.
    
    r_protect_time: int
        Average protected (immunity time) of post-covid patient (in days).
    
    n_days: int
        Number of days used to plot equilibrium figure of initial SVID state
    
    **kwargs : dict
        Keyword arguments for covid_science.modeling.SVID_modeling function 
        except 'start_date','input_df','population_param'.

    Returns
    -------
    DataFrame is returned in 'shared_output' instead.

    """
    start_time = time.perf_counter()
    try:
            
        if not isinstance(shared_count,ArrayProxy):
            raise TypeError("'shared_count' type is not correct.")
        if not isinstance(time_list,BaseListProxy):
            raise TypeError("'time_list' type is not correct.")
        if not isinstance(error_dict,DictProxy):
            raise TypeError("'error_dict' type is not correct.")
        if not isinstance(shared_output,NamespaceProxy):
            raise TypeError("'shared_output' type is not correct.")
        if not isinstance(lock,AcquirerProxy):
            raise TypeError("'lock' type is not correct.")
        if type(input_data) not in [tuple]:
            raise TypeError("'input_data' type is not correct.")
        if type(birth_data) not in [int,float,tuple,np.float64,np.int64]:
            raise TypeError("'birth_data' type is not correct.")
        if type(death_data) not in [int,float,tuple,np.float64,np.int64]:
            raise TypeError("'death_data' type is not correct.")
        if input_data[1] not in input_data[0].columns:
            input_df = input_data[0]
            input_df.insert(1,input_data[1],code)
            model_df = input_df.copy()
        else:
            input_df = input_data[0]
            model_df = (input_df.loc[input_df[input_data[1]]==code]
                        .reset_index(drop=True).copy())
        
        #last data date in the DataFrame
        if start_date is None:
            last_day = pd.Timestamp(model_df['date'].values[-1])
        else:
            last_day = pd.Timestamp(start_date)
        
        #birth rate data
        if type(birth_data) in [int,float,np.float64,np.int64]:
            birth_rate = float(birth_data)
        else:
            birth_df = birth_data[0]
            birth_rate = birth_df.loc[birth_df[birth_data[1]]==code,
                                      birth_data[2]]
            birth_rate = (1+(float(birth_rate)/1000))**(1/365)-1
        
        #death rate data
        if type(death_data) in [int,float,np.float64,np.int64]:
            death_rate = float(death_data)
        else:
            death_df = death_data[0]
            death_rate = death_df.loc[death_df[death_data[1]]==code,
                                      death_data[2]]
            death_rate = (1+(float(death_rate)/1000))**(1/365)-1
        
        kwargs['population_param'] = (death_rate,p)
        with warnings.catch_warnings(record=True):
            model_result = SVID_modeling(start_date=last_day,
                                         input_df=model_df,**kwargs)
        
        initial_SVID = model_df.loc[model_df['date']==last_day,
                                    ['S0','V0','I0','D0']].to_numpy(dtype=np.int64)
        initial_params = model_df.loc[model_df['date']==last_day,
                                      ['beta','beta_v','gamma','theta','alpha',
                                       'alpha0','pi']].to_numpy(dtype=np.float64)
        initial_real = model_df.loc[model_df['date']==last_day,
                                    ["daily_case_non_vaccinated",
                                     "daily_case_vaccinated",
                                     "death_avg","recovery_case",
                                     "new_full_vaccinated",
                                     "new_boost_req"]].to_numpy(dtype=np.float64)
        init_reproduction_rates = model_df.loc[model_df['date']==last_day,
                                    ['daily_R0', 'avg_R0']].to_numpy(dtype=np.float64)
        init_reproduction_rates = init_reproduction_rates.squeeze()
        ##newly added processing
        
        n = len(kwargs['model_number'])
        predict_svid = model_result[0]
        predict_score = model_result[1]
        predict_param = model_result[2]
        predict_days = kwargs['predict_days']
        multioutput = kwargs['multioutput']
        model_list = kwargs['model_number']
        best_model = kwargs['best_model']
        
        day_range = pd.date_range(last_day,periods=predict_days+1)
        if kwargs["best_model"] and type(predict_score)!=str:
            dummy_df = pd.DataFrame({"date":day_range,
                                 input_data[1]:np.full(day_range.size,code),
                                 "method":np.full(day_range.size,np.NaN)
                                     })
            dummy_df["best_method"]=model_list[model_result[3]]
            dummy_df[["S0","V0","I0","D0"]]=np.append(initial_SVID,predict_svid,
                                                      axis=0)
            dummy_df[['beta','beta_v','gamma','theta',
                      'alpha','alpha0','pi']]=np.append(initial_params,predict_param,
                                                        axis=0)
            if multioutput=='raw_values':
                dummy_df['score']= pd.Series(list(np.repeat(predict_score
                                                            .reshape(1,-1),
                                                            predict_days+1,
                                                            axis=0)))
            else:
                dummy_df['score']= pd.Series(np.repeat(predict_score,
                                                       predict_days+1))
        
        else:
            dummy_df = pd.DataFrame()
            for i in range(n):
                buffer_df = pd.DataFrame({"date":day_range,
                                     input_data[1]:np.full(day_range.size,code),
                                     "method":np.full(day_range.size,model_list[i])
                                         })
                buffer_df["best_method"]=np.NaN
                buffer_df[["S0","V0","I0","D0"]]=np.append(initial_SVID,
                                                           predict_svid[i],
                                                          axis=0)
                buffer_df[['beta','beta_v','gamma','theta',
                          'alpha','alpha0','pi']]=np.append(initial_params,
                                                            predict_param[i],
                                                            axis=0)
                if type(predict_score)==str:
                    buffer_df['score']=predict_score
                else:
                    if multioutput=='raw_values':
                        buffer_df['score']= pd.Series(list(
                            np.repeat(predict_score[i].reshape(1,-1),
                                      day_range.size,
                                      axis=0)))
                    else:
                        buffer_df['score']= pd.Series(
                            np.full(day_range.size,predict_score[i]))
                
                dummy_df = pd.concat([dummy_df,buffer_df],ignore_index=True)
                
        if (best_model and type(predict_score)==str) or (not best_model):
            real_forecast_arr = np.ndarray(shape=(0,6))
            for i in range(n):
                real_forecast = convert_to_daily_data(last_day,
                                                      model_df,
                                                      'date',
                                                      ['S0','V0','I0','D0',
                                                       'beta','beta_v','gamma',
                                                       'theta','alpha','alpha0',
                                                       'recovery_case'],
                                                      predict_svid[i],
                                                      predict_param[i][:,:-1],
                                                      r_protect_time)
                real_forecast_arr = np.vstack((real_forecast_arr,initial_real))
                real_forecast_arr = np.vstack((real_forecast_arr,real_forecast))
            dummy_df[["daily_case_non_vaccinated",
                      "daily_case_vaccinated",
                      "death_avg",
                      "recovery_case",
                      "new_full_vaccinated",
                      "new_boost_req"]]=real_forecast_arr
        else:
            real_forecast = convert_to_daily_data(last_day,
                                                  model_df,
                                                  'date',
                                                  ['S0','V0','I0','D0','beta',
                                                   'beta_v','gamma','theta',
                                                   'alpha','alpha0',
                                                   'recovery_case'],
                                                  predict_svid,
                                                  predict_param[:,:-1],
                                                  r_protect_time)
            real_forecast_arr = np.vstack((initial_real,real_forecast))
            dummy_df[["daily_case_non_vaccinated",
                      "daily_case_vaccinated",
                      "death_avg",
                      "recovery_case",
                      "new_full_vaccinated",
                      "new_boost_req"]]=real_forecast_arr
        
        #add equilibrium dataframe
        equil_params = initial_params.squeeze().tolist()
        equil_params.extend([death_rate,p])
        equil_data = odeint(SVID_ODE,initial_SVID.squeeze(),
               np.arange(n_days),
               args=tuple(equil_params))
        equil_data = equil_data[:,:-1] #remove D
        equil_df = pd.DataFrame(equil_data.astype(np.int64),
                                columns=['S','V','I'])
        equil_df.insert(0,'date',pd.date_range(last_day,periods=n_days))
        equil_df.insert(1,input_data[1],code)
        equil_df[['daily_R0','avg_R0']]=init_reproduction_rates
        
        with lock:
            shared_output.shared_df = pd.concat([shared_output.shared_df,
                                                 dummy_df],ignore_index=True)
            shared_output.equil_df = pd.concat([shared_output.equil_df,
                                                 equil_df],ignore_index=True)
            shared_count[0] +=1
            end_time=time.perf_counter()
            time_list.append(end_time-start_time)
            return

    except Exception as error:
        with lock:
            if len(error.args)==0:
                string = str(error)
            else:
                string = (type(error)).__name__ + ": " + error.args[0]
            error_dict[code]=string
            shared_count[0] += 1
            shared_count[1] += 1
            end_time = time.perf_counter()
            time_list.append(end_time-start_time)
            return