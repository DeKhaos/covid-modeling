""" This module provides small functions to help with data preparation."""

import pandas as pd
import numpy as np
import re,datetime,os
import warnings
from covid_science.utility_func import trim_mean,take_element
import covid_science.c_preparation as csp
#------------------------------------------------------------------

def read_stored_data(directory_path,encoding='utf-8',**kwargs):
    """
    This function is used to read latested stored csv file with %YYYY-%MM-%DD 
    format as the file name.
    
    Parameters
    ----------
    directory_path: str
        Path to the file location.
    
    encoding: str
        Encoding to use for UTF when reading/writing.
    
    **kwargs:
        Extra arguments from pandas.read_csv() function
    
    Return
    ----------
    DataFrame
        DataFrame output from csv file.
    """
    assert type(directory_path)==str,"Type should be string."
    try:
        file_list = [
            file for file in os.listdir(directory_path) if (file.endswith('.csv') 
            and re.match(r"[0-9]{1,4}-(1[0-2]|0[1-9])-(0[1-9]|[1-2][0-9]|3[0-1])\.csv",file))
            ]
        datetime_list = [
            datetime.datetime.strptime(re.sub(r".csv$","",file),"%Y-%m-%d").date() 
            for file in os.listdir(directory_path) if (file.endswith('.csv') 
           and re.match(r"[0-9]{1,4}-(1[0-2]|0[1-9])-(0[1-9]|[1-2][0-9]|3[0-1])\.csv",file))
            ]

        if file_list==[] and datetime_list==[]:
            raise FileNotFoundError("No files with correct datetime format were found.")
        elif max(datetime_list)>datetime.date.today():
            read_file_index = datetime_list.index(max(datetime_list))
            raise Warning(f"Latest file is '{file_list[read_file_index]}' and the date is in the future. Please check the data again.")
        else:
            read_file_index = datetime_list.index(max(datetime_list))
            read_file = pd.read_csv(directory_path + "\\" + 
                                    file_list[read_file_index],
                                    encoding=encoding,**kwargs)
            return read_file
    except Exception as error:
        raise error

#------------------------------------------------------------------

class recovery_case:
    def __init__(self,time_distribution,data_len,distribution_unit="week",
                 recovery_after=7):
        """
        This class is built to  to return estimated recovery case after a 
        minimum number of days.
        
        Note
        ---------- 
        Data input should be continuous for data accurancy. It mean missing 
        dates should be added before processing.
        
        Parameters
        ----------
        time_distribution: list
            Percentage distribution of recovered case as time sery after a set 
            amount of time. 
            Distribution will be automatically converted to day distribution.
            Total sum must be 1.0.
            
            Example
            ----------
            recovery_case([0.4,0.6],"week",50,7) #distribution by weeks
            
            Recovery case will be estimated for time sery by day of 50 data point.
            -->40% patients recovered on the 1st week after 7 days since infected.
            
            Each day of 1st week have equal (40%/7) percent for patient to recover.
            -->60% patients recovered on the 2nd week after 7 days since infected. 
            Each day of 2st week have equal (60%/7) percent for patient to recover.
    
        data_len: int
            Length of recovery case time sery data that return after calculation, 
            it should be the same length as input data.
            It should be taken into consideration that data MUST be time series 
            data by date.(e.g. data_len = DataFrame.shape[0] if applied axis=1)
            
        distribution_unit: str
            Time unit of the time_distribution list, can be either of 
            ["day","week","month","year"].
            
            Conversion rate:
            "day"  : no conversion
            "week" : 7 days
            "month": 30 days
            "year" : 365 days
            
        recovery_after: int
            Minimum day required for recovery case.
        """
        assert type(time_distribution)==list,"'time_distribution' type should be list."
        assert any([True for x in time_distribution if x<0])==False,"Percentage should be positive number."
        assert sum(time_distribution)==1,"Total sum must be 1."
        assert all([True  if type(x)==int else False for x in [data_len,recovery_after]])==True,"Type should be integer."
        assert any([True for x in [recovery_after,data_len] if x<0])==False,"Input can't be negative number."
        assert distribution_unit in ["day","week","month","year"],"Not supported time unit."
        
        def convert_distribution_to_date(time_distribution,conversion_rate):
            """Convert percentage distribution from bigger  TU (time unit) to 
            smaller TU. Percentage will be distributed evenly to smaller TU 
            section that related to the relevant bigger TU.

            Example
            ---------- 
            [0.1,0.9] # week distribution -> [0.1/7,0.1/7,0.1/7,0.1/7,0.1/7,0.1/7,
            0.1/7,0.9/7,0.9/7,0.9/7,0.9/7,0.9/7,0.9/7,0.9/7]
            -->10% to recover on the 1st week will be distributed evenly to 7 
            days of that week.
            -->90% to recover on the 2nd week will be distributed evenly to 7 
            days of that week.

            Parameters
            ----------
            conversion_rate: int
                Conversion rate from bigger TU to smaller TU.
                
            Return
            ----------
            np.array
            """

            assert type(conversion_rate)==int,"Type should be integer."

            convert_distribution = np.zeros(len(time_distribution)*conversion_rate)
            for i,TU_percent in enumerate(time_distribution):
                convert_distribution[i*conversion_rate:
                                     (i*conversion_rate)+conversion_rate] +=\
                    TU_percent/conversion_rate
            return convert_distribution
        
        conversion_ratio = {"day":1,"week":7,"month":30,"year":365}
        
        self.time_distribution = convert_distribution_to_date(
            time_distribution,conversion_ratio[distribution_unit])
        self.recovered_case = np.zeros(data_len) #time sery of recovery cases
        self.recovery_after = recovery_after
     
    def estimate_recovery_case(self,input_data,label,origin_death_label=None,
                               data_usage=1,method_limit=30,seed=None):
        """
        Estimating the distribution of recovery case as time sery by date.
        
        Parameters
        ----------
        input_data: pd.DataFrame
            The DataFrame which will be used for data processing.
            
        label: str
            The total infected case column which to apply the function to (take 
            only 1 value).
            
        origin_death_label: str,optional
            The column which contain data of original case of death cases which
            to substract value from 'label' column, normally use when user want
            to get an accurate distribution of recovery case comparing to daily
            infected case and death case originally from that day.
            
        data_usage: float [0,1]/or pd.Series, optional
            Percent of case used from each data point to calculate.
            
            Example
            ----------
            data_usage = 0.9 mean with data point of 500 cases, 90%*500=450 
            cases will be used to estimate the recover cases.
            
        method_limit: int,optional
            The up to limit which np.random.choice will be used instead of fast 
            filling method. The main purpose of this is to keep the randomness 
            to a certain degree if necessary but still keep a fast calculation 
            speed.
            
        seed: int, optional
            Seed for random generation. If 'method_limit'=0, seed won't be 
            applied.
             
        Return
        ----------
        np.ndarray
            Calculate and add estimated recovery cases to self.recovered_case.
        """
        assert type(input_data)==pd.DataFrame,"'input_data' should be DataFrame"
        assert type(label)==str,"'label' type should be string."
        if origin_death_label is not None:
            assert type(origin_death_label)==str,"'origin_death_label' "\
                                                 "type should be string."
        if isinstance(data_usage,(int,float)):
            assert any([data_usage>1,data_usage<0])==False,"'date_usage' is out of range."
            modifier = data_usage
        elif type(data_usage)==pd.Series:
            assert len(data_usage)==len(self.recovered_case),"'data_usage' length doesn't match 'recovery_case.data_len'."
            modifier = data_usage.to_numpy()
            if np.any(pd.isna(modifier)):
                raise ValueError("'data_usage' values can't be NaN.")
            assert np.any([modifier>1,modifier<0])==False,"'date_usage' is out of range."
        else:
            raise TypeError("'data_usage' input type doesn't supported.")
        
        total_day = len(self.time_distribution)
        array_length = len(self.recovered_case)
        
        #round the daily total case first
        values_to_fill = np.round(np.nan_to_num(input_data[label].to_numpy()))
        
        if origin_death_label is not None:
            substract_value = np.round(np.nan_to_num(
                                    input_data[origin_death_label].to_numpy()
                                                    ))
            values_to_fill = values_to_fill - substract_value
        
        
        #check if np.floor is better than np.round
        values_to_fill = np.round(values_to_fill*modifier).astype(np.int64)
        
        #fancy index for buffer array, using to add arranged_case
        idx_for_count = take_element(self.recovery_after + total_day -1,
                                      array_length,
                                      total_day)
        arranged_case = csp.arrange_value_to_array(
                                    values_to_fill,
                                    total_day,
                                    p=self.time_distribution,
                                    method_limit=method_limit,
                                    seed=seed)[0]
        #buffer which serve as a frame for processed data
        buffer_array = np.zeros((array_length,
                                 array_length+total_day+self.recovery_after))
        buffer_array[np.arange(array_length)[:,np.newaxis],
                     idx_for_count]+=arranged_case
        buffer_array = buffer_array.sum(axis=0)
        self.recovered_case = buffer_array[:array_length]
    def reset_recovery_count(self):
        """
        Reset recovery cases count.
        """
        self.recovered_case = np.zeros(len(self.recovered_case))

#------------------------------------------------------------------

class death_case:
    def __init__(self,time_distribution,data_len,distribution_unit="week",death_after=7):
        """
        This class is built to:
        +Trace back estimated origin reported case of death case from the given 
        database.
        +Estimate death case from reported case.
        
        Note
        ---------- 
        Data input should be continuous for data accurancy. It mean missing 
        dates should be added before processing.
        
        Parameters
        ----------
        time_distribution: list
            Percentage distribution of death case as time sery after a set 
            amount of time. 
            Distribution will be automatically converted to day distribution.
            Total sum must be 1.0.
            
            Example
            ----------
            death_case_origin([0.4,0.6],"week",50,7) #distribution by weeks
            
            Origin case will be estimated for time sery by day of 50 data point.
            
            -->40% patients death case will be on the 1st week after 7 days 
            since infected. 
            Each day of 1st week have equal (40%/7) percent for patient to die 
            on that day.
            
            -->60% patients death case will be on the 2nd week after 7 days 
            since infected. 
            Each day of 2st week have equal (60%/7) percent for patient to die 
            on that day.
            
        data_len: int
            Length of origin case time sery data that return after calculation, 
            it should be the same length as input data.
            It should be taken into consideration that data MUST be time series 
            data by date.
            
            Example
            ----------
            data_len = DataFrame.shape[0] if applied axis=1
            
        distribution_unit: str
            Time unit of the time_distribution list, can be either of 
            ["day","week","month","year"].
            
            Conversion rate:
            "day"  : no conversion
            "week" : 7 days
            "month": 30 days
            "year" : 365 days
            
        death_after: int
            Minimum day required for death case to appear.
        """
        assert type(time_distribution)==list,"'time_distribution' type should be list."
        assert any([True for x in time_distribution if x<0])==False,"Percentage should be positive number."
        assert sum(time_distribution)==1,"Total sum must be 1."
        assert all([True  if type(x)==int else False for x in [data_len,death_after]])==True,"Type should be integer."
        assert any([True for x in [death_after,data_len] if x<0])==False,"Input can't be negative number."
        assert distribution_unit in ["day","week","month","year"],"Not supported time unit."
        
        def convert_distribution_to_date(time_distribution,conversion_rate):
            """Convert percentage distribution from bigger  TU (time unit) to 
            smaller TU. Percentage will be distributed evenly to smaller TU 
            section that related to the relevant bigger TU.

            Example: [0.1,0.9] # week distribution -> [0.1/7,0.1/7,0.1/7,0.1/7,
            0.1/7,0.1/7,0.1/7,0.9/7,0.9/7,0.9/7,0.9/7,0.9/7,0.9/7,0.9/7]
            -->10% to die on the 1st week will be distributed evenly to 7 days 
            of that week.
            -->90% to die on the 2nd week will be distributed evenly to 7 days 
            of that week.

            Parameters
            ----------
            conversion_rate: int
                Conversion rate from bigger TU to smaller TU.
            
            Return
            ----------
            np.array
            """

            assert type(conversion_rate)==int,"Type should be integer."

            convert_distribution = np.zeros(len(time_distribution)*conversion_rate)
            for i,TU_percent in enumerate(time_distribution):
                convert_distribution[i*conversion_rate:
                                     (i*conversion_rate)+conversion_rate] +=\
                    TU_percent/conversion_rate
            return convert_distribution
        
        conversion_ratio = {"day":1,"week":7,"month":30,"year":365}
        
        self.time_distribution = convert_distribution_to_date(
            time_distribution,conversion_ratio[distribution_unit])
        self.origin_case = np.zeros(data_len) #time sery of origin case of death case
        self.death_case = np.zeros(data_len) #time sery of death case calculated from infect case time sery
        self.death_after = death_after
    def estimate_death_case(self,input_data,label,data_usage=1,
                                method_limit=30,seed=None):
        """
        Estimating the distribution of death case as time sery by date.
        
        Parameters
        ----------
        input_data: pd.DataFrame
            The DataFrame which will be used for data processing.
            
        label: str
            Column which to apply the function to (take only 1 value).
        
        data_usage: float [0,1]/or pd.Series, optional
            Percent of case used from each data point to calculate.
            
            Example
            ----------
            data_usage = 0.9 mean with data point of 500 cases, 90%*500=450 
            cases will be used to estimate the death cases.
            
        method_limit: int,optional
            The up to limit which np.random.choice will be used instead of fast 
            filling method. The main purpose of this is to keep the randomness 
            to a certain degree if necessary but still keep a fast calculation 
            speed.
            
        seed: int, optional
            Seed for random generation. If 'method_limit'=0, seed won't be 
            applied.
             
        Return
        ----------
        np.ndarray
            Calculate and add estimated death cases to self.death_case.
        """
        assert type(input_data)==pd.DataFrame,"'input_data' should be DataFrame"
        assert type(label)==str,"'label' type should be string."
        if isinstance(data_usage,(int,float)):
            assert any([data_usage>1,data_usage<0])==False,"'date_usage' is out of range."
            modifier = data_usage
        elif type(data_usage)==pd.Series:
            assert len(data_usage)==len(self.death_case),"'data_usage' length doesn't match 'death_case.data_len'."
            modifier = data_usage.to_numpy()
            if np.any(pd.isna(modifier)):
                raise ValueError("'data_usage' values can't be NaN.")
            assert np.any([modifier>1,modifier<0])==False,"'date_usage' is out of range."
        else:
            raise TypeError("'data_usage' input type doesn't supported.")
        
        total_day = len(self.time_distribution)
        array_length = len(self.death_case)
        
        values_to_fill = np.nan_to_num(input_data[label].to_numpy())
        
        #np.ceil is probably better than np.round
        values_to_fill = np.ceil(values_to_fill*modifier).astype(np.int64)
            
        #fancy index for buffer array, using to add arranged_case
        idx_for_count = take_element(self.death_after + total_day -1,
                                      array_length,
                                      total_day)
        arranged_case = csp.arrange_value_to_array(
                                    values_to_fill,
                                    total_day,
                                    p=self.time_distribution,
                                    method_limit=method_limit,
                                    seed=seed )[0]
        #buffer which serve as a frame for processed data
        buffer_array = np.zeros((array_length,
                                 array_length+total_day+self.death_after))
        buffer_array[np.arange(array_length)[:,np.newaxis],
                     idx_for_count]+=arranged_case
        buffer_array = buffer_array.sum(axis=0)
        self.death_case = buffer_array[:array_length]
    def estimate_origin_by_infect(self,input_data,death_label,case_label,
                                  method_limit=30,seed=None):
        """
        Approximately locate the origin recorded case of the death case in time 
        sery data using both infected case report and death case report.
        
        Methodology
        ----------
        Assumpt that all case have a similar death percentage.
        Assumpt that all possible death case of day X will be distributed 
        accordingly to time_distribution (by day) after 'death_after' 
        number of day.
        
        Therefore, we can find a near correct relationship as follows:
            
            day_x_total_death = (N[y-n+1]*p[n]+ N[y-n+2]*p[n-1] + ... +N[y]*p[1])*D
            --> total_case[(y-n+1) to y]= (N[y-n+1]*p[n]+ N[y-n+2]*p[n-1] + ... +N[y]*p[1])
            
            With: 
                N[y]: Total case at day y, y is the last contribution date
                    to death case at x day.
                p[n]: distribution percent of death case at day n-th in 
                    time_distribution.
                D: average death percentage of each case.
                
        From that we can derive the distribution of death case on day x 
        will be from day (y-n+1) to day y with distribution percentage as
        follows:
            Template: [(day y,percent %),...]
            
            [(day y-n+1, N[y-n+1]*p[n]/total_case),
             (day y-n+2, N[y-n+2]*p[n-1]/total_case),
            ...
            (day y, N[y]*p[1])/total_case]
            
            With:
                total_case = total infected case from day y-n+1 to day y.
                    
        Parameters
        ----------
        input_data: pd.DataFrame
            The DataFrame which will be used for data processing.
            
        death_label: str
            Death case data column.
        
        case_label: str
            Infected case data column.
            
        method_limit: int,optional
            The up to limit which np.random.choice will be used instead of fast 
            filling method. The main purpose of this is to keep the randomness 
            to a certain degree if necessary but still keep a fast calculation 
            speed.
            
        seed: int, optional
            Seed for random generation. If 'method_limit'=0, seed won't be 
            applied.
            
        Return
        ----------
        np.array
            Calculate and add estimated origin cases to self.origin_case.
        """
        assert type(input_data)==pd.DataFrame,"'input_data' should be DataFrame."
        assert type(death_label)==str,"'death_label' should be string."
        assert type(case_label)==str,"'case_label' should be string."
        
        total_day = len(self.time_distribution)
        array_length = len(self.origin_case)
        #revert the death case time_distribution
        p = np.flip(self.time_distribution)
        
        #convert NaN row to 0
        death_case = np.nan_to_num(input_data[death_label].to_numpy())
        new_case = np.nan_to_num(input_data[case_label].to_numpy())
        
        #convert datatype, round up the death case
        death_case = np.round(death_case).astype(np.int64)
        new_case = np.round(new_case).astype(np.int64)
        
        
        #buffer array for putting upper_lim to 'arrange_value_to_array' func
        buffer_array = np.round(
                            np.append(np.zeros((self.death_after-1)+total_day),
                                      new_case)
                                )
        #output buffer for self.origin_case after each iteraction
        output_buffer = np.zeros((self.death_after-1)+total_day+array_length)
        
        #ignore warning when divide 0/0
        warnings.simplefilter("ignore", category=RuntimeWarning) 
        left_over_arr = np.array([])
        idx = take_element((total_day-1),array_length,total_day)
        for i in range(array_length):
            lim_array = buffer_array[idx[i]].squeeze()
            origin_case_distribution = lim_array*p/(lim_array*p).sum()
            origin_case_distribution = np.nan_to_num(origin_case_distribution,
                                                     posinf=0.0, neginf=0.0)
            
            filled_arr,left_arr = csp.arrange_value_to_array(
                death_case[i:i+1],
                total_day,
                p=origin_case_distribution,
                upper_lim=lim_array,
                method_limit=method_limit,
                seed=seed)
            left_over_arr = np.append(left_over_arr,left_arr)
            buffer_array[idx[i]]-=filled_arr.squeeze()
            output_buffer[idx[i]]+=filled_arr.squeeze()
        warnings.simplefilter("default", category=RuntimeWarning)
        self.origin_case += output_buffer[(self.death_after-1)+total_day:]
        self.unassigned_death_case = left_over_arr
    def reset_origin_count(self):
        """
        Reset origin cases count and unassigned death case count.
        """
        self.origin_case = np.zeros(len(self.origin_case)) 
        self.unassigned_death_case = np.zeros(len(self.unassigned_death_case))
    def reset_death_count(self):
        """
        Reset death cases count.
        """
        self.death_case = np.zeros(len(self.death_case))
    
#------------------------------------------------------------------   

def convert_distribution_to_small_scale(distribution,conversion_rate):
    """Convert percentage distribution from bigger scale to smaller scale. 
    Percentage will be distributed evenly to smaller elements section that 
    related to the relevant bigger scale.

    Example: [0.1,0.9] #divide by 7-> [0.1/7,0.1/7,0.1/7,0.1/7,0.1/7,0.1/7,
    0.1/7,0.9/7,0.9/7,0.9/7,0.9/7,0.9/7,0.9/7,0.9/7]

    Parameters
    ----------
    conversion_rate: int
        Conversion rate from bigger distribition scale to smaller scale.
    
    Return
    ----------
    np.array
    """
    assert type(distribution)==list,"'distribution' type must be list."
    assert any([True for x in distribution if x<0])==False,"Percentage should be positive number."
    assert sum(distribution)==1,"Total sum must be 1."
    assert type(conversion_rate)==int,"Type should be integer."

    convert_distribution = np.zeros(len(distribution)*conversion_rate)
    for i,percent in enumerate(distribution):
        convert_distribution[i*conversion_rate:
                             (i*conversion_rate)+conversion_rate] +=\
            percent/conversion_rate
    return convert_distribution

#------------------------------------------------------------------

def fill_missing_date(input_df,fill_col,format_date=None):
    """
    Fill in missing date of a chronological time sery data and sort DataFrame 
    ascendingly.
    
    Warning
    ----------
    Function can't handle mutiple indexes.Initial index, other than with numbers,
    will be replaced with number index.
    
    Parameters
    ----------
    input_df: pd.DataFrame
        Input DataFrame must have data layout on axis=0 (row).   
    
    fill_col: str
        Column which contains time sery data. Take 1 column only.
        
    format_date: str
        String format is necessary if date column isn't "datetime" type.
        
    Return
    -------
    DataFrame
        New DataFrame with added rows for missing dates.
    """
    assert isinstance(input_df,pd.DataFrame),"Type should be DataFrame."
    
    col_position = input_df.columns
    if format_date!=None:
        input_df[fill_col] = pd.to_datetime(input_df[fill_col],format=format_date)
    else:
        if input_df[fill_col].dtype == np.dtype('O'):
            raise Exception("Filling column isn't datetime dtype. Please add format_date to parse time.")
    
    input_df.sort_values(by=[fill_col],inplace=True)
    input_df = input_df.set_index(fill_col)
    date_range = pd.date_range(input_df.index.min(),input_df.index.max())
    input_df = input_df.reindex(date_range)
    input_df.reset_index(inplace=True)
    input_df.rename(columns={input_df.columns[0]:fill_col},inplace=True)

    return input_df[col_position]

#------------------------------------------------------------------

class vaccinated_case:
    def __init__(self,vaccine_perc,vaccine_info,data_len,target_priority=None):
        """
        This class is built to estimate the time sery of current fully 
        vaccinated population.
            
        Assumpe that:
            
        -Minimum day require for vaccine to take effect is 1 day.
        -Vaccine is distributed evenly if not specified.
        -Assume no cross vaccination happen, and if it does, the effectiveness
        is considered to be the same as the first injected vaccine.
        -All vaccine types require at least 2 dose to be fully effective.
        
        Parameters
        ----------
        vaccine_perc : 2-D array|list of percent|ratio
            The percentage of each vaccine type available at the time of data 
            point.
            If type is list, the same ratio will be applied for all data point.
            If type is array, length must be the same as 'data_len' and a 2-D 
            array.
            
        vaccine_info : tuple of vaccine information.
            Information for each vaccine type. Please note this parameter can 
            also be used as representative groups for all vaccines with similar
            properties.
            All data in Input format must be filled or error will be raised (6 
            arguments).    
            All time data is in 'day' and take only integer numbers.
            
            Input format:
            ----------
            (vaccine_type_1_info,vaccine_type_2_info,...)
            
            with:
            'vaccine_type_info':['name',
                                 '2nd_dose_interval',
                                 '2nd_dose_immunity_threshold_after',
                                 'boost_dose_immunity__threshold_after',
                                 '2nd_dose_wear_off_time',
                                 'booster_dose_wear_off_time']
            
        data_len : int
            Length of the input data.
            
        target_priority : 2-D array|list of percent|ratio, optional
            There are 3 targets of vaccine usaged (1st_dose,2nd_dose,booster)   
            'target_priority' is the percent|ratio of priority 
            vaccine distribution at a given time to each target type.
            If type is None, the ratio is even for all targets.
            If type is list, the same ratio will be applied for all data point.
            If type is array, length must be the same as 'data_len' and a 2-D 
            array.
        
        Returns
        ----------
        None
            Prepare buffer for class attributes.
        The main attribute of the class will be:
        + vaccinated_case.full_immu_matrix:     Time sery of current fully 
                                                immunity population
        + vaccinated_case.req_2nd_matrix:       Time sery of current 2nd dose 
                                                required
        + vaccinated_case.req_boost_matrix:     Time sery of current booster 
                                                dose required
        + vaccinated_case.new_full_matrix:      Time sery of current new full 
                                                vaccinated 
        + vaccinated_case.new_req_boost_matrix: Time sery of new booster 
                                                dose required
        """
        assert isinstance(vaccine_perc,(list,np.ndarray)),"'vaccine_perc' must be list|np.ndarray."
        assert isinstance(vaccine_info,tuple),"'vaccine_info' type must be tuple."
        assert isinstance(data_len,int),"'data_len' type should be integer."
        assert isinstance(target_priority,(list,np.ndarray)) or (target_priority is None),"'target_priority' must be list|np.ndarray."
        
        if isinstance(vaccine_perc,list):
            if len(vaccine_perc)!=len(vaccine_info):
                raise ValueError("'vaccine_perc' length should match number of vaccine type in 'vaccine_info'.")
        if isinstance(target_priority,list):
            if len(target_priority)!=3:
                raise ValueError("'target_priority' length should match number of vaccine targets (3).")
        if isinstance(vaccine_perc,np.ndarray):
            if vaccine_perc.ndim!=2 or vaccine_perc.shape[1]!=len(vaccine_info):
                raise ValueError("'vaccine_perc' must be a 2-D array and 2nd-D length match number of vaccine type in 'vaccine_info'.")
            if vaccine_perc.shape[0]!=data_len:
                raise ValueError("'vaccine_perc' must be a 2-D array and 1nd-D length match 'data_len'.")
        if isinstance(target_priority,np.ndarray):
            if target_priority.ndim!=2 or target_priority.shape[1]!=3:
                raise ValueError("'target_priority' must be a 2-D array and 2nd-D length=3.")
            if target_priority.shape[0]!=data_len:
                raise ValueError("'target_priority' must be a 2-D array and 1nd-D length match 'data_len'.")  
        
        info_len = len(vaccine_info[0])
        assert info_len==6,"'vaccine_info' element length must consist of 6 arguments."
        for vac_type in vaccine_info:
            if len(vac_type)!=info_len:
                raise ValueError("All element of 'vaccine_info' must have same length.")
            if type(vac_type[0])!=str:
                raise TypeError("Vaccine name must be string.")
            if all(isinstance(i,int) for i in vac_type[1:])==False:
                raise TypeError("Date datas of 'vaccine_info' must be integer.")
        
        self.vaccine_perc = np.array(vaccine_perc,dtype=np.float64)
        # self.vaccine_info = vaccine_info
        self.vaccine_info = np.array(vaccine_info,dtype='O')
        self.data_len = data_len
        self.target_priority = np.array(target_priority)
        n_len = len(vaccine_info)
        #time sery of current fully immunity population
        self.full_immu_matrix = np.zeros((data_len,n_len))  
        self.new_full_matrix = np.zeros((data_len,n_len))
        #time sery of current 2nd dose required
        self.req_2nd_matrix = np.zeros((data_len,n_len))    
        #time sery of current booster dose required
        self.req_boost_matrix = np.zeros((data_len,n_len))  
        self.new_req_boost_matrix = np.zeros((data_len,n_len))
        #time sery of how vaccine distribited
        self.vac_distribute_matrix = np.zeros((data_len,3,n_len)) 
    def estimate_vac_percent(self,start_date,vac_weight,check_dict,other_weight=None):
        """
        Estimate time sery of distribution amount of vaccine base on weight of 
        input vaccine types and vaccine application dates. The function will 
        replace the value of 'vaccine_perc' from init class input.

        Parameters
        ----------
        start_date : str|Timestamp
            Start date of the time sery data, use for comparision with vaccine 
            application date on 'check_dict'.
            
        vac_weight : dict
            List all main vaccine name and their weighted values.
            
            Input format:
            ----------
            {'vaccine_a':(group_a,weight_a),'vaccine_b':(group_b,weight_b)}
            
            *weight (int|float)
            
            *group (int): index of vaccine group (from 0). There are many 
            vaccine groups depending on their research method (inactivated 
            virus, mRNA,etc.). In order to simplify the solution, similar 
            vaccine types will be grouped based on the index of 'group'. The 
            groups will be refered to match 'vaccinated_case.vaccine_info' 
            number of vaccine types.
            
        check_dict : dict
            List all application date of each vaccine types. Date can be string
            or Timestamp.
            
            Input format:
            ----------
            {'vaccine_a':date_a,'vaccine_b':date_a}
            
        other_weight : tuple, optional
            Set the vaccine weight for remaining vaccines in 'check_dict' if 
            not available in 'vac_weight'.
            
            Input format:
            ----------
            (group_list,weight_list)
            
            *group_list: index|list of index
            *weight_list: weight|list of weight

        Returns
        -------
            Modify vaccinated_case.vaccine_perc, will use default 'vaccine_perc'
            if 'check_dict' is empty or any 'check_dict' keys doesn't have
            starting date.
        """
        start_date = pd.to_datetime(start_date)
        assert isinstance(vac_weight,dict),"'vac_weight' should be a dictionary."
        assert isinstance(check_dict,dict),"'check_dict' should be a dictionary."
        assert isinstance(other_weight,tuple) or (other_weight is None),\
            "'other_weight' should be a tuple."
        
        new_vac_per = np.zeros((self.data_len,len(self.vaccine_info)))
        
        #use base percent if there is a key without date
        if check_dict=={} or np.any(pd.isna(list(check_dict.values()))):
            return
        for avail_vac in check_dict.keys():
            if vac_weight.get(avail_vac,other_weight) is not None:
                add_weight = vac_weight.get(avail_vac,other_weight)
                #No. of days between start_date and check_dict start date
                position = (pd.to_datetime(check_dict[avail_vac])-start_date
                            ).days
                if pd.isna(position)==False: #if approved date is valid
                    if position<0:
                        new_vac_per[:,add_weight[0]]+=add_weight[1]
                    elif position>=new_vac_per.shape[0]:
                        pass
                    else:
                        new_vac_per[position:,add_weight[0]]+=add_weight[1]
        self.vaccine_perc=new_vac_per
    def calculate_vaccine_distribution(self,input_data,col_name,
                                           method_limit=30,seed=None):
        """
        Approximately estimate the current fully vaccinated population in time
        sery data.
        
        Parameters
        ----------
        input_data : pd.DataFrame
            The DataFrame which will be used for data processing.
        col_name : str
            Column which to apply the function to (take only 1 value).
            
        method_limit: int,optional
            The up to limit which np.random.choice will be used instead of fast 
            filling method. The main purpose of this is to keep the randomness 
            to a certain degree if necessary but still keep a fast calculation 
            speed.
            
        seed: int, optional
            Seed for random generation. If 'method_limit'=0, seed won't be 
            applied.
            
        Returns
        ----------
        None
            The result will be return to those attributes:
            + vaccinated_case.full_immu_matrix.
            + vaccinated_case.req_2nd_matrix
            + vaccinated_case.req_boost_matrix
            + vaccinated_case.new_full_matrix
            + vaccinated_case.new_req_boost_matrix
        """
        assert type(input_data)==pd.DataFrame,"'input_data' must be DataFrame."
        assert type(col_name)==str,"'col_name' should be string."
        n_len = self.vaccine_info.shape[0]
        
        #estimate the amount of vaccine for each type
        v_sery = np.nan_to_num(input_data[col_name].to_numpy()).astype(np.int64)
        vac_per_type = csp.arrange_value_to_array(
            v_sery,
            n_len,
            p=self.vaccine_perc,
            method_limit=method_limit,
            seed=seed)[0].astype(np.int64)
        #estimate the distribution of vaccine by priority
        if self.target_priority.ndim==0:
            p=np.full(self.data_len,None)
        elif self.target_priority.ndim==1:
            p=np.full((self.data_len,3),self.target_priority,
                      dtype=np.float64)
        else:
            p=self.target_priority
        
        vac_info = self.vaccine_info[:,1:].astype(np.int64)
        
        ##make index matrix for getting data from 'self.vac_distribute_matrix'
        x1 = vac_info[:,1]
        x2 = vac_info[:,2]
        x3 = vac_info[:,1] + vac_info[:,3]
        x4 = vac_info[:,2] + vac_info[:,4]
        x5 = vac_info[:,0]
        x0 = np.zeros(3)
        max_stack = np.int64(np.max([x1,x2,x3,x4,x5,x0]))
        
        #1st-D
        X = np.vstack((x1,x2,x3,x4,x5,x0,x0)).T
        X = np.add.outer(np.arange(self.vac_distribute_matrix.shape[0]),
                         -X) + max_stack
        X = X.astype(np.int64)
        #2nd-D
        y = np.array([1,2,1,2,0,1,2])[np.newaxis,:]
        #3rd-D
        z = np.arange(n_len)[:,np.newaxis]
        
        for idx in range(vac_per_type.shape[0]):
            #NOTE:currently, the limit for 1st shot is np.inf, but in 
            #reality it's the maximum population that hasn't get vaccined
            #but since we have data on number of vaccine used, we assume that
            #Ministry of Health already take that into account when vaccinate
            #the population
            
            #estimate the distribution of vaccine by target priority
            if idx !=0: 
                self.vac_distribute_matrix[idx] = csp.arrange_value_to_array(
                          vac_per_type[idx],3,
                          p=p[idx],
                          upper_lim=np.stack((np.full(n_len,np.inf),
                                             self.req_2nd_matrix[idx-1],
                                             self.req_boost_matrix[idx-1]),
                                             axis=1),
                          method_limit=method_limit,
                          seed=seed)[0].transpose()
            else:
                self.vac_distribute_matrix[idx] = csp.arrange_value_to_array(
                          vac_per_type[idx],3,
                          p=p[idx],
                          upper_lim=np.stack((np.full(n_len,np.inf),
                                             np.zeros(n_len),
                                             np.zeros(n_len)),
                                             axis=1),
                          method_limit=method_limit,
                          seed=seed)[0].transpose()
            
            #buffer array to allow index that is not in 
            #'self.vac_distribute_matrix' by extending it.
            buffer_array = np.vstack((np.zeros((max_stack,
                                           self.vac_distribute_matrix.shape[1],
                                           self.vac_distribute_matrix.shape[2]
                                                )),
                                      self.vac_distribute_matrix))
            v1,v2,v3,v4,v5,v6,v7 = buffer_array[X[idx],y,z].T
            
            if idx !=0:
                #current fulled vaccinated = previous date fulled 
                # + 2nd_dose immu_threshold 
                # + booster immu_threshold 
                # - wear-off immunity (2nd+booster)
                curr_full_vac = (self.full_immu_matrix[idx-1] +v1+v2-v3-v4)
                
                #current 2nd dose requirement = previous 2nd require  
                # + 1st dose that can get 2nd dose now 
                # - dose for 2nd shoot used today.
                curr_2nd_req = (self.req_2nd_matrix[idx-1]+v5-v6)
                
                #current booster requirement = previous booster require
                # + wear off(2nd + booster) 
                # - dose for booster shoot used today.
                curr_boost_req = (self.req_boost_matrix[idx-1]+v3+v4-v7)
            else:
                zero_array = np.zeros(n_len)
                curr_full_vac = (zero_array +v1+v2-v3-v4)
                curr_2nd_req = (zero_array+v5-v6)
                curr_boost_req = (zero_array+v3+v4-v7)
                
            self.full_immu_matrix[idx]= curr_full_vac
            self.req_2nd_matrix[idx]= curr_2nd_req
            self.req_boost_matrix[idx]= curr_boost_req
            self.new_full_matrix[idx]= (v1+v2)
            self.new_req_boost_matrix[idx]= (v3+v4)
    def reset_count(self):
        """
        Reset counting of calculated attributes.
        """
        n_len = len(self.vaccine_info)
        self.full_immu_matrix = np.zeros((self.data_len,n_len))  
        self.new_full_matrix = np.zeros((self.data_len,n_len))
        self.req_2nd_matrix = np.zeros((self.data_len,n_len))    
        self.req_boost_matrix = np.zeros((self.data_len,n_len))  
        self.new_req_boost_matrix = np.zeros((self.data_len,n_len))
        self.vac_distribute_matrix = np.zeros((self.data_len,3,n_len))
        
#------------------------------------------------------------------

def process_raw_infect_data(input_df,
                            col=None,
                            window_trim=None,
                            fill_miss_date=None,
                            death_input=None,
                            death_per=None,
                            recov_input=None,
                            total_N=None,
                            method_limit=30,
                            seed=None):
    """
    This function is used to avoid repeatation of covid data process.
    
    User can choose to remove outlier by rolling mean, add missing date in data
    or predict recovery cases and death origin case, etc. depending on the 
    inputs.
    
    Warning
    ----------
    This function is to be used with for time sery with numeric data only.
    Empty rows of input columns will be filled for calculation according to 
    the following logic:
        -Data is time sery.
        -Linear interpolate will be used to fill missing data for points inside 
        the database for chosen 'col' columns.
        -Outer left-side empty values of time sery will be filled with 0.
        -Outer right-side empty values of time sery will be filled with 
        linear interpolate.
    
    Parameters
    ----------
    input_df : pd.DataFrame
        Input DataFrame for data processing. Remember the function can't process
        horizontal time sery.
        
    col : str|list of str, optional
        Columns that needed to remove noises and outlier.
        
    window_trim : tuple|list of tuple, optional
        Calculted the mean of input columns with chosen rolling 'window', trim
        a percentage of data point for outlier removal.
        
        Input format
        ----------
        
        For single column input:
        (output_col,window,trim_percent)
        
        For mutiple columns input:
        [(output_col1,window1,trim1),(output_col2,window2,trim2),...]
        
        Example
        ----------
        process_raw_infect_data(input_df,col="col_1",window_trim=("avg_7",7,25))
        
        ->meaning the data of "col_1" column will be taken as input for sorted 
        pandas.rolling with window=7 and have 25% of data point remove from 
        each rolling and return the output to "avg_7" column.
        
        Reference
        ----------
        Refer to utility_func.trim_mean function for more information.
        
    fill_miss_date : list of str, optional
        Fill in missing date from minimum date to maximum date in data range of
        'date_col_name' column.
        
        Input format
        ----------
        ['date_col_name',format]
        
        Note
        ----------
        Chosen column must have datetime dtype or datetime format.
        
        Reference
        ----------
        Refer to preparation.fill_missing_date function for more information.
        
    death_input : tuple, optional
        Estimate the origin case of each death case based on input information.
        
        Input format:
        ----------
        ("death_case_column","infect_case_column","output_column",
         time_distribution,{kwargs})
        
        *kwargs: extra argument of preparation.death_case in dictionary except
        'data_len'.       
        
        Reference
        ----------
        Refer to preparation.death_case class for more information.
        
    death_per : int|float|tuple|pd.Series, optional
        If data type is tuple:    
            Calculate the death percentage bases on death case and total case of 
            each day. This gives a more accurate look at the fatality rate of the 
            disease on a set period of time. It's better than just calculate the 
            death percentage by divide the sum of death/sum of infect case.
        If data type is others:
            Death percent is given. Therefore, it is used as parameter.
        
        
        Input format:
        ----------
        tuple: ('origin_death_case_col','case_col','output_col',rolling_window,
                trim_percent)
        pd.Series: Length must be equal to input DataFrame. Please NOTE that
        in case of there are missing date in 'input_df' and 'fill_miss_date'
        is used. Sery length of 'death_per' need to match that of the modfied 
        DataFrame.
         
    recov_input : tuple, optional
        Estimate the recovery case of time sery data based on input information.
        
        Input format:
        ----------
        ('infect_case_column','output_column',use_death_data
         time_distribution,{kwargs})
        
        *use_death_data: take boolean value. It mean to subtract original case
        of death case from 'infect_case_column' data instead of using death 
        percentage.
        *kwargs: extra argument of preparation.recovery_case in dictionary except 
        'data_len'. 
        
        Reference
        ----------
        Refer to preparation.recovery_case class for more information.
    
    total_N : tuple, optional
        Estimate the total population that susceptible for disease calculation.
        
        Input format:
        ----------
        (population_at_t=0,daily_birth_rate,dailty_death_rate,'out_col_name')
        
     method_limit: int,optional
         The up to limit which np.random.choice will be used instead of fast 
         filling method. The main purpose of this is to keep the randomness 
         to a certain degree if necessary but still keep a fast calculation 
         speed.
        
    seed : int, optional
        Seed for random generation.

    Returns
    -------
    DataFrame
        New DataFrame with new added columns based on input parameters.
    """
    
    assert isinstance(input_df,pd.DataFrame),"Type should be DataFrame."
    assert (col is None) or isinstance(col,str) or (all([isinstance(i,str) for i in col])\
        if isinstance(col,list) else False),"Columns input should be string/list of string."
        
    assert (window_trim is None) or (isinstance(window_trim,tuple) and len(window_trim)==3)\
        or (all([(isinstance(i,tuple) and len(i)==3) for i in window_trim])\
            if isinstance(window_trim,list) else False),\
        "'window_trim' input should be tuple/list of tuple and tuple length should be 3: (col_name,window,trim_percent)."
    if isinstance(window_trim,list):
        if isinstance(col,str):
            raise ValueError("'window_trim' and 'col' types don't match for calculation.")
        elif isinstance(col,list) and len(window_trim)!=len(col):
            raise ValueError("'col' and 'window_trim' length doesn't match.")
    elif isinstance(window_trim,tuple) and isinstance(col,list):
        raise ValueError("'window_trim' and 'col' types don't match for calculation.")
        
    assert ((all([isinstance(i,str) for i in fill_miss_date]) or isinstance(fill_miss_date[0],str))\
            if (isinstance(fill_miss_date,list) and len(fill_miss_date)==2) else False)\
        or (fill_miss_date is None),"'fill_miss_date' length should be 2 and list of string: [date_col_name,format]."
    
    assert isinstance(death_input,tuple) or (death_input is None),"'death_input' only take tuple input."
    if death_input is not None:
        if len(death_input)<4 or len(death_input)>5:
            raise ValueError("'death_input' requires 4 arguments and extra"+
                             " keyword arguments in dictionary if necessarry.")
        elif all(type(i)==str for i in death_input[0:3])==False:
            raise ValueError("'death_input' columns input need to be string type.")
            
    assert (death_per is None) or isinstance(death_per,(int,float,tuple,pd.Series)),\
        "'death_per' data type input isn't supported."
    if type(death_per)==tuple:
        if len(death_per)!=5:
            raise ValueError("'death_per' input parameter doesn't meet requirements.")
        elif all(type(i)==str for i in death_per[0:3])==False:
            raise ValueError("'death_per' columns input need to be string type.")
    elif type(death_per)==pd.Series:
        if len(death_per)!=len(input_df) and (fill_miss_date is None):
            raise ValueError("'death_per' input Sery need to match input_df length.")
            
    assert isinstance(recov_input,tuple) or (recov_input is None),\
        "'recov_input' only take tuple input."
    if recov_input is not None:
        if len(recov_input)<4 or len(recov_input)>5:
            raise ValueError("'recov_input' requires 4 arguments and extra"+
                             " keyword arguments in dictionary if necessarry.")
        elif isinstance(recov_input[0],str)==False or isinstance(recov_input[1],str)==False:
            raise ValueError("'recov_input' columns input need to be string type.")
        elif isinstance(recov_input[2],bool)==False:
            raise TypeError("'use_death_data' property of 'recov_input' need to be boolean type.")
        if (death_input is None) and recov_input[2]==True:
            raise ValueError("'death_input' cannot be empty when setting use_death_data=True.")
    assert (total_N is None) or (type(total_N)==tuple and len(total_N)==4),\
        "'total_N' input should be tuple of 4: (N0,birth_rate,death_rate,col_name)"
    if total_N is not None:
        if all(isinstance(i,(int,float)) for i in total_N[0:3])==False:
            raise TypeError("Some item types in 'total_N' are not supported.")
        if type(total_N[-1])!=str:
            raise TypeError("column name input of 'total_N' should be string.")
    
    #add missing date
    if fill_miss_date is None:
        df = input_df.copy()
    else:
        df = fill_missing_date(input_df,*fill_miss_date) #auto create a copy
    #fill empty values of processing columns
    if (col is not None) and (window_trim is not None):
        df[col]=df[col].interpolate(method="linear",
                                    limit_direction='both',
                                    limit_area='inside')
        df[col]=df[col].interpolate(method="spline",
                                    order=1,
                                    limit_direction='forward')
        df[col]=df[col].fillna(0)
        
        #for some case, interpolate will cause negative fill-in values, so for
        #the final solution, all such values will be set to 0.
        for column in col:
            df.loc[df[column]<0,column]=0
        
    #remove outlier
    if isinstance(col,str)  and (window_trim is not None):
        df[window_trim[0]]=trim_mean(df[col],window_trim[1],window_trim[2])
    elif isinstance(col,list) and (window_trim is not None):
        for i in range(len(col)):
            column = col[i]
            df[window_trim[i][0]]=trim_mean(df[column],
                                            window_trim[i][1],
                                            window_trim[i][2])
    #estimate origin death case
    if death_input is not None:
        if len(death_input)==4: #check if there are extra arguments
            death_origin = death_case(death_input[3],df.shape[0])
        else:
            death_origin = death_case(death_input[3],df.shape[0],**death_input[4])
        death_origin.estimate_origin_by_infect(df,death_input[0],death_input[1],
                                               method_limit=method_limit,
                                               seed=seed)
        #add origin death case column
        df[death_input[2]]=death_origin.origin_case
        #modify death case to remove unassigned origin death case
        avg_death = df[death_input[0]].to_numpy(dtype=np.float64)
        avg_death = np.where((avg_death-death_origin.unassigned_death_case)>0,
                              (avg_death-death_origin.unassigned_death_case),
                              0.0)
        df[death_input[0]]=avg_death
    #estimate death percentage when required
    if type(death_per)==tuple:
        warnings.simplefilter("ignore", category=RuntimeWarning) 
        df[death_per[2]]=(trim_mean(df[death_per[0]],
                                    death_per[3],
                                    death_per[4])/
                          trim_mean(np.round(df[death_per[1]]),
                                    death_per[3],
                                    death_per[4]))
        warnings.simplefilter("default", category=RuntimeWarning) 
        
        #interpolate NaN values
        df[death_per[2]]=df[death_per[2]].interpolate(method="linear",
                                                      limit_direction='both')
        #fill NaN with 0 if whole sery is NaN
        df[death_per[2]].fillna(0,inplace=True) 
    #estimate recovery case
    if recov_input is not None:
        if len(recov_input)==4: #check if there are extra arguments
            recov_case = recovery_case(recov_input[3],df.shape[0])
        else:
            recov_case = recovery_case(recov_input[3],df.shape[0],**recov_input[4])
        
        if recov_input[2]:
            recov_case.estimate_recovery_case(df,recov_input[0],
                                              origin_death_label=death_input[2],
                                              data_usage=1,
                                              method_limit=method_limit,
                                              seed=seed)
        else:
            if death_per is None:
                recov_case.estimate_recovery_case(df,recov_input[0],
                                                  data_usage=1,
                                                  method_limit=method_limit,
                                                  seed=seed)
            elif type(death_per)==tuple:
                recov_case.estimate_recovery_case(df,recov_input[0],
                                                  data_usage=1-df[death_per[2]],
                                                  method_limit=method_limit,
                                                  seed=seed)
            else:
                recov_case.estimate_recovery_case(df,recov_input[0],
                                                  data_usage=1-death_per,
                                                  method_limit=method_limit,
                                                  seed=seed)
        #add recovery case column
        df[recov_input[1]]=recov_case.recovered_case
    #estimate the susceptible population
    if total_N is not None:
        array_1 = np.full(df.shape[0],total_N[0])
        array_2 = np.arange(0,df.shape[0])
        N_popul = array_1*(np.exp(total_N[1]-total_N[2])**array_2)
        df[total_N[3]]=N_popul.astype(np.int64)
    return df