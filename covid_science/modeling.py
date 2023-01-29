""" This module provides small functions to help with data modeling."""

import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import warnings,inspect
from functools import partial
from covid_science.utility_func import trim_mean,Time_Sery_array

#------------------------------------------------------------------

def fill_empty_param(input_df,input_col,avg_data=1,weights=None,
                     include_zero=False,
                     suppress_annoucement=True):
    """
    Fill in rows with empty model parameter using mean of nearest non-negative 
    values.
    Totally empty columns will not be filled.
    
    Parameters
    ----------
    input_df : pd.DataFrame
        The DataFrame which needed to fill in empty parameters.
        
    input_col : list of str
        List of columns that needed to fill in empty values.
        
    avg_data : int
        Number of nearest non-negative values.
    
    weights : str|list|int|float
        Weights of nearest non-negative values.
        if string: 
            take only method 'linear_nearest', which mean the weight 
            will decrease linearly from nearest data point to the furthest.
        if int|float:
            the next data point will weight n times smaller than the previous
            starting from the nearest data point.
        if list:
            the weights will match the input, need to have the same length as
            'avg_data'.
            
    include_zero: bool
        Also include nearest zero values if possible.
        
    suppress_annoucement: bool, optional
        Supress the result warning of filling process.
    
    Returns
    -------
    input_df : DataFrame
        Return a copy of original DataFrame with filled in values.
    """
    
    assert isinstance(input_df,pd.DataFrame),"'input_df' should be DataFrame."
    assert isinstance(input_col,list) and all([isinstance(i,str) for i in 
                                              input_col]),\
                                "'input_col' should be a list of column names."
    if type(avg_data)!=int: raise TypeError("'avg_data' type should be integer.")
    if avg_data<1: raise ValueError("'avg_data' should be at least 1.")
    
    if weights is None:
        pass
    elif isinstance(weights,str):
        if weights != 'linear_nearest':
            raise ValueError("Input string method not supported for 'weight'.")
    elif isinstance(weights,list):
        pass
    elif isinstance(weights,(int,float)):
        if weights<1:
            raise ValueError("Multiplier value for 'weights' only take 1 or bigger.")
    else:
        raise TypeError(f"'{type(weights)}' is not supported for 'weights'.")
    
    assert isinstance(include_zero,bool),"'include_zero' type should be boolean."
    
    input_df = input_df.copy()
    
    #convert column values and index to np.ndarray
    work_array = input_df[input_col].to_numpy().transpose()
    work_idx = input_df[input_col].index
    work_idx = np.tile(work_idx,(work_array.shape[0],1))
    
    def vector_loop(input_array,index_array,col,df,avg_data,weights,include_zero):
        """
        Support function for using np.vectorize. Find and replace empty 
        values with mean of nearest non-negative values from relevant columns.
            
        Parameters
        ----------
        input_array : np.ndarray
            Values of column converted to array.
            
        index_array : np.ndarray
            Indexes of column converted to array.
            
        col : str
            Current processing column name.
            
        df : TYPE
            Input DataFrame.
            
        avg_data : int
            Number of nearest non-negative values.
            
        weights : str|list
            Weights of nearest non-negative values.
            
        include_zero: bool
            Also include nearest zero values if possible.

        Returns
        -------
        None
        """
        
        empty_array = input_array[np.isnan(input_array)]
        empty_idx = index_array[np.isnan(input_array)]
        
        if empty_array.size==0:
            if not suppress_annoucement:
                print(f"No empty point in '{col}' column to fill in.")
            return
        
        if include_zero:
            fill_array= input_array[(~np.isnan(input_array))] # ignore all NaN 
            fill_idx = index_array[(~np.isnan(input_array))]
        else:
            fill_array= input_array[(input_array!=0) & (~np.isnan(input_array))] #ignore all NaN and 0 rows  
            fill_idx = index_array[(input_array!=0) & (~np.isnan(input_array))]
        
        if fill_array.size==0:
            if not suppress_annoucement:
                print(f"No data point in '{col}' column to use as base for "
                      +"filling in empty values.")
            return
        
        #distance from NaN idx to all other idx
        distance_array=np.abs(fill_idx-empty_idx[:,np.newaxis]) 
        
        #idx of sorted distance array
        sorted_array=np.argsort(distance_array,axis=1,kind='stable')
        #nearest values array
        avg_value = np.take(fill_array,sorted_array)
        
        #mean of k nearest values
        if weights is None:
            avg_fill_values = avg_value[:,:avg_data].mean(axis=1)
        elif weights =='linear_nearest':
            value_count = avg_value[:,:avg_data].shape[1]
            avg_fill_values = np.average(
                avg_value[:,:avg_data],
                axis=1,weights=np.arange(value_count,0,-1)
                )
        elif isinstance(weights,(int,float)):
            value_count = avg_value[:,:avg_data].shape[1]
            exponent = np.arange(value_count-1,-1,-1)
            avg_fill_values = np.average(
                avg_value[:,:avg_data],
                axis=1,weights=np.power(weights,exponent)
                )
        else:
            avg_fill_values = np.average(
                avg_value[:,:avg_data],axis=1,weights=weights
                )
        df.iloc[empty_idx,df.columns.get_loc(col)]=avg_fill_values
    
    #vectorize fill-in function
    repeat_func = np.vectorize(vector_loop,signature='(i),(j),()->()',
                               excluded=['df','avg_data','weights','include_zero'])
    #return nothing because it modifies the df directly
    repeat_func(work_array,work_idx,input_col,df=input_df,avg_data=avg_data,
                weights=weights,include_zero=include_zero)
    
    return input_df

#------------------------------------------------------------------

def reproduction_number(input_df,input_col,p,mu,n=30,outlier_trim=25):
    """
    Calculate the immediate R0 at the current data point and the average R0 of
    'n' previous data point in the time sery. Since 'n' take previous days into
    calculation. The first data points of the time sery won't have enough data
    point, so backfill will be used for average R0 of these points.

    Parameters
    ----------
    input_df : pd.DataFrame
        The DataFrame which contain information for calculation.
        
    input_col : list of str
        List of columns which contain necessary data for calculation,
        ['beta','beta_v','gamma','theta','alpha','alpha0','pi']
        
    p : float
        Non-vaccinated percent of recruitment rate.
        
    mu : float
        Death rate/time.
        
    n : int, optional
        Number of data point to use for average calculation.
        
    outlier_trim : float|int, optional
        The percent of outlier that will be removed from average calculation.

    Returns
    -------
    np.ndarray
        Array of immedate and average R0, (data points,output_columns)
    """
    assert isinstance(input_df,pd.DataFrame),"'input_df' should be DataFrame."
    assert isinstance(input_col,list) and all([isinstance(i,str) for i in 
                                              input_col]),\
                                "'input_col' should be a list of column names."
    assert len(input_col)==7,"'input_col' length should be 7."
    
    if not (isinstance(p,(float,np.float32,np.float64)) and
        (p>=0.0 and p<=1.0)):
        raise ValueError("'p' should be a float number and in [0,1].")
        
    if not (isinstance(mu,(float,np.float32,np.float64)) and mu>=0.0):
        raise ValueError("'mu' should be a non-negative float number.")
        
    if not (isinstance(n,(int,np.int32,np.int64)) and n>=1):
        raise ValueError("'n' should be a positive integer number.")
    
    #raw R0 at each data point
    beta,beta_v,gamma,theta,alpha,alpha0,pi = input_df[input_col].to_numpy().transpose()
    
    numerator = (alpha*beta_v + alpha0*beta +beta*p*mu + beta_v*mu)
    denominator = ((gamma+mu+theta)*(alpha+alpha0+mu) + mu*beta_v*p)
    raw_R0 = numerator/denominator
    
    #avg_R0 of n days
    avg_df = pd.DataFrame(trim_mean(input_df[input_col],
                                 rolling_window=n,
                                 trim_per=outlier_trim))
    avg_df = avg_df.fillna(method='bfill')
    
    beta,beta_v,gamma,theta,alpha,alpha0,pi = avg_df.to_numpy().transpose()
    
    numerator = (alpha*beta_v + alpha0*beta +beta*p*mu + beta_v*mu)
    denominator = ((gamma+mu+theta)*(alpha+alpha0+mu) +mu*beta_v*p)
    avg_R0 = numerator/denominator
    
    return np.stack((raw_R0,avg_R0),axis=0).transpose()
    
#------------------------------------------------------------------

def data_predict(x,y,method='auto',degree=5,
                 scoring='neg_mean_absolute_percentage_error',
                 greater_better=False):
    """
    Function to fit input 1-features 'x,y' to a set of y=f(x) functions and 
    choose the one with the best score.
    
    If no best fit is found,'self.best_fit'=('none',np.NaN) and won't
    return 'self.predict' attribute.
    
    Parameters
    ----------
    x : array-like
        Independent variable.
        
    y : array-like
        Dependent variable.
        
    method : str, optional
        -'auto': choose the best fit curve.
        Otherwise secific the desired function:
        -'poly_fit'
        -'exponential_base[0,1]'
        -'exponential_base[1,inf]'
        -'sigmoid'.
        
    degree : int, optional
        Highest fit degree for regression line fit.
        
    scoring : str, optional
        Predefined regression scoring method from sklearn.
        
    greater_better : bool, optional
        Define if greater score is better or vice versa.

    Returns
    -------
    'predict_property' object
        Using to predict data.

    """
    
    x = np.array(x)
    y = np.array(y)
    if x.ndim!=1 or y.ndim!=1:
        raise ValueError("Only take single feature inputs for 'x' and 'y'.")
    if x.size != y.size:
        raise ValueError("'x' and 'y' should match in length.")
        
    assert isinstance(method,str),"'method' type should be string."
    assert isinstance(degree,int),"'degree' type should be integer."
    if degree<1:
        raise ValueError("'degree' should be bigger than 0.")
    assert isinstance(scoring,str),"'scoring' type should be string."
    assert isinstance(greater_better,bool),"'greater_better' type should be string."
    
    class curve_function:
        """
        Support class which fits the data to a number of certain equations and 
        return 'self.score','self.parameters','self.predict' attributes. 
    
        If there is no fit, all attributes value will be np.NaN.
        
        For polynomial curve, with method='get_poly_order', best poly fit 
        degree will be returned in 'self.degree'.
        """
        def __init__(self,x,y,name,scoring='r2',**kwargs):
            """
            Return predicted object with relevant attributes.
            
            Parameters
            ----------
            x : np.ndarray
                
            y : np.ndarray
                
            name : str
                Curve model name. Currently take 1 of following method:
                -'get_poly_order'
                -'poly_fit'
                -'exponential_base[0,1]'
                -'exponential_base[1,inf]'
                -'sigmoid'
            
            scoring: str, optional
                Chosen scoring method.
                
            **kwargs : TYPE
                Take 2 keyword arguments:
                -'max_degree' for method 'get_poly_order' 
                -'degree' for method 'poly_fit'

            Returns
            -------
            'curve_function' object
                Fitted object.
            """
            #pre-define fit functions and their wrapper for 'predict' attr
            
            def exp_input(a,b,c): #exponential wrapper
                """
                Fill in wrapper function with parameters.
                """
                def return_f(x):
                    """
                    Exponential function with intercept 'a', slope 'b' and 
                    power base of 'c'
                    """
                    return a + b*np.power(c,x)
                return return_f
            
            def exp_function(x,a,b,c):
                return a + b*np.power(c,x)
            
            def logicstic_input(a,b,c,d): #logistic wrapper
                """
                Fill in wrapper function with parameters.
                """
                def return_f(x):
                    """
                    Sigmoid function with parameter 'a','b','c','d'.
                    """
                    return a + b/(1 + np.power(c,x+d))
                return return_f
            
            def logistic_function(x,a,b,c,d):
                return a + b/(1 + np.power(c,x+d))
            
            #get the best poly degree
            if name=='get_poly_order':
                order_range = np.arange(0,kwargs['max_degree']+1)
                #pipeline is used for 1-feature polynominal then grid search to
                #return the best polynomial order
                pipe_input = [('scale',StandardScaler()), 
                              ('polynomial', 
                               PolynomialFeatures(include_bias=False)
                               ), 
                              ('model',LinearRegression())]
                pipeline = Pipeline(pipe_input)
                grid = GridSearchCV(pipeline,
                                          {'polynomial__degree':order_range},
                                          cv=4,scoring=scoring
                                          )
                best_poly_fit = grid.fit(x.reshape(-1, 1),y)
                self.order = best_poly_fit.best_params_['polynomial__degree']
                self.score = best_poly_fit.score(x.reshape(-1, 1),y)
            #fit data to polyfit for polynomial parameters
            elif name=='poly_fit':
                try:
                    params = np.polyfit(x,y,kwargs['degree'])
                    f = np.poly1d(params)
                    self.score = get_scorer(scoring)._score_func(y,f(x))
                    self.predict = f                   
                    self.params = params
                except:
                    self.score = np.NaN
                    self.predict = np.NaN
                    self.params = np.NaN
            #fit data to exponential,sigmoid function. Main skeleton to add
            #more fit function
            elif name in ['exponential_base[0,1]',
                          'exponential_base[1,inf]',
                          'sigmoid']:
                try:
                    if name=='exponential_base[0,1]':
                        input_f = exp_function
                        bounds = ([-np.inf,-np.inf,0],[np.inf,np.inf,1])
                        predict_f = exp_input
                    elif name=='exponential_base[1,inf]':
                        input_f = exp_function
                        bounds = ([-np.inf,-np.inf,1],[np.inf,np.inf,np.inf])
                        predict_f = exp_input
                    else:
                        input_f = logistic_function
                        bounds = (-np.inf,np.inf)
                        predict_f = logicstic_input
                    curve_param = curve_fit(input_f,x,y,bounds=bounds)[0]
                    
                    self.score = get_scorer(scoring)._score_func(y,
                                                    input_f(x,*curve_param))
                    self.predict = predict_f(*curve_param)
                    self.params = curve_param
                except:
                    self.score = np.NaN
                    self.predict = np.NaN
                    self.params = np.NaN
            else:
                self.score = np.NaN
                self.predict = np.NaN
                self.params = np.NaN
                
    class predict_property:
        """
        Return best fit function based with their attribute on chosen 'method'.
        """
        def __init__(self,fn_arr,score_arr,param_arr,predict_arr,
                     method='auto',greater_is_better=True,
                     **kwargs):
            """
            Base on the input parameters. Best fit will be chosen if possible
            and return the 'predict_property' object with option to predict x 
            from y. 
            
            If no best fit is found,'self.best_fit'=('none',np.NaN) and won't
            return 'self.predict' attribute.

            Parameters
            ----------
            fn_arr : np.ndarray
                Function names.
                
            score_arr : np.ndarray
                Score array.
                
            param_arr : np.ndarray
                Function parameters array.
                
            predict_arr : np.ndarray
                Predict object with corresponding function.
                
            method : str, optional
                Chosen fit function name. If not supported or can't be fitted, 
                most attribute will return np.NaN
                
                Default is 'auto', which will go through all supported fit
                functions and chose the best one if possible.
                
            greater_is_better: bool, optional
                Change comparison depending on the scoring method.
                
            **kwargs :
                Take keyword 'degree' when dealing with polynominal.

            Returns
            -------
            'predict_property' object.
                Using to predict data.

            """
            if np.all(np.isnan(score_arr)): #no fitted function is found
                self.best_fit = ('none',np.NaN)
                self.accuracy_result = {}
            else:
                if method=='auto':
                    self.poly_degree = kwargs['degree']
                    s = f'poly_fit:degree_{self.poly_degree}'
                    fn_arr[fn_arr=='poly_fit']=s
                    
                    if greater_is_better:
                        position = np.nanargmax(score_arr)
                    else:
                        position = np.nanargmin(score_arr)
                    self.best_fit=(fn_arr[position],score_arr[position])
                    self.accuracy_result = dict(zip(fn_arr,score_arr))
                    self.predict=predict_arr[position]
                    self.params = param_arr[position]
                else:
                    position = np.argwhere(fn_arr==method)[0,0]
                    self.predict=predict_arr[position]
                    self.params = param_arr[position]
                    if method=='poly_fit': #handling polynomial case
                        self.poly_degree = kwargs['degree']
                        s = f'poly_fit:degree_{self.poly_degree}'
                        self.accuracy_result = {s:score_arr[position]}
                        self.best_fit = (s,score_arr[position])
                    else: #other cases
                        self.accuracy_result = {fn_arr[position]:score_arr[position]}
                        self.best_fit = (fn_arr[position],score_arr[position])
    warnings.filterwarnings('ignore')
    
    if method=='auto':
        #skeleton for input all methods
        f_name = np.array(['poly_fit',
                  'exponential_base[0,1]',
                  'exponential_base[1,inf]',
                  'sigmoid'])
        #best fit degree
        b_degree = curve_function(x,y,'get_poly_order',scoring=scoring,
                                  max_degree=degree).order
    elif method =='poly_fit':
        f_name=np.array([method])
        b_degree = curve_function(x,y,'get_poly_order',scoring=scoring,
                                  max_degree=degree).order
    else:
        f_name=np.array([method])
    #simple function to apply vectorize to 'curve_function' class
    def repeat_f(method_name,x,y,scoring,**kwargs):
        result = curve_function(x,y,method_name,scoring=scoring,**kwargs)
        return result.score,result.params,result.predict
    
    fit_loop = np.vectorize(repeat_f,
                            signature='()->(),(),()',
                            otypes=['f','O','O'],
                            excluded=['x','y','degree','scoring'])
    
    if method in ['auto','poly_fit']:
        score_arr,param_arr,predict_arr = fit_loop(f_name,x=x,y=y,
                                                   degree=b_degree,
                                                   scoring=scoring)
        warnings.resetwarnings()
        result = predict_property(f_name,score_arr,param_arr,predict_arr,
                                  method=method,degree=b_degree,
                                  greater_is_better=greater_better)
    else:
        score_arr,param_arr,predict_arr = fit_loop(f_name,x=x,y=y,
                                                   scoring=scoring)
        warnings.resetwarnings()
        result = predict_property(f_name,score_arr,param_arr,predict_arr,
                                  method=method,greater_is_better=greater_better)
    
    #result object with 'predict' if chosen 'method' is applicable
    return result
    
#------------------------------------------------------------------

def param_prediction(input_sery,
                     start=34,
                     train_p=28,
                     test_p=7,
                     n_predict=7,
                     method='auto',
                     alpha=0.05,
                     scoring='neg_mean_absolute_percentage_error',
                     greater_better=False,
                     allowed_dif=0.05,
                     sarima_order=None,
                     auto_arima_kwargs=None,
                     arima_kwargs=None,
                     curve_kwargs=None
                     ):
    
    """
    Using a comparision between ARIMA and curve_fit method to choose the best
    fit method to predict the model parameter. This function won't give the 
    best estimation for model parameters because in a sense, the parameters are
    not really time dependence variables. But with small number of predict 
    points, it's still usable.
    
    The best order of ARIMA model will be used from pmdarima.auto_arima to 
    train statsmodels.tsa.arima.model.ARIMA model.
    
    If the 'start' position and 'n_predict' is within the sery. Score between
    prediction and real data can be calculated, otherwise it will return 
    np.NaN.
    
    Parameters
    ----------
    input_sery : pd.Series|np.ndarray
        Model parameter data.
        
    start : int
        Index position in 'input_sery',starting from [0,len(input_sery)]. 
        Previous index of 'input_sery' will be reset.
        
    train_p : int, optional
        The number of training points for in-sample test.
        
    test_p : int, optional
        The number of in-sample testing points.
        
    n_predict : int, optional
        The number of prediction points after fitting (train_p + test_p) number
        of data points.
        
    method : str, optional of ['arima','curve_fit','auto']
        'auto' choose the best fit model based on scoring method, if not 
        specific. Otherwise predict the data based on the chosen method.
        
    alpha : float, optional
        Confident interval of ARIMA method.
        
    scoring : str, optional
        Predefined regression scoring method from sklearn.
        
    greater_better : bool, optional
        Define if greater score is better or vice versa.
        
    allowed_dif : float, optional
        Different scoring distance allowed between 'arima' and 'curve_fit', if
        threshold is passed, ARIMA model will always be chosen.
        
    sarima_order: iterable, optional
        Iterable with element contains [order,seasonal_order] for the ARIMA 
        model orders. If not specified, pmdarima.auto_arima will be used to 
        calculated the best orders.
        
    auto_arima_kwargs : dict, optional
        If not satisfied with the default parameters, manually input the 
        parameters for 'pmdarima.auto_arima'.
        Refer to pmdarima.auto_arima for more information
        
    arima_kwargs : dict, optional
        Extra arguments to put input ARIMA result model.
        Refer to statsmodels.tsa.arima.model.ARIMA for more information.
        
    curve_kwargs : dict, optional
        Extra arguments for curve_fit model.
        Refer to covid_science.modeling.data_predict function.
        
    Returns
    -------
    Tuple.
        ('model_name',predict_points,{scores})
    """
    
    if type(input_sery) not in [pd.Series,np.ndarray]:
        raise TypeError("'input_sery' type should be pd.Series|np.ndarray.")
    
    if not isinstance(start,(int,np.int32,np.int64)):
        raise TypeError("'start' type should be integer.")
        
    elif start>=len(input_sery) or start<0:
        raise ValueError("'start' index is out of range.")
    
    if not isinstance(train_p,int):
        raise TypeError("'train_p' type should be integer.")
    elif train_p<1:
        raise ValueError("'train_p' only take positive value.")
    
    if not isinstance(test_p,int):
        raise TypeError("'test_p' type should be integer.")
    elif test_p<1:
        raise ValueError("'test_p' only take positive value.")
    
    if start+1-test_p-train_p<0: ###not sure if it's correct
        raise ValueError("Training data is out of 'input_sery' range.")
    
    if not isinstance(n_predict,int):
        raise TypeError("'n_predict' type should be integer.")
    elif n_predict<1:
        raise ValueError("'n_predict' only take positive value.")
        
    if method not in ['arima','curve_fit','auto']:
        raise ValueError(f'Method \'{method}\' is not supported.')
    
    if not isinstance(alpha,float):
        raise TypeError("'alpha' type should be float.")
    elif alpha<0:
        raise ValueError("'alpha doesn't take negative value.")
    
    assert isinstance(scoring,str),"'scoring' type should be string."
    assert isinstance(greater_better,bool),"'greater_better' type should be string."
    
    if not isinstance(allowed_dif,float):
        raise TypeError("'allowed_dif' type should be float.")
    elif allowed_dif<0:
        raise ValueError("'allowed_dif doesn't take negative value.")
    
    if sarima_order is not None:
        if isinstance(sarima_order,(list,tuple,np.ndarray)) and len(sarima_order)==2:
            pass
        else:
            raise ValueError("'sarima_order' should be a iterable of 2 elements.")
    
    def replace_negative(array):
        """
        Since parameters can't be negative, this function replace negative 
        value with previous closest non negative value, otherwise will fill in
        with 0.
        """
        if isinstance(array,pd.Series):
            test_array = array.copy()
            test_value = array.values
            if np.any(test_array>=0):
                negative = np.argwhere(test_value<0)
                not_neg = np.argwhere(test_value>=0)
                distance = np.abs(not_neg.reshape(1,-1) - negative)
                closest = np.argmin(distance,axis=1)
                p_position = np.take(not_neg,closest)
                test_array[test_array<0] = test_value[p_position]
            else:
                test_array[test_array<0]=0
            return test_array
        else:
            test_array = np.array(array)
            if np.any(test_array>=0):
                negative = np.argwhere(test_array<0)
                not_neg = np.argwhere(test_array>=0)
                distance = np.abs(not_neg.reshape(1,-1) - negative)
                closest = np.argmin(distance,axis=1)
                p_position = np.take(not_neg,closest)
                test_array[test_array<0] = test_array[p_position]
            else:
                test_array = np.where(test_array<0,0,test_array)
            return test_array
        
    #start: starting index in the array [0,n]
    if type(input_sery)==pd.Series:
        input_sery = input_sery.reset_index(drop=True)
    else:
        input_sery = pd.Series(input_sery)

    train = input_sery.iloc[(start+1-test_p-train_p):(start+1-test_p)]
    test = input_sery.iloc[(start+1-test_p):start+1]
    data_slice = input_sery.iloc[(start+1-test_p-train_p):start+1]
    max_slice_idx = data_slice.index[-1]
    warnings.filterwarnings('ignore')
    
    do_arima = True
    do_curve_fit = True
    if method=='arima':
        do_curve_fit = False
    elif method=='curve_fit':
        do_arima = False
        
    #arima fit  
    if do_arima:
        if sarima_order is None:
            if auto_arima_kwargs is None:
                param_find = pm.auto_arima(data_slice, start_p=0, start_q=0,          
                                seasonal=False,   # No Seasonality
                                start_P=0,
                                start_Q=0,
                                error_action='ignore',
                                #best sample acc test
                                information_criterion='oob', 
                                out_of_sample_size=test_p,
                                #set enforce=False for faster parameters found
                                sarimax_kwargs={'enforce_stationarity':False,
                                                'enforce_invertibility':False})
            else:
                param_find = pm.auto_arima(data_slice,start_p=0, start_q=0,
                                seasonal=False,   # No Seasonality
                                start_P=0,
                                start_Q=0,
                                error_action='ignore',
                                out_of_sample_size=test_p,
                                sarimax_kwargs={'enforce_stationarity':False,
                                                'enforce_invertibility':False},
                                **auto_arima_kwargs)
            order = param_find.get_params()['order']
            season_order = param_find.get_params()['seasonal_order']
        #if there is a prechoice arima order
        else: 
            order = sarima_order[0]
            season_order = sarima_order[1]
            
        #use best fit order from auto_arima|prechoice to input to ARIMA
        if arima_kwargs is None:
            arima_model = ARIMA(train,order=order,seasonal_order=season_order)
        else:
            arima_model = ARIMA(train,order=order,seasonal_order=season_order,
                                **arima_kwargs)
        fitted=arima_model.fit(method_kwargs={"warn_convergence": False})
        #parameter prediction from ARIMA model
        arima_predict = replace_negative(fitted.forecast(test.shape[0]))
        score_arima = get_scorer(scoring)._score_func(test,arima_predict)
    #best curve fit
    if do_curve_fit:
        if curve_kwargs is None:
            curve_model = data_predict(train.index,train)
        else:
            curve_model = data_predict(train.index,train,**curve_kwargs)
        if hasattr(curve_model,'predict'):
            curve_predict = replace_negative(curve_model.predict(test.index))
            score_curve = get_scorer(scoring)._score_func(test,curve_predict)
        else:
            score_curve = np.NaN
    
    #return corresponding model base on 'method'
    if method=='auto':
        if np.isnan(score_curve):
            result_model =  arima_model.clone(data_slice).fit(
                method_kwargs={"warn_convergence": False})
        else:
            if (greater_better and (score_curve-score_arima<=allowed_dif)) or\
            ((not greater_better) and (score_arima-score_curve<=allowed_dif)):
                result_model =  arima_model.clone(data_slice).fit(
                    method_kwargs={"warn_convergence": False})
            else:
                result_model = data_predict(data_slice.index,data_slice)
    elif method =='arima':
        result_model =  arima_model.clone(data_slice).fit(
            method_kwargs={"warn_convergence": False})
    else:
        if np.isnan(score_curve): # if no best curve_fit found for method
            result_model = np.NaN
        else:
            if curve_kwargs is None:
                result_model = data_predict(data_slice.index,data_slice)
            else:
                result_model = data_predict(data_slice.index,data_slice,
                                            **curve_kwargs)
    
    #return result object
    #if no best fit
    if isinstance(result_model,float) and np.isnan(result_model): 
        result = ('none',np.NaN,{})
    elif hasattr(result_model,'get_forecast'): #if ARIMA model
        predicted = result_model.get_forecast(n_predict)
        predict_array,up_conf,down_conf = predicted.summary_frame(alpha=alpha)[['mean','mean_ci_lower','mean_ci_upper']].T.to_numpy()
        
        predict_array =replace_negative(predict_array)
        up_conf=replace_negative(predict_array)
        down_conf=replace_negative(predict_array)
        
        expected_score = get_scorer(scoring).\
            _score_func(data_slice,
                        replace_negative(
                        result_model.predict(start=0,end=data_slice.shape[0]-1)
                                        )
                        )
        
        #check for real score if all predict points are in the sery
        if start + n_predict < input_sery.shape[0]:
            real_score = get_scorer(scoring).\
                _score_func(input_sery.iloc[max_slice_idx+1:
                            max_slice_idx+1+n_predict],
                            predict_array)
        else:
            real_score = np.NaN
            
        result = ('arima',
                  predict_array,
                  {'insample_test_score':score_arima,
                   'data_slice_score':expected_score,
                   'real_score':real_score,
                   'up_conf_predict':up_conf,
                   'down_conf_predict':down_conf}
                  )
    else: #if curve_fit model
        predict_array =  result_model.predict(np.arange(max_slice_idx+1,
                                                        max_slice_idx+1+
                                                        n_predict))
        predict_array = replace_negative(predict_array)
        
        expected_score = get_scorer(scoring)._score_func(data_slice,
                        replace_negative(result_model.predict(data_slice.index)
                                         )
                                                        )
        #check for real score if all predict points are in the sery
        if start + n_predict < input_sery.shape[0]:
            real_score = get_scorer(scoring)._score_func(input_sery.iloc[
                                max_slice_idx+1:max_slice_idx+1+n_predict],
                                                        predict_array)
        else:
            real_score = np.NaN
            
        result = ('best curve_fit: ' + result_model.best_fit[0],
                  predict_array,
                  {'insample_test_score':score_curve,
                   'data_slice_score':expected_score,
                   'real_score':real_score,
                   'method_comparision':result_model.accuracy_result}
                  )
    warnings.resetwarnings()
    return result
        
#------------------------------------------------------------------

def get_arima_order(input_df,
                    cols,
                    start=34,
                    n_slice=35,
                    test_p=7,
                    **kwargs):
    """
    Estimate the ARIMA model orders and seasonal orders from the input 
    DataFrame.

    Parameters
    ----------
    input_df : pd.DataFrame
        The input DataFrame.
        
    cols : list
        List of DataFrame columns to process.
        
    start : int, optional
        Index position in 'input_df',starting from [0,len(input_df)]. 
        Previous index of 'input_df' will be reset.
        
    n_slice : int, optional
        The number of data point to put to the ARIMA model.
        
    test_p : int, optional
        The test size for ARIMA model if information_criterion='oob' is used.
        
    **kwargs : optional
        If not satisfied with the default parameters, manually input the 
        parameters for 'pmdarima.auto_arima'.
        Refer to pmdarima.auto_arima for more information.

    Returns
    -------
    List
        Return (orders,seasonal orders) of each processed columns.
    """
      
    if type(input_df)!= pd.DataFrame:
        raise TypeError("'input_df' type should be pd.DataFrame.")
    
    if not isinstance(start,(int,np.int32,np.int64)):
        raise TypeError("'start' type should be integer.")
        
    elif start>=input_df.shape[0] or start<0:
        raise ValueError("'start' index is out of range.")
    
    if not isinstance(n_slice,int):
        raise TypeError("'n_slice' type should be integer.")
    elif n_slice<1:
        raise ValueError("'n_slice' only take positive value.")
    
    if not isinstance(test_p,int):
        raise TypeError("'test_p' type should be integer.")
    elif test_p<1:
        raise ValueError("'test_p' only take positive value.")
    
    if start+1-n_slice<0:
        raise ValueError("Training data is out of 'input_df' range.")
    
    #reset previous index and starting from 0
    input_df = input_df.reset_index(drop=True)
    
    data_slice = input_df[cols].iloc[(start+1-n_slice):start+1]
    if kwargs == {}:
        partial_func = partial(pm.auto_arima,
                              start_p=0, 
                              start_q=0,            
                              seasonal=False,   # No Seasonality
                              start_P=0,
                              start_Q=0,
                              error_action='ignore',
                              information_criterion='oob',#best sample acc test
                              out_of_sample_size=test_p,
                              #set enforce=False for faster parameters found
                              sarimax_kwargs={'enforce_stationarity':False,
                                              'enforce_invertibility':False})
    else:
        # partial_func = partial(pm.auto_arima,**kwargs)
        partial_func = partial(pm.auto_arima,
                              start_p=0, 
                              start_q=0,            
                              seasonal=False,   # No Seasonality
                              start_P=0,
                              start_Q=0,
                              error_action='ignore',
                              out_of_sample_size=test_p,
                              #set enforce=False for faster parameters found
                              sarimax_kwargs={'enforce_stationarity':False,
                                              'enforce_invertibility':False},
                              **kwargs)
    vector_func = np.vectorize(partial_func,signature="(i)->()")
    vector_result = vector_func(data_slice.to_numpy().T)
    
    order_list = []
    season_list =[]
    
    for arima in vector_result:
        order_list.append(arima.get_params()['order'])
        season_list.append(arima.get_params()['seasonal_order'])
    
    return list(zip(order_list,season_list))
#------------------------------------------------------------------
    
def SVID_modeling(start_date,
                  predict_days,
                  input_df,
                  date_col,
                  initial_state,
                  model_param,
                  population_param,
                  model_number = [1,2,3],
                  best_model=False,
                  scoring='neg_mean_absolute_percentage_error',
                  multioutput="raw_values",
                  greater_better=False,
                  preset_arima_order=None,
                  param_kwargs=None):
    
    """
    Predict the future SVID based on prediction of future parameter and initial
    SVID. 
    
    There are 2 cases where the accuracy score will be calculated:
    -The start_date and the predicted days are all within the input_df.
    -If some or all of predicted days are out of data range. Then the score 
    will be estimated by using another closest 'start_date' with all its 
    predicted days are within data range.
    
    Parameters
    ----------
    start_date : str|pd.Timestamp
        Starting date of modeling, should be a valid day within 'input_df'.
        
    predict_days : int
        Number of forecast day.Minium is 1. It is recommended to predict too
        far since the result won't be correct.
        
    input_df : pd.DataFrame
        The DataFrame which contain all data for 'initial_state' and 
        'model_param'.
        
    date_col : str
        DataFrame column which contains date data.
        
    initial_state : list of str|list of int
        Starting state of the model.
        If all str, corresponding columns in 'input_df' will be used.
        If all int, it will be used as initial state for SVID. This is usefull
        when one need to guess what would happen if the current starting point
        is different from the current
        
    model_param : list of str
        List of DataFrame columns which contain parameters with order,['beta',
        'beta_v',gamma','theta','alpha','alpha0','pi']
        
    population_param : tuple
        A tuple of (death_rate/day,birth non_vaccinated fraction).
        
    model_number : list of int, optional
        A set of model to choose from, with at least of 1 model must be chosen.
        1: delta method, in which X_next_day = X_previous_day + f()
        2: ODEs which use all forecasted 'model_param' to forecast SVID
        3: ODEs which use only 'model_param' of 'start_date' to forecast SVID
        
    best_model : bool, optional
        If True, choose the best performance model.
        
    scoring : str, optional
        Predefined regression scoring method from sklearn.
        
    multioutput : str, optional
        Take one of three methods,('raw_values','variance_weighted',
                                   'uniform_average')
        By defaults, raw scores for each S,V,I,D will be calculated and return.
        
    greater_better : Bool, optional
        Define if greater score is better or vice versa.
        
    preset_arima_order:list, optional
        This parameter is used to reduce the calculation time with preset ARIMA
        orders applied to 'model_param' parameters.
        
        'preset_arima_order' is a list with each element as (order,
        seasonal_order) of ARIMA model. Length must match the number of 
        'model_param' columns used.
        
        Note: If 'preset_arima_order' is used. The estimated score if any will
        be different since 'preset_arima_order' will also be applied for ARIMA
        models.
        
    param_kwargs : dict, optional
        Extra keywords to pass to param_prediction function,
        except {'start','n_predict','sarima_order'}
        Refer to covid_science.modeling.param_prediction for more information.

    Returns
    -------
    Tuple.
        If best_model==True:    
            (predict_array,predict_score,predict_params,position)
            #with 'position' is the best model index in chosen 'model_number'
        Else:
            (predict_array,predict_score,predict_params)
            
        
        NOTE: In the return tuple, 'predict_params' will always have dtype
        object (due to case of return np.NaN if empty). 
        Therefore to avoid error when using, dtype convert should be 
        preprocessed.
    """
    assert isinstance(start_date,(str,pd.Timestamp)),"'start_date' should be "\
                                                    "string or pd.Timestamp."
    assert isinstance(predict_days,int) and predict_days>0,"Number of"\
                                        " 'predict_days' should be int and >0."
    assert isinstance(input_df,pd.DataFrame),"'input_df' should be DataFrame."
    assert isinstance(date_col,str),"'date_col' should be string."
    assert (isinstance(initial_state,list) and len(initial_state)==4) and\
        (all([isinstance(i,str) for i in initial_state]) or 
         all([isinstance(i,int) for i in initial_state])),\
                               "'intial_state' should be a list of 4 elements."
    assert (isinstance(model_param,list) and len(model_param)==7) and \
    all([isinstance(i,str) for i in model_param]),\
    "'model_param' should be a list of 7 elements."
    assert isinstance(population_param,tuple) and len(population_param)==2,\
        "'population_param' should be a tuple of 2 values."
    assert type(model_number)==list,"'model_number' should be in a list."
    assert len(model_number)>=1,"Must use at least 1 'model_number'."
    if not all([i in [1,2,3] for i in model_number]):
        raise ValueError("Input 'model_number' incorrect.")
    if len(model_number) != len(set(model_number)):
        raise ValueError("model_number' don't take duplications.")
    
    if model_number != sorted(model_number):
        raise ValueError("Please sort 'model_number' ascendingly.")
    
    assert isinstance(best_model,bool),"'best_model' type should be string."
    assert isinstance(scoring,str),"'scoring' type should be string."
    assert isinstance(multioutput,str),"'multioutput' type should be string."
    if multioutput not in ['raw_values','variance_weighted','uniform_average']:
        raise ValueError(f"'multioutput':{multioutput} not supported.")
    assert isinstance(greater_better,bool),"'greater_better' type should be string."
    
    if preset_arima_order is not None:
        if (type(preset_arima_order)!=list or 
            len(preset_arima_order)!=len(model_param)):
            raise ValueError("'preset_arima_order' must be a list and match "\
                             "'model_param' length.")
    
    if param_kwargs is not None:
        if any(i in param_kwargs.keys() for i in ['start','n_predict','sarima_order']):
            raise ValueError("'param_kwargs' can't have keywords in ['start',"\
                             "'n_predict','sarima_order'].")
    
    def delta_method_predict(initial_values,parameter,mu,p,return_array=None):
        """
        This method estimate the next value base on the the difference distance
        between 2 adjacent data points using some parameter as the prediction
        for the difference.

        Parameters
        ----------
        initial_values : np.ndarray
            Starting values.
            
        parameter : np.ndarray
            Matrix of forecast parameters with dimension (n_datapont,n_param).
            
        mu : float
            Death rate per day.
            
        p : float
            Non-vaccinated birth percent.
            
        return_array : np.ndarray, optional
            Ignore, this is for recursion purpose.

        Returns
        -------
        np.ndarray
            Forecast values with shape (n_datapoint,n_attribute).
        """
        
        #initial state of model
        s0,v0,i0,d0=initial_values
        #total disease-induced population
        n0 = s0+ v0+i0+d0
        #parameter of guessing points
        beta,beta_v,gamma,theta,alpha,alpha0,pi = parameter[0,:]
        
        #prediction model
        s = s0-beta*s0*i0/n0 -(alpha+mu)*s0 + alpha0*v0 + pi*p
        v = v0-beta_v*v0*i0/n0 -(alpha0+mu)*v0 + theta*i0 + alpha*s0 + pi*(1-p)
        i = i0-(mu+gamma+theta)*i0 + (beta*s0+beta_v*v0)*i0/n0
        d = d0+gamma*i0 -mu*d0
        
        new_init = np.array([s,v,i,d])
        new_parameter = parameter[1:]
        
        if return_array is None:
            return_array = np.vstack((initial_values,[s,v,i,d]))
        else:
            return_array = np.vstack((return_array,[s,v,i,d]))
            
        if new_parameter.size>0: #use recursion untill no parameter available
            return delta_method_predict(new_init,new_parameter,mu,p,
                                        return_array)
        else:
            return return_array 
        
    def SVID_ODE(initial,t,beta,beta_v,gamma,theta,alpha,alpha0,pi,mu,p):
        """
        ODE system to calculate the SVID predictions, used to apply to 
        scipy.integrate.odeint.

        Parameters
        ----------
        initial : np.ndarray
            Starting values.
            
        t : np.ndarray
            Independent variable.
            
        beta,beta_v,gamma,theta,alpha,alpha0,pi,mu,p: int|float
            Model parameters.

        Returns
        -------
        list
            Model ODEs output.
        """
        
        s,v,i,d = initial
        n = s+v+i+d
        dsdt = -beta*s*i/n -(alpha+mu)*s + alpha0*v + pi*p
        dvdt = -beta_v*v*i/n -(alpha0+mu)*v + theta*i + alpha*s + pi*(1-p)
        didt = -(mu+gamma+theta)*i + (beta*s+beta_v*v)*i/n
        dddt = gamma*i -mu*d
        return [dsdt,dvdt,didt,dddt]
    
    #start_date: str or pd.Timestamp object
    
    frame = inspect.currentframe()
    
    ####input data
    
    #start index
    start = input_df.loc[input_df[date_col]==start_date].index[0]
    #intial_state S V I D (columns)
    if all([isinstance(i,str) for i in initial_state]):
        initial_SVID = input_df.loc[start,initial_state].to_numpy(dtype=np.int64)
    else: #use intial input number as intial state
        initial_SVID = np.array(initial_state)
        
    #model param: 'date','beta','beta_v','gamma','theta','alpha','alpha0'
    param_SVID = input_df[model_param]
    initial_parameter = input_df.loc[start,model_param
                                     ].to_numpy(dtype=np.float64)
    
    #test array if avaible
    if all([isinstance(i,int) for i in initial_state]):
        test = None
    else:
        if start + predict_days<input_df.shape[0]:
            test = input_df.loc[start+1:start+predict_days,initial_state
                                ].to_numpy()
        else:
            test = None
    
    #predict future day parameters
    
    if (1 in model_number) or (2 in model_number):
        if preset_arima_order is None:
            signature = "(i)->(),(k),()"
            excluded = ['start','n_predict','sarima_order']
            input_order = None
        else:
            signature = "(i),(j)->(),(k),()"
            excluded = ['start','n_predict']
            input_order = np.array(preset_arima_order,dtype='O')
        if param_kwargs is None:
            param_func = np.vectorize(param_prediction,
                                      signature=signature,
                                      excluded = excluded)
            param_predict = param_func(param_SVID.to_numpy().T,
                                       start=start,
                                       n_predict=predict_days,
                                       sarima_order=input_order)[1].T
            param_predict = np.vstack((initial_parameter.reshape(1,-1),
                                        param_predict))[:-1]
        else:
            excluded.extend(list(param_kwargs.keys()))
            param_func = np.vectorize(param_prediction,
                                      signature=signature,
                                      excluded=excluded)
            param_predict = param_func(param_SVID.to_numpy().T,
                                       start=start,
                                       n_predict=predict_days,
                                       sarima_order=input_order,
                                       **param_kwargs
                                        )[1].T
            param_predict = np.vstack((initial_parameter.reshape(1,-1),
                                        param_predict))[:-1]
    #death rate
    mu = population_param[0] 
    #birth_vaccinated percent
    p = population_param[1]  
    
    result_array,result_score,result_params = [],[],[]
    
    ###TEST 1,using denta for all forecasted parameters with intial SVID
    
    if 1 in model_number:
        result_t1 = delta_method_predict(initial_SVID,param_predict,mu,p
                                         )[1:]
        #np.ceil in order to reduction compare to initial state SVID if 
        #round down for forecast points.
        result_t1 = np.ceil(result_t1)
        #make sure D at least not decrease
        result_t1[:,3] = np.where(result_t1[:,3]<initial_SVID[3],
                                  initial_SVID[3],result_t1[:,3])
        result_t1 = result_t1.astype(np.int64)
        if test is not None:
            score_t1 = get_scorer(scoring)._score_func(test,result_t1,
                                                       multioutput='raw_values'
                                                       )
        else:
            score_t1 = np.full(4,np.NaN)
        result_array.append(result_t1)
        result_score.append(score_t1)
        result_params.append(param_predict)
          
    ###TEST 2,use all predicted parameter for ODEs
    
    if 2 in model_number:
        initial_value = initial_SVID
        result_t2 = initial_SVID
        for params in param_predict:
            x1,x2,x3,x4,x5,x6,x7=params
            sol = odeint(SVID_ODE,initial_value,[0,1],
                         args=(x1,x2,x3,x4,x5,x6,x7,mu,p))
            initial_value = sol[1].flatten().tolist()
            result_t2 = np.vstack((result_t2,initial_value))

        result_t2 = np.ceil(result_t2[1:])
        result_t2[:,3] = np.where(result_t2[:,3]<initial_SVID[3],
                                  initial_SVID[3],result_t2[:,3])
        result_t2 = result_t2.astype(np.int64)
        
        if test is not None:
            score_t2 = get_scorer(scoring)._score_func(test,result_t2,
                                                       multioutput='raw_values'
                                                       )
        else:
            score_t2 = np.full(4,np.NaN)
            
        result_array.append(result_t2)
        result_score.append(score_t2)
        result_params.append(param_predict)

    ###TEST 3,use only initial day parameters for ODEs
    if 3 in model_number:
        t = np.arange(0,predict_days+1)
        sol = odeint(SVID_ODE,initial_SVID,t,args = (*initial_parameter,mu,p))
        
        result_t3 = np.ceil(sol[1:])
        result_t3[:,3] = np.where(result_t3[:,3]<initial_SVID[3],
                                  initial_SVID[3],result_t3[:,3])
        result_t3 = result_t3.astype(np.int64)
        
        if test is not None:
            score_t3 = get_scorer(scoring)._score_func(test,result_t3,
                                                       multioutput='raw_values'
                                                       )
        else:
            score_t3 = np.full(4,np.NaN)
            
        result_array.append(result_t3)
        result_score.append(score_t3)
        # result_params.append(initial_parameter.reshape(1,-1))
        result_params.append(np.full((predict_days,initial_parameter.size),
                                     initial_parameter.reshape(1,-1)))
    result_array = np.array(result_array)
    result_score = np.array(result_score)
    #DUE to TEST 3 only use 1 row params for all cases,dtype must be object
    result_params = np.array(result_params,dtype=np.float64)
    
    # if test is None, it mean that the number of forecast days is out of data 
    # range and not enough data for scoring. We will try to find the closest 
    # range of data within df to estimate the scores in this case.
    if (test is None) and (not all([isinstance(i,int) for i in initial_state])):
        
        #check if possible to gather test array
        if input_df.shape[0]-predict_days>0: 
            #new starting point
            test_start_date = input_df.loc[input_df.shape[0]-1-predict_days,
                                           date_col]
            
            #get default settings from main function
            arginfo = inspect.getargvalues(frame)
            
            #list of default function arguement names
            keys = arginfo[0] 
            #list of default function arguement value
            values = list(map(arginfo[3].get,keys))
            
            if arginfo[2] is None: #if there is no keyword arguments
                pass
            else:
                kwargs_dict = arginfo[3][arginfo[2]]
                if kwargs_dict!={}:
                    keys.extend(kwargs_dict.keys())
                    values.extend(kwargs_dict.values())
                else:
                    pass
            arg_dict = dict(zip(keys,values)) #input for recursion
            
            arg_dict['start_date']=test_start_date #change starting point
            
            try:
                test_result = SVID_modeling(**arg_dict)
            except:
                if best_model:
                    test_result = (np.NaN,"Can't find best method due to "
                                    "errors or not enough data points to " 
                                    "train|test model parameters.")
                else:
                    test_result = (np.NaN,"Can't find scores due to errors or "
                                    "not enough data points to train|test model"
                                    " parameters.")
            
            if len(test_result)==4:
                return (result_array[test_result[3]],test_result[1],
                        result_params[test_result[3]],test_result[3])
            else:
                return (result_array,test_result[1],result_params)
    
    if np.all(np.isnan(np.array(result_score))) or isinstance(result_score,str):
        pass
    elif multioutput=="uniform_average":
        result_score = np.mean(result_score,axis=1)
    elif multioutput == "variance_weighted":
        avg_weights = ((test - np.average(test, axis=0)) ** 2
                       ).sum(axis=0, dtype=np.float64)
        #since variance_weighted doesn't work with 'predict_days'=1 since the
        #weight will be 0, so we must assume the weight in that case.
        if avg_weights.sum()==0:
            avg_weights = test.sum(axis=0, dtype=np.float64)
            if avg_weights.sum()==0:
                avg_weights = [1,1,1,1]
        result_score = np.average(result_score, weights=avg_weights,axis=1)
        
    if best_model:
        if np.all(np.isnan(np.array(result_score))) or\
        isinstance(result_score,str):
            return (result_array,result_score,result_params)
        # if score is 'raw_values','uniform_average' value will be used to rank 
        if greater_better:
            if result_score.ndim!=1:
                position = np.argmax(np.mean(result_score,axis=1))    
            else:
                position = np.argmax(result_score)  
        else:
            if result_score.ndim!=1:
                position = np.argmin(np.mean(result_score,axis=1))
            else:
                position = np.argmin(result_score)
            
        result_array = result_array[position]
        result_score = result_score[position]
        result_params = result_params[position]
    if locals().get('position','nothing')=='nothing':
        return (result_array,result_score,result_params)
    else:
        return (result_array,result_score,result_params,position)

#------------------------------------------------------------------

def convert_to_daily_data(start_date,
                          input_df,
                          date_col,
                          model_cols,
                          predict_state_array,
                          predict_params_array,
                          r_protect_time):
    """
    Convert forecasted SVID and model parameters to equivelence daily data. 

    Parameters
    ----------
    start_date : str|pd.Timestamp
        Starting date of modeling, should be a valid day within 'input_df'.
        
    input_df : pd.DataFrame
        The DataFrame which contain all data for initial state of SVID model.
        
    date_col : str
        'input_df' column which contains date data.
        
    model_cols : list of str
        'input_df' columns which contain neccessary data to convert model 
        result, ['S0','V0','I0','D0',beta','beta_v','gamma','theta','alpha',
        'alpha0','daily_recovery_case']
        
    predict_state_array : np.ndarray
        Forecasted SVID state of n-days after 'start_date', (S,V,I,D)
        
    predict_params_array : np.ndarray
        Forecasted SVID model parameters of n-days after 'start_date',
        ('beta','beta_v','gamma','theta','alpha','alpha0'). If array 1-D is 1,
        broadcast all parameters.
        
    r_protect_time : int
        Average protected (immunity time) of post-covid patient (in days).

    Returns
    -------
    np.ndarray.
        Return arrays of converted data, (n points,n output)
        With output: (daily_case_non_vaccinated,
                      daily_case_vaccinated,
                      daily_death,
                      daily_recovery,
                      daily_new_full_vac,
                      daily_vaccine_boost_req)
    """
    
    assert isinstance(start_date,(str,pd.Timestamp)),"'start_date' should be "\
                                                    "string or pd.Timestamp."
    assert isinstance(input_df,pd.DataFrame),"'input_df' should be DataFrame."
    assert isinstance(date_col,str),"'date_col' should be string."
    
    assert (isinstance(model_cols,list) and len(model_cols)==11) and \
    all([isinstance(i,str) for i in model_cols]),\
    "'model_param' should be a list of 11 elements, read document for "
    "reference."
    
    if predict_state_array.shape[1]!=4:
        raise ValueError("'predict_state_array' columns doesn't match input "
                         "requirement (need 4 columns).")
    
    if predict_params_array.shape[1]!=6:
        raise ValueError("'predict_params_array' columns doesn't match input "
                         "requirement (need 6 columns).")
        
    if (predict_state_array.shape[0]!=predict_params_array.shape[0]) and\
        predict_params_array.shape[0]!=1:
        raise ValueError("Both 'predict_state_array' and "
                         "'predict_params_array' should match in length.")
    
    
    assert isinstance(r_protect_time,int),"'r_protect_time' should be an "
    "integer number."
    
    #model start point
    start_p = input_df.loc[input_df[date_col]==start_date]
    start_idx = start_p.index[0]
    
    initial_array = np.concatenate((start_p[model_cols[:4]].to_numpy(),
                                   predict_state_array))
    
    n = predict_state_array.shape[0]
    if predict_params_array.shape[0]==1: #duplication for fast calculation
        predict_params_array = np.full((n,6),predict_params_array)
    
    params_array = np.concatenate((start_p[model_cols[4:10]].to_numpy(),
                                   predict_params_array))
    
    S,V,I =  initial_array[:-1,:3].T
    N = initial_array[:-1].sum(axis=1).T
    
    #forecast infect case
    f_daily_nonvac_case = params_array[:-1,0] * ((I*S/N).T)
    f_daily_vac_case = params_array[:-1,1] * ((I*V/N).T)
    
    #forecast death case
    f_daily_death = params_array[:-1,2] * (I.T)
    
    #forecast recovery case
    f_daily_recov = params_array[:-1,3] * (I.T)
    
    #forecast new full vaccinated
    f_daily_full_vac = params_array[:-1,4] * (S.T)
    
    #forecast new vaccine + recovery wear-off
    
    daily_added_S = params_array[:-1,5] * (V.T)
    
        #expire post-disease protection
    recovery = Time_Sery_array(
        np.concatenate((input_df.loc[:start_idx,model_cols[-1]].to_numpy(),
                        f_daily_recov))) #known recovery + forecast recovery
    
    recovery = recovery.fill_out_range(start_idx+1 - r_protect_time,
                                       start_idx+1+n - r_protect_time,
                                       fill_low=np.NaN,
                                       fill_up=np.NaN)
    recovery = np.where(np.isnan(recovery),0,recovery)
    
        #daily vaccine wear-off predict
    # f_daily_boost_req = daily_added_S - recovery
    f_daily_boost_req = daily_added_S #no need to minus recovery case
    return np.stack((f_daily_nonvac_case,f_daily_vac_case,
                     f_daily_death,f_daily_recov,
                     f_daily_full_vac,f_daily_boost_req),axis=0).T

#------------------------------------------------------------------
def regression_score(y_true, y_pred,**kwargs):
    """
    Combination of regression scores to evaluate the model.

    Parameters
    ----------
    y_true : array
        (n_sample,n_outputs).
    y_pred : array
        (n_sample,n_outputs).
    **kwargs: 
        Extra keyword arguments to pass to all metric scoring method.

    Returns
    -------
    dict
        Return result of scoring.
    """
    
    explained_variance=metrics.explained_variance_score(y_true, y_pred,**kwargs)
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred,**kwargs)
    r2=metrics.r2_score(y_true, y_pred,**kwargs)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred,**kwargs)
    mse=metrics.mean_squared_error(y_true, y_pred,**kwargs)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred,**kwargs)
    
    return {'explained_variance':explained_variance,
            'mean_squared_log_error':mean_squared_log_error,
            'r2':r2,
            'MAE':mean_absolute_error,
            'MSE':mse,
            'RMSE':np.sqrt(mse),
            'MAPE':mape}