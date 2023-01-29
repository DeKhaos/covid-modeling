"""This module collect functions that are used by other modules, which to 
avoid circular module import."""

import numpy as np
import pandas as pd
import math
import dash_bootstrap_components as dbc
import plotly.express as px
import re
import json

#extra module import for application plotting purpose
from numpy import format_float_scientific as ffs
import datetime
#------------------------------------------------------------------

def take_element(a,b,c):
    """
    Simple fancy index with the purpose to take x previous element from an 
    1-D array but with moving start index.

    Parameters
    ----------
    a : int
        starting index (included in taken elements)
        
    b : int
        Number of repeatation along the axis.
        
    c: int
        Number of element to take each slice.
        
    Returns
    -------
    np.ndarray
        Indice matrix.
    """
    x = np.arange(a+1-c,a+1).reshape(1,-1)
    y = np.arange(0,b).reshape(-1,1)
    return x+y

#------------------------------------------------------------------

class Time_Sery_array(np.ndarray):
    """
    Subclass of np.ndarray which used to process time sery data and some new
    supported funcions.
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        #assert obj.ndim ==1,"Take only 1-dimention array."
        return obj   
    def fill_out_range(obj,a,b,fill_low=0,fill_up=0,obj_array=False):
        """
        Fill in array with fill in value if call indexes are out of array range.
        This function only means to work for the index of 1st axis (row index).
        
        Parameters
        ----------
        a: int
            Lower index.
            
        b: int
            Upper index.
            
        fill_low: int|float|np.array|list, optional
            Fill value for indexes out of range in lower section.
            If obj_array=True, can input as list|np.array.
            
        fill_up: int|float|np.array|list, optional
            Fill value for indexes out of range in upper section.
            If obj_array=True, can input as list|np.array.
            
        obj_array: bool, optional
            If True,take whole 'fill_up','fill_up' sequence as an array element.
            If False, fill in value will try to match input array shape.
            Only work for 1-D array.
            
        Return
        ----------
        np.array
            Modified  np.ndarray to match purpose.
            
        """
        if obj_array:
            assert all(isinstance(i,int) for i in [a,b]),"Index should be integer."
            assert obj.ndim==1,"'obj_array' only work on 1-D array."
        else:
            assert all(isinstance(i,(int,np.int32,np.int64)) for i in [a,b]),"Index should be integer."
            assert type(fill_low) in [int,float,str,np.int32,np.int64,np.float32,
                                      np.float64,np.NaN],\
                "'fill_low' should be int|float|str|np.NaN."
            assert type(fill_up) in [int,float,str,np.int32,np.int64,np.float32,
                                     np.float64,np.NaN],\
                "'fill_up' should be int|float|str|np.NaN."
        up_idx = len(obj)-1
        low_shape = list(obj.shape)
        up_shape = list(obj.shape)
        
        if a>b:
            raise Exception("Lower index is bigger than upper index.")
        else:
            if a<0 and b>up_idx+1:
                low_shape[0]=abs(a)
                low_shape=tuple(low_shape)
                if obj_array: #take whole 'fill_low' as an array element
                    extend_a = np.full(
                        (low_shape[0],2),
                        np.array([np.array(fill_low,dtype='O'),None],dtype='O')
                        )
                    extend_a = extend_a[:,0]
                else:
                    extend_a = np.full(low_shape,fill_low)
                
                up_shape[0]=b-up_idx-1
                up_shape=tuple(up_shape)
                if obj_array: #take whole 'fill_up' as an array element
                    extend_b = np.full(
                        (up_shape[0],2),
                        np.array([np.array(fill_up,dtype='O'),None],dtype='O')
                        )
                    extend_b = extend_b[:,0]
                else:
                    extend_b = np.full(up_shape,fill_up)
                    
                return np.concatenate((extend_a,obj,extend_b))
            elif b>up_idx+1 and a<=up_idx:
                up_shape[0]=b-up_idx-1
                up_shape=tuple(up_shape)
                if obj_array:
                    extend_b = np.full(
                        (up_shape[0],2),
                        np.array([np.array(fill_up,dtype='O'),None],dtype='O')
                        )
                    extend_b = extend_b[:,0]
                else:
                    extend_b = np.full(up_shape,fill_up)
                return np.concatenate((obj[a:],extend_b))
            elif a>up_idx and b>a: #if both excess max index, array of only fill in value
                up_shape[0]=b-a
                up_shape=tuple(up_shape)
                if obj_array:
                    extend_b = np.full(
                        (up_shape[0],2),
                        np.array([np.array(fill_up,dtype='O'),None],dtype='O')
                        )
                    extend_b = extend_b[:,0]
                    return extend_b
                else:
                    return np.full(up_shape,fill_up)
            elif b<0: # if both negative, array of only fill in value
                low_shape[0]=b-a
                low_shape=tuple(low_shape)
                if obj_array:
                    extend_a = np.full((
                        low_shape[0],2),
                        np.array([np.array(fill_low,dtype='O'),None],dtype='O')
                        )
                    extend_a = extend_a[:,0]
                    return extend_a
                else:
                    return np.full(low_shape,fill_low)
            elif a<0 and b>=0:
                low_shape[0]=abs(a)
                low_shape=tuple(low_shape)
                if obj_array:
                    extend_a = np.full(
                        (low_shape[0],2),
                        np.array([np.array(fill_low,dtype='O'),None],dtype='O')
                        )
                    extend_a = extend_a[:,0]
                else:
                    extend_a = np.full(low_shape,fill_low)
                return np.concatenate((extend_a,obj[:b]))
            else:
                return obj[a:b].view(np.ndarray)    
    def weight_calculation(obj,array_weights):
        """
        Calculate the expected value from a discrete probability distribution.
        
        Parameters
        ----------
        array_weights : np.ndarray|list
            Weight|probability of array elements.

        Returns
        -------
        result : dictionary
            Return the mean, standard deviation and variance.
        """
        weight = np.array(array_weights)
        
        assert obj.dtype.kind in ['i','f'],\
            "Function only support 'input_array' with integer|float dtype."
        assert obj.ndim==1,"Function only support 1-D arrays."
        assert weight.ndim==1,"'array_weights' must be 1-D array."
        assert weight.dtype.kind in ['f','i'],\
            "Function only support 'array_weights' with integer|float dtype."
        assert weight.size==obj.size,\
            "'array_weights' length must match 'input_array' length."
        assert any(np.isnan(weight))==False and any(np.isnan(obj))==False,\
            "Function doesn't support arrays with NaN values."
        
        value = obj.view(np.ndarray) 
        weight = weight/weight.sum()
        mean = (value*weight).sum()
        var = (np.power((value-mean),2)*weight).sum()
        std = np.sqrt(var)
        
        obj.weighted_std = std
        obj.weighted_mean = mean
        obj.weighted_var = var
        result = {"mean":mean,"variance":var,"std":std}
        
        return result

#------------------------------------------------------------------

def trim_mean(input_data,rolling_window=7,trim_per=50):
    """
    Calculate mean of remaining data point in a roll after trimming outlier.
    
    Parameters
    ----------
    input_data : np.ndarray|pd.Series|pd.DataFrame
        Only work with numeric arrays. The maximum number of dimension for 
        array is 2.
        
    rolling_window: int
        Number of data point to take for each roll. Minimum is 1.
    
    trim_per : integer/float
        The percentage of data that will be trimmed.
        
    Returns
    -------
    np.ndarray (n_data_point,n_variable)
        Return the mean of remaining numeric values.
    """
    if type(input_data)==pd.DataFrame:
        array = np.array(input_data.T)
    else:
        array = np.array(input_data)
        if array.ndim not in [1,2]:
            raise ValueError("'input_data' should have at most 2-D.")
            
    if array.dtype.kind not in ['i','f']:
        raise TypeError("'input_array' should be only be integer|float type.")
    
    if rolling_window<1:
        raise ValueError("'rolling_window' must be at least 1.")
    
    if array.ndim==1 and np.all(np.isnan(array[:rolling_window])):
        raise ValueError("'input_array' can't contain NaN values.")
    elif array.ndim==2 and np.all(np.isnan(array[:,:rolling_window])): 
        raise ValueError("Each 'input_array' variable can't contain NaN values.")
        
    if trim_per<0 or trim_per>100:
        raise Exception("Trim percent not valid.")
    
    #trimming index
    lower = math.ceil((rolling_window-1)*trim_per/200)
    upper = math.ceil((rolling_window-1)*(100-trim_per/2)/100)
    if lower==upper:
        raise ValueError("No data point remains for the input 'trim_per', "\
                         "please reduce the number.")
    if array.ndim==1:
        max_idx = len(array)-1
    else:
        max_idx = array.shape[1]-1
    
    #rolling surpass max index
    if rolling_window>max_idx+1:
        return np.full(array.shape,np.NaN).T
    
    #repeat rolling for each variable along the 1st axis
    slice_idx = take_element(rolling_window-1,
                             (max_idx+1)-(rolling_window-1),
                             rolling_window)
    if array.ndim==1:
        rolling_array = np.sort(np.take(array,slice_idx),axis=1)
        rolling_array = rolling_array[:,lower:upper+1].mean(axis=1)
        result_array = np.append(np.full(rolling_window-1,np.NaN),
                                 rolling_array)
        return result_array.T
    else:
        rolling_array = np.sort(np.take(array,slice_idx,axis=1),axis=2)
        rolling_array = rolling_array[:,:,lower:upper+1].mean(axis=2)
        result_array = np.append(np.full((rolling_array.shape[0],
                                          rolling_window-1),np.NaN),
                                 rolling_array,
                                 axis=1)
        return result_array.T

#------------------------------------------------------------------

def SVID_ODE(initial,t,beta,beta_v,gamma,theta,alpha,alpha0,pi,mu,p):
    """
    Covid-19 SVID model for ODE system
    """
    s,v,i,d = initial
    n = s+v+i+d
    dsdt = -beta*s*i/n -(alpha+mu)*s + alpha0*v + pi*p
    dvdt = -beta_v*v*i/n -(alpha0+mu)*v + theta*i + alpha*s + pi*(1-p)
    didt = -(mu+gamma+theta)*i + (beta*s+beta_v*v)*i/n
    dddt = gamma*i -mu*d
    return [dsdt,dvdt,didt,dddt]

#------------------------------------------------------------------

def add_row_choices(titles,option_groups,select_options,group_ids,
                    persistence_type='local',persistence=None,
                    input_type=None,class_name=None,style=None,row_id=None):
    """
    A simple function for create select option components to use with dash 
    callback and plotly figures.

    Parameters
    ----------
    titles : list|tuple
        list of title for each dropdown menu.
        
    option_groups : list|tuple
        List of options for each dbc.Select.
        
    select_options: list|tuple
        Default selected values of dropdown menus
        
    group_ids : list|tuple
        Unique id for each components.
        
    persistence_type: (a value equal to: 'local', 'session', 'memory'; default 'local')
        Where persisted user changes will be stored: memory: only kept in
        memory, reset on page refresh. local: window.localStorage, data is
        kept after the browser quit. session: window.sessionStorage, data
        is cleared once the browser quit.
        
    persistence: list of boolean | string | number, optional
        Used to allow user interactions in this component to be persisted
        when the component - or the page - is refreshed.
        Must input as a list that match the number of dbc components.
        
    input_type: list of str, optional
        List of dash_bootstrap_components attribute which support `options`
        argument. Default is dbc.Select
        
    class_name: str, optional
        Often used with CSS to style elements with common properties.
        
    style: dict, optional
        Defines CSS styles which will override styles previously set.
        
    row_id: str|dict, optional
        The ID of this component, used to identify dash components in callbacks. 
        The ID needs to be unique across all of the components in an app.
        
    Returns
    -------
    Tuple
        dbc.Row
            Contains all title and dropdown menus.
            
        group_ids : tuple
            All ids with each dbc.Select.

    """
    if persistence is None:
        assert len(titles)==len(option_groups)==len(group_ids)==len(select_options),\
            "All input argument must have same length."
        persistence=[None for _ in range(len(titles))]
    else:
        assert len(titles)==len(option_groups)==len(group_ids)==len(select_options)==len(persistence),\
            "All input argument must have same length."
    
    if input_type is None:
        input_type = ['Select' for _ in range(len(titles))]
    else:
        assert len(input_type)==len(titles),\
            "'input_type' should have same length as 'titles'."
    
    base_row = []
    for item in zip(titles,option_groups,select_options,group_ids,input_type,persistence):
        element = dbc.Col([
            dbc.Label(item[0]),
            getattr(dbc,item[4])(id=item[3],
                options=item[1],
                value=item[2],
                class_name=class_name,
                style=style,
                persistence_type=persistence_type,
                persistence=item[5]
                )
            ],
            width='auto',
            class_name=class_name,
            style=style)
        base_row.append(element)
    if row_id is None:
        return (dbc.Row(base_row),tuple(group_ids))
    else:
        return (dbc.Row(base_row,id=row_id),tuple(group_ids))


#------------------------------------------------------------------

def condition_plot(plot_type,df,arg_list,default_values,preset=None):
    """
    Create figure for dash callback based on input parameters.

    Parameters
    ----------
    plot_type : str
        Plotly.express attribute that target plot type.
        
    df : pd.DataFrame
        Dataframe used to plot figure.
        
    arg_list : List of dict
        Special list of dict arguments retrieved from dash component id.
        
    default_values : list
        Selected value of dash components in 'arg_list'.
        
    preset : dict, optional
        Dict of parameters for preprocessing before plotting.

    Returns
    -------
    fig : plotly figure object
        The return figure after processing.

    """
    #create plot object
    plot_obj = getattr(px,plot_type)
    
    #preset value of plot and input dataframe
    if preset is None:
        df_arg = {}
        plot_arg = {}
        layout_arg = {}
    else:
        df_arg = {} if preset['dataframe'] is None else preset['dataframe'] 
        plot_arg = {} if preset['plot'] is None else preset['plot']
        layout_arg = {} if preset['layout'] is None else preset['layout']
    
    #apply string code to dataframe
    if df_arg=={}:
        pass
    else:
        preset_code = re.sub(r"{}","df",df_arg['code_obj'])
        if df_arg['obj_type']=='expression':
            df = eval(preset_code)
        elif df_arg['obj_type']=='statement':
            exec(preset_code)

    #processing argument list
    for item in zip(arg_list,default_values):
        
        #apply string code to dataframe
        if 'dataframe' in item[0]['target']:
            code_variable = np.array(item[0]['format'],dtype='O')
            if type(item[1])==str:
                try:
                    #check if this is a number
                    code_variable[code_variable=='variable']=int(item[1])
                except:
                    #wrap string inside single quote to keep as string when eval
                    code_variable[code_variable=='variable']="'" + item[1] + "'"
            else:
                code_variable[code_variable=='variable']=item[1]
            if item[0]['obj_type']=='expression':
                df = eval(item[0]['code_obj'].format(*code_variable))
            elif item[0]['obj_type']=='statement':
                exec(item[0]['code_obj'].format(*code_variable))
        
        #prepare plot input arguments
        if 'plot' in item[0]['target']:
            if re.match(r"\[.*\]",item[1]): #check if input is a list as str
                plot_arg[item[0]['plot_arg']]=json.loads(item[1])
            else:
                plot_arg[item[0]['plot_arg']]=item[1]
        
        #prepare layout arguments
        if item[0]['layout_arg'] is None:
            pass
        else:
            layout_arg[item[0]['layout_arg']]=item[1]
    fig = plot_obj(data_frame=df,**plot_arg)
    
    if layout_arg=={}:
        pass
    else:
        if 'special_layout' in layout_arg.keys():
            for item in layout_arg['special_layout']:
                layout_arg[item[0]]=eval(item[1])
            layout_arg.pop('special_layout')
            fig.update(**layout_arg)
        else:
            fig.update(**layout_arg)
    return fig