import pandas as pd
import numpy as np
cimport numpy as np
cimport cython
np.import_array()
from covid_science.utility_func import take_element
import warnings
ctypedef np.float64_t float64_t
ctypedef np.int64_t int64_t
ctypedef np.int32_t int32_t
#------------------------------------------------------------------
    
@cython.boundscheck(False)
@cython.wraparound(False)
def arrange_value_to_array(np.ndarray[int64_t,ndim=1] input_value,
                           unsigned int array_len,
                           np.ndarray[float64_t,ndim=1] init_value=None,
                           np.ndarray[float64_t,ndim=2] init_array=None,
                           np.ndarray p=None,
                           np.ndarray upper_lim=None,
                           unsigned int method_limit=30,
                           seed=None):
    """
    arrange_value_to_array(input_value,
                           array_len,
                           init_value=None,
                           init_array=None,
                           p=None,
                           upper_lim=None,
                           method_limit=30,
                           seed=None):
    
    Fill whole number to an array (like putting beans into a set of glass). 
    Fill only up to the limit of each position. This function designed to work 
    only on 1-D|2-D array.

    Parameters
    ----------
    input_value : 1-D np.ndarray
        Number value to fill in the array, can only have size=1 if 'init_value'
        is used. Each array element is equal to the amount use to fill for an 
        array row if 'init_array' is used.
        
    array_len : integer
        Length of fill-in array.
        
    init_value : 1-D float np.ndarray, optional
        Default numbers inside the 1-D array. Decimal will be ignored.
        If None, initial values is 0, also don't take infinite value.
        
    init_array: 2-D float np.ndarray, optional
        Default numbers inside the 2-D array. Decimal will be ignored.
        If None, initial values is 0, also don't take infinite value.
        Each element along the first axis will be used for fill in.
        
    p : 1-D|2-D float np.ndarray, optional
        Distribution ratio, can be either percentage or not.
        1-D array will be broadcast if 'init_array' is used.
        2-D array input must match 'init_array' shape.
        If None, distribution is even for all cases.
        NOTE: Calculation won't be based on distribution probability but 
        expectation (E=sigma(p*x)).Therefore, with small sample, result might 
        not as expected, but calculation for big sample is faster than using 
        numpy.random.choice.
        
    upper_lim : 1-D|2-D float np.ndarray, optional
        Upper limit of each position. Decimal will be ignored.
        1-D array will be broadcast if 'init_array' is used.
        2-D array input must match 'init_array' shape.
        If None, default limit is infinite.
        
    method_limit: int, optional
        The up to limit which np.random.choice will be used instead of fast 
        filling method. The main purpose of this is to keep the randomness to a
        certain degree if necessary but still keep a fast calculation speed.
    
    seed: int, optional
        Seed for random generation. If 'method_limit'=0, seed won't be applied.
        
    Returns
    -------
        (loop_array,remain_array)
    
    loop_array : 2-D array
        Return array with fill-in values.
        
    remain_value : 1-D np.ndarray
        The remain values not filled.
    """
    cdef np.ndarray initial_array
    cdef np.ndarray[float64_t,ndim=1] new_p,fractional,integral,trim_array,remain_array
    cdef np.ndarray[float64_t,ndim=2] loop_array,loop_p,loop_up_lim
    cdef np.ndarray[int64_t,ndim=1] match_idx,rank,random_choice,counts
    cdef int i,n,loop_length
    cdef float current_sum,lim_sum,remain_value
    cdef object rng_gen
    
    #condition for input between {input_value,init_value,init_array}
    if input_value.shape[0] == 0:
        raise ValueError("'input_value' can't be empty.")
    elif np.any(input_value<0):
        raise ValueError("'input_value' don't take negative values.")
    elif not np.all(np.isfinite(input_value)):
        raise ValueError("'input_value' don't take infinite values.") 
    
    if (init_value is not None) and (init_array is not None):
        raise TypeError("'init_value' and 'init_array' can't be used at the "\
                        "same time.")
    elif (init_value is None) and (init_array is None): 
        if input_value.shape[0] == 1:
            initial_array = np.zeros(array_len)
        else:
            initial_array = np.zeros((input_value.shape[0],array_len))
    elif init_value is not None: #use for scalar init
        if input_value.shape[0] == 1:
            #dont take infinity initial values.
            initial_array = np.floor(init_value)
            if initial_array.shape[0] != array_len:
                raise ValueError("'init_value' length should match 'array_len'.")
            if not np.all(np.isfinite(initial_array)):
                raise ValueError("'init_value' don't take infinite values.")
        else:
            raise ValueError("'input_value' length must be 1 when using "\
                             "'init_value'.")
    else: #use for 2-D array init input
        if input_value.shape[0] == init_array.shape[0]:
            #dont take infinity initial values.
            initial_array = np.floor(init_array)
            if initial_array.shape[1] != array_len:
                raise ValueError("'init_array' 2-D length should match "\
                                 "'array_len'.")
            if not np.all(np.isfinite(initial_array)):
                raise ValueError("'init_array' don't take infinite values.")
        else:
            raise ValueError("'input_value' array length should match "\
                             "'initial_array' 1nd dimension length.")
    #initial condition for input p
    if p is not None:
        if p.dtype.kind != 'f':
            raise TypeError("'p' array only take np.float64 values.")
        elif p.ndim not in [1,2]:
            raise ValueError("'p' array only take 1 or 2-D array.")
    #set condition for compatitble with init_value and init_array
    if initial_array.ndim ==1:
        if p is None:
            p=np.full(array_len,1.0)
        elif p.ndim != 1:
            raise ValueError("'p' dimension doesn't match 'init_value'.")
        elif p.shape[0]!=array_len:
            raise ValueError("'p' length doesn't match 'array_len'.")
    else:
        if p is None:
            p = np.full((initial_array.shape[0],array_len),1.0)
        elif (p.ndim == 2) and (p.shape[1]!=array_len):
            raise ValueError("'p' 2nd-Dimension length doesn't match 'array_len'.")
        elif (p.ndim == 2) and (p.shape[0]!=input_value.shape[0]):
            raise ValueError("'p' 1nd-Dimension length doesn't match "\
                             "'input_value' length.")
        elif (p.ndim == 1) and p.shape[0]!=array_len:
            raise ValueError("'p' length doesn't match 'array_len'.")
    #initial condition for input upper_lim
    if upper_lim is not None:
        if upper_lim.dtype.kind != 'f':
            raise TypeError("'upper_lim' array only take np.float64 values.")
        elif upper_lim.ndim not in [1,2]:
            raise ValueError("'upper_lim' array only take 1 or 2-D array.")
    
    #set condition for compatitble with init_value and init_array
    if initial_array.ndim ==1:
        if upper_lim is None:
            upper_lim = np.full(array_len,np.inf)
        elif upper_lim.ndim != 1:
            raise ValueError("'upper_lim' dimension doesn't match 'init_value'.")
        elif upper_lim.shape[0]!=array_len:
            raise ValueError("'upper_lim' length doesn't match 'array_len'.")
        upper_lim=np.floor(upper_lim)
    else:
        if upper_lim is None:
            upper_lim = np.full((initial_array.shape[0],array_len),np.inf)
        elif (upper_lim.ndim == 2) and (upper_lim.shape[1]!=array_len):
            raise ValueError("'upper_lim' 2nd-Dimension length doesn't match 'array_len'.")
        elif (upper_lim.ndim == 2) and (upper_lim.shape[0]!=input_value.shape[0]):
            raise ValueError("'upper_lim' 1nd-Dimension length doesn't match "\
                             "'input_value' length.")
        elif (upper_lim.ndim == 1) and upper_lim.shape[0]!=array_len:
            raise ValueError("'upper_lim' length doesn't match 'array_len'.")
        upper_lim=np.floor(upper_lim)
                    
    if np.any(initial_array>upper_lim):
        raise ValueError("init values must be smaller|equal comparing to limits.")
    
    #sychronize calculation when using either 'init_value' or 'init_array'
    if input_value.shape[0]==1 and ((init_value is not None) or \
                                (init_value is None) and (init_array is None)):
        loop_length = 1
        loop_array = initial_array[np.newaxis,:]
        loop_p = p[np.newaxis,:]
        loop_up_lim = upper_lim[np.newaxis,:]
    else:
        loop_length = input_value.shape[0]
        loop_array = initial_array
        if p.ndim ==1:
            loop_p = np.full((input_value.shape[0],array_len),p)
        else:
            loop_p = p
        if upper_lim.ndim ==1:
            loop_up_lim = np.full((input_value.shape[0],array_len),upper_lim)
        else:
            loop_up_lim = upper_lim
    
    
    remain_array = np.array([])
    for i in range(loop_length):
        #maximum fill in amount for each loop
        lim_sum = np.where(loop_p[i]>0,loop_up_lim[i],0).sum()
        remain_value = float(input_value[i]) #remain value to fill after each while loop
        current_sum = loop_array[i].sum() #array current sum
    
        if input_value[i] >= method_limit:
            #only stop when no more input value or array is full.
            while remain_value>0 and current_sum<lim_sum: 
                #only add to avalable idx with upper_lim available and p>0    
                match_idx = np.where((loop_array[i]<loop_up_lim[i]) &(loop_p[i]>0))[0]
                new_p = loop_p[i][np.where((loop_array[i]<loop_up_lim[i]) &(loop_p[i]>0))]
                new_p = (new_p/new_p.sum())
                
                #separate the fractionl and integral part
                fractional,integral = np.modf(new_p*remain_value)
                loop_array[i,match_idx]+=integral
                
                n = fractional.sum().round().astype(np.int64)
                
                #add rounded fractional part to index with highest fractional without 
                #exceed total input_value
                if n>0: 
                    rank = fractional.argsort()
                    loop_array[i,match_idx[rank][-n:]] +=1
                
                #clip any excess value compare to upper_lim
                trim_array=np.clip(loop_array[i],np.zeros(array_len),loop_up_lim[i]) 
                remain_value = loop_array[i].sum()-trim_array.sum()
                loop_array[i]=trim_array
                current_sum = loop_array[i].sum()
        else: # in case of need for random output to a certain degree
            rng_gen = np.random.default_rng(seed)
            while remain_value>0 and current_sum<lim_sum:
                match_idx = np.where((loop_array[i]<loop_up_lim[i]) &(loop_p[i]>0))[0]
                new_p = loop_p[i][np.where((loop_array[i]<loop_up_lim[i]) &(loop_p[i]>0))]
                new_p = (new_p/new_p.sum())
                random_choice = rng_gen.choice(match_idx,int(remain_value),p=new_p)
                match_idx,counts = np.unique(random_choice,return_counts=True)
                
                loop_array[i,match_idx] += counts
                    
                trim_array=np.clip(loop_array[i],np.zeros(array_len),loop_up_lim[i]) 
                remain_value = loop_array[i].sum()-trim_array.sum()
                loop_array[i]=trim_array
                current_sum = loop_array[i].sum()
        
        remain_array = np.append(remain_array,remain_value)
    return loop_array,remain_array

#------------------------------------------------------------------
ctypedef fused optional_2:
    float
    np.ndarray
    
@cython.boundscheck(False)
@cython.wraparound(False)
def model_initial(np.ndarray[float64_t,ndim=2] input_data,
                  int r_protect_time,
                  optional_2 s_vac_ratio):
    """
    model_initial(input_data,
                  r_protect_time,
                  s_vac_ratio)
    
    Estimate the initial state of model input (S0,V0,I0,D0) at a given time in
    the time sery. Please note that for the function to work correctly, data 
    should be a time sery from the beginning of the pandemic.
    
    Parameters
    ----------
    input_data : 2-D np.ndarray of float|int
        The data array needed for calculation, (n_points,n_parameter)
        With n_parameter require ('new_daily infect','new_daily_recover',
                                  'new_daily_death','current_full_vaccinated',
                                  'disease_susceptible_population')
    r_protect_time : int
        Average protected (immunity time) of post-covid patient (in days).
        
    s_vac_ratio : float|np.ndarray of float
        The infection chance ratio between vaccinated and non-vaccinated
        individuals. (should be bigger or at least equal to 1).
        Please use float data.

    Returns
    -------
    np.ndarray
        (n_points,output), with output: ('S0','V0','I0','D0',
                                         'daily_case_non_vaccinated',
                                         'daily_case_vaccinated')
    """
    
    if r_protect_time<0:
        raise ValueError("'r_protect_time' should be an non-negative number.")
       
    cdef np.ndarray[float64_t, ndim=1] curr_I,curr_D,current_vaccinated
    cdef np.ndarray[float64_t, ndim=1] total_recov,wore_off_r,case_sum
    cdef np.ndarray[float64_t, ndim=1] curr_V = np.zeros(input_data\
                                                            .shape[0])
    cdef np.ndarray[float64_t, ndim=1] curr_S = np.zeros(input_data\
                                                            .shape[0])
    cdef np.ndarray[float64_t, ndim=1] daily_vac_case = np.zeros(input_data\
                                                                .shape[0])
    cdef int i
    cdef float v0_check,pre_S,pre_V,input_V,v_infect_per,v_infected,ratio
    cdef np.ndarray[float64_t, ndim=1] ratio_array
    
    if isinstance(s_vac_ratio,np.ndarray):
        if np.any(s_vac_ratio<1):
            raise ValueError("All 's_vac_ratio' array values should be bigger"\
                             " or at least equal to 1.")
        ratio_array = s_vac_ratio
    else:
        if s_vac_ratio<1:
            raise ValueError("'s_vac_ratio' value should be bigger or at"\
                             " least equal to 1.")
        ratio = s_vac_ratio
    
    #total death case by day
    curr_D = np.nancumsum(input_data[:,2],dtype=np.float64)
    
    #daily current vaccinated population
    current_vaccinated = input_data[:,3]
    
    #total recovery case by day
    total_recov = np.nancumsum(input_data[:,1],dtype=np.float64)
    
    #daily active infected case
    curr_I = np.nancumsum(input_data[:,0],dtype=np.float64)\
             - curr_D - total_recov
    
    if r_protect_time >= input_data.shape[0]:
        case_sum = np.zeros(input_data.shape[0])
    else:
        case_sum = np.hstack((np.zeros(r_protect_time),
                              np.nancumsum(input_data[:,1],dtype=np.float64)\
                              [:input_data.shape[0]-r_protect_time])
                             )
    #daily after-infected protection wore-off population    
    wore_off_r= np.where(np.arange(total_recov.shape[0])<r_protect_time,0,case_sum)
    
    #estimate S,V of first data point in the array
    v0_check = current_vaccinated[0] + total_recov[0] - wore_off_r[0]
    if v0_check > 0:
        #nominated S and V use for calculate %vaccinated in daily infected case
        pre_S = input_data[0,4]- curr_I[0] - curr_D[0] - v0_check
        pre_V = current_vaccinated[0] + total_recov[0] - wore_off_r[0]
        if isinstance(s_vac_ratio,np.ndarray):
            # not real % of 'v_infect_per' but estimated, the bigger the 
            # number of infected case, the more accuracy it is
            v_infect_per = (pre_V)/(pre_S*ratio_array[0]+pre_V)
        else:
            v_infect_per = (pre_V)/(pre_S*ratio+pre_V)
        v_infected = v_infect_per*input_data[0,0]
        curr_V[0] = v0_check - v_infected
        
        #daily infected case that already vaccinated
        daily_vac_case[0] = v_infected
    else:
        curr_V[0] = v0_check
    curr_S[0] = input_data[0,4]-curr_I[0]-curr_D[0]-curr_V[0]
    
    #calculate the rest of S,V in the array by using previous S,V
    if isinstance(s_vac_ratio,np.ndarray): #in 
        for i in range(1,input_data.shape[0]):
            pre_S = curr_S[i-1]
            pre_V = curr_V[i-1]
            v_infect_per = (pre_V)/(pre_S*ratio_array[i]+pre_V)
            v_infected = v_infect_per*input_data[i,0]
            curr_V[i] = current_vaccinated[i] + total_recov[i] - wore_off_r[i] - v_infected
            curr_S[i] = input_data[i,4]-curr_I[i]-curr_D[i]-curr_V[i]
            daily_vac_case[i] = v_infected
    else:
        for i in range(1,input_data.shape[0]):
            pre_S = curr_S[i-1]
            pre_V = curr_V[i-1]
            v_infect_per = (pre_V)/(pre_S*ratio+pre_V)
            v_infected = v_infect_per*input_data[i,0]
            curr_V[i] = current_vaccinated[i] + total_recov[i] - wore_off_r[i] - v_infected
            curr_S[i] = input_data[i,4]-curr_I[i]-curr_D[i]-curr_V[i]
            daily_vac_case[i] = v_infected
    
    return np.stack((curr_S,curr_V,curr_I,curr_D,input_data[:,0]-daily_vac_case,
                     daily_vac_case),axis=1)
            
#------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def model_parameter(np.ndarray[float64_t,ndim=2] input_data,
                     int r_protect_time,
                     int avg_infect_t=1,
                     int avg_death_t=1,
                     int avg_recov_t=1,
                     int avg_rotate=1):
    """
    model_parameter(input_data,
                    r_protect_time,
                    avg_infect_t=1,
                    avg_death_t=1,
                    avg_recov_t=1,
                    avg_rotate=1):
        
    Estimate the input parameter of SVID model at a given time in
    the time sery.

    Parameters
    ----------
    input_data : 2-D np.ndarray of float|int
        The data array needed for calculation, (n_points,n_parameter)
        With n_parameter require:
            ['init_susceptible','init_vaccinated','init_infected',
             'daily_death_case','daily_recov_case','daily_new_full_vacc',
             'daily_boost_require','daily_new_case_non_vaccinated',
             'daily_new_case_vaccinated'','disease_susceptible_population']
        
    r_protect_time : int
        Average protected (immunity time) of post-covid patient (in days).
        
    avg_infect_t : int, optional
        Estimate beta,beta_v at a specific date based on data of n days ago.
        
    avg_death_t : int, optional
        Estimate theta at a specific date based on data of n days ago.
        
    avg_recov_t : int, optional
        Estimate gamma at a specific date based on data of n days ago.
        
    avg_rotate : int, optional
        Estimate alpha,alpha0 at a specific date based on data of n days ago.

    Returns
    -------
    np.ndarray
        (n_points,output), with output: ('beta,'beta_v','gamma','theta',
                                         'alpha','alpha0')

    """
    
    if r_protect_time<0:
        raise ValueError("'r_protect_time' should be an non-negative number.")
    if np.any(np.array([avg_infect_t,avg_death_t,avg_recov_t,avg_rotate])<1):
        raise ValueError("Data slice of ('avg_infect_t','avg_death_t', "\
                         "'avg_recov_t','avg_rotate') should be at least 1.")
    
    cdef np.ndarray[float64_t, ndim=1] S,V,I,N,new_D,new_R,new_I_non
    cdef np.ndarray[float64_t, ndim=1] new_I_vac,daily_vac,boost
    cdef np.ndarray[float64_t, ndim=1] beta,beta_v,gamma,theta,alpha,alpha0
    cdef np.ndarray[int32_t, ndim=2] idx_array1,idx_array2
    cdef np.ndarray[float64_t, ndim=1] I_mean,S_mean,N_mean,V_mean,D_mean
    cdef np.ndarray[float64_t, ndim=1] R_mean,f0_n,f0_v,vaccinated
    cdef np.ndarray[float64_t, ndim=1] wore_off_r,daily_boost,added_S
    cdef int max_t,max_col
    
    max_t = max(avg_infect_t,avg_death_t,avg_recov_t,avg_rotate)
    max_col = input_data.shape[1]
    
    S,V,I,new_D,new_R,daily_vac,\
    boost,new_I_non,new_I_vac,N = np.vstack((np.full((max_t,max_col),np.NaN),
                                             input_data,
                                             np.full((1,max_col),np.NaN)
                                             )).T
    
    warnings.simplefilter("ignore", category=RuntimeWarning)
    # #calculate beta,beta_v
    idx_array1 = take_element(avg_infect_t,input_data.shape[0],avg_infect_t)
    idx_array2 = take_element(avg_infect_t+1,input_data.shape[0],avg_infect_t)
    
    I_mean = np.nanmean(np.take(I,idx_array1),axis=1)
    S_mean = np.nanmean(np.take(S,idx_array1),axis=1)
    N_mean = np.nanmean(np.take(N,idx_array1),axis=1)
    V_mean = np.nanmean(np.take(V,idx_array1),axis=1)
    f0_n = np.nanmean(np.take(new_I_non,idx_array2),axis=1)
    f0_v = np.nanmean(np.take(new_I_vac,idx_array2),axis=1)
    
    beta = N_mean*f0_n/(I_mean*S_mean)
    beta = np.where(f0_n==0,np.NaN,beta)
    beta = np.where(f0_n==np.inf,np.NaN,beta)
    beta = np.where(np.isinf(beta),np.NaN,beta)
    
    beta_v = N_mean*f0_v/(I_mean*V_mean)
    beta_v = np.where(f0_v==0,np.NaN,beta_v)
    beta_v = np.where(f0_v==np.inf,np.NaN,beta_v)
    beta_v = np.where(np.isinf(beta_v),np.NaN,beta_v)
    
    ## #calculate gamma
    idx_array1 = take_element(avg_death_t,input_data.shape[0],avg_death_t)
    idx_array2 = take_element(avg_death_t+1,input_data.shape[0],avg_death_t)
    
    I_mean = np.nanmean(np.take(I,idx_array1),axis=1)
    D_mean = np.nanmean(np.take(new_D,idx_array2),axis=1)
    
    gamma = D_mean/I_mean
    gamma = np.where(D_mean==0,np.NaN,gamma)
    gamma = np.where(I_mean==np.inf,np.NaN,gamma)
    gamma = np.where(np.isinf(gamma),np.NaN,gamma)
    
    ## #calculate theta
    idx_array1 = take_element(avg_recov_t,input_data.shape[0],avg_recov_t)
    idx_array2 = take_element(avg_recov_t+1,input_data.shape[0],avg_recov_t)
    
    I_mean = np.nanmean(np.take(I,idx_array1),axis=1)
    R_mean = np.nanmean(np.take(new_R,idx_array2),axis=1)
    
    theta = R_mean/I_mean
    theta = np.where(R_mean==0,np.NaN,theta)
    theta = np.where(I_mean==np.inf,np.NaN,theta)
    theta = np.where(np.isinf(theta),np.NaN,theta)
    
    ## #calculate alpha
    
    idx_array1 = take_element(avg_rotate,input_data.shape[0],avg_rotate)
    idx_array2 = take_element(avg_rotate+1,input_data.shape[0],avg_rotate)
    
    S_mean = np.nanmean(np.take(S,idx_array1),axis=1)
    vaccinated = np.nanmean(np.take(daily_vac,idx_array2),axis=1)
    
    alpha = vaccinated/S_mean
    alpha = np.where(vaccinated==0,np.NaN,alpha)
    alpha = np.where(S_mean==np.inf,np.NaN,alpha)
    alpha = np.where(np.isinf(alpha),np.NaN,alpha)
    
    ## #calculate alpha0
    
    new_R = np.hstack((np.full(max_t+r_protect_time,np.NaN),input_data[:,4],
                       np.full(1,np.NaN)))
    
    idx_array1 = take_element(avg_rotate,input_data.shape[0],avg_rotate)
    idx_array2 = take_element(avg_rotate+1,input_data.shape[0],avg_rotate)
    
    wore_off_r = np.nanmean(np.take(new_R,idx_array2),axis=1)
    wore_off_r = np.where(np.isnan(wore_off_r),0,wore_off_r)
    daily_boost = np.nanmean(np.take(boost,idx_array2),axis=1)
    daily_boost = np.where(np.isnan(daily_boost),0,daily_boost)
    
    added_S = wore_off_r + daily_boost
    added_S[added_S.shape[0]-1]=np.NaN # due to last point 'daily_boost'=np.NaN
    V_mean = np.nanmean(np.take(V,idx_array1),axis=1)
    
    alpha0 = added_S/V_mean
    alpha0 = np.where(added_S==0,np.NaN,alpha0)
    alpha0 = np.where(V_mean==np.inf,np.NaN,alpha0)
    alpha0 = np.where(np.isinf(alpha0),np.NaN,alpha0)
    
    warnings.simplefilter("default", category=RuntimeWarning)
    
    return np.stack((beta,beta_v,gamma,theta,alpha,alpha0),axis=1)