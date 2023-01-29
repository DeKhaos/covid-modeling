
import numpy as np
cimport numpy as np
cimport cython
np.import_array()
from sympy import symbols,Eq,MatrixSymbol,Matrix
from sympy.utilities.autowrap import autowrap
from scipy.optimize import fsolve

ctypedef np.float64_t float64_t

#------------------------------------------------------------------

def SVID_func():
    """
    SVID_func()
    
    Return the frame for calculating equilibrium points of SVID system.
    The main purpose is to convert the ODEs to C language for fast execution.

    Returns
    -------
    Tuple
        Return a tuple of ODEs for case of R0<1 and R0>=1.

    """
    
    cdef object beta,beta_v,gamma,theta,alpha,alpha0,n
    cdef object p_,pi,mu_
    cdef object y,dY
    cdef object f1,f2,f3,f4,f5,f6,f7
    cdef object c_func1,c_func2

    beta,beta_v,gamma,theta,alpha,alpha0,pi = symbols("beta,beta_v,gamma,"
                                                   "theta,alpha,alpha0,pi")
    p_,mu_ = symbols("p,mu")
    
    y = MatrixSymbol('y',4,1)
    dY = MatrixSymbol('dY',4,1)
    
    n = y[0] + y[1] + y[2] +y[3]
    f1 = -beta*y[0]*y[2]/n -(alpha+mu_)*y[0] + alpha0*y[1] + pi*p_
    f2 = -beta_v*y[1]*y[2]/n - (alpha0+mu_)*y[1] + theta*y[2]+ alpha*y[0] + pi*(1-p_)
    f3 = -(mu_+gamma+theta) + (beta*y[0]+beta_v*y[1])/n
    f4 = -n*mu_ + pi
    
    f5 = (-(alpha+mu_)*y[0] + alpha0*y[1] + pi*p_)
    f6 = (- (alpha0+mu_)*y[1] + alpha*y[0] + pi*(1-p_))
    f7 = y[2]
    
    #C function for R0<1
    c_func1 = autowrap(Eq(dY,Matrix((f5,f6,f7,f4))),language='C',
                      backend='cython',
                      args=(y,beta,beta_v,gamma,theta,alpha,alpha0,
                            pi,p_,mu_)
                      )
    #C function for R0>=1
    c_func2 = autowrap(Eq(dY,Matrix((f1,f2,f3,f4))),language='C',
                      backend='cython',
                      args=(y,beta,beta_v,gamma,theta,alpha,alpha0,
                            pi,p_,mu_)
                      )
   
    return c_func1,c_func2

#------------------------------------------------------------------

def auto_wrapper(x0,f,*args):
    """
    auto_wrapper(x0,f,*args)
    
    Simple wrapper to increase the dimension of input from 'a' function and 
    reduce the dimension of 'b' function output.
    
    The main application is for scipy.optimize methods when using Cython & C
    functions.

    Parameters
    ----------
    x0 : np.ndarray
        Input array of x.
        
    f : function
        C function wrapped by Cython.
        
    *args : optional
        Extra arguments for 'f'.

    Returns
    -------
    np.ndarray
        'f' output.
    """
    cdef object dY
    dY = f(x0[:,np.newaxis],*args)
    return dY.squeeze()

#------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def find_equilibrium(np.ndarray[float64_t, ndim=2] model_params,
                     np.ndarray[object, ndim=1] equil_func,
                     float p,
                     float pi,
                     float mu,
                     int step = 10):
    """
    find_equilibrium(np.ndarray[float64_t, ndim=2] model_params,
                     np.ndarray[object, ndim=1] equil_func,
                     float p,
                     float pi,
                     float mu,
                     int step = 10)
    
    Calculate the equilibrium states of a given array of initial model 
    conditions at each data point.
    
    Calculation will return array of np.NaN for data entry without a possibly 
    equilibrium state.
    
    Parameters
    ----------
    model_params : np.ndarray
        Model input requirements for calculate equilibrium state.
        ['beta','beta_v','gamma','theta','alpha','alpha0','R0']
        
    equil_func : np.ndarray of 'O' dtype
        Array which contain ODEs for case of R0<1 and R0>=1. This will be used
        for equilibrium states calcuation.
        
    p : float
        Non-vaccinated ratio of recruitment population.
        
    pi : float
        Population recruit rate.
        
    mu : float
        Population death rate.
        
    step : int, optional
        Step for refining initial state guess range. The bigger the step the 
        longer the calculation.

    Returns
    -------
    equil_array : np.ndarray
        The calculated equilibrium states of 'model_params' inputs.

    """
    
    #initial guess of the equilibrium state
    #R0<1
    cdef np.ndarray[float64_t,ndim=1] init_c1 = np.array([pi/mu/4,pi/mu/4,
                                                          0,0],
                                                          dtype=np.float64)
    #R0>=1
    cdef np.ndarray[float64_t,ndim=1] init_c2 = np.array([pi/mu/4,pi/mu/4,
                                                          pi/mu/10,pi/mu/2],
                                                          dtype=np.float64)
    
    # define new_init,test_init for retrying calcuation when fsolve return
    # negative values by using some new intial guess.
    cdef np.ndarray[float64_t,ndim=1] new_init = np.arange(pi/mu,-pi/mu,
                                                           step = -pi/mu/step)
    cdef int n = len(new_init)
    cdef np.ndarray[float64_t,ndim=2] test_init = np.full((n,4),
                                                          new_init.reshape(-1,1
                                                                           ))
    cdef int i,j
    cdef object func
    cdef np.ndarray[float64_t,ndim=1] check_result
    
    cdef np.ndarray[float64_t, ndim=2] equil_array = np.zeros(
                                           (model_params.shape[0],4)
                                                             ) #output array
                                               
    for i in range(model_params.shape[0]):
        
        if model_params[i,6]<1: #R0<1
            func = equil_func[0]
            check_result= fsolve(auto_wrapper,
                                  init_c1,
                                  args = (func,*model_params[i,:6].tolist(),pi,
                                          p,mu),
                                  xtol=1e-9
                                  )
        else:
            func = equil_func[1]
            check_result= fsolve(auto_wrapper,
                                  init_c2,
                                  args = (func,*model_params[i,:6].tolist(),pi,
                                          p,mu),
                                  xtol=1e-9
                                  )
        #if there is a negative value, recalculate the equilbrium point
        if np.any(check_result<0) and (not np.allclose(check_result,
                                                       np.abs(check_result))):
            
            for j in range(n):
                check_result= fsolve(auto_wrapper,
                                     test_init[j], #new initial guess
                                     args = (func,*model_params[i,:6].tolist(),
                                             pi,p,mu),
                                     xtol=1e-9
                                    )
            
                if np.all(check_result>0):
                    break
                
            else: #return NaN if all retry fail
                check_result = np.full(4,np.NaN)
            
        for j in range(check_result.shape[0]):
            equil_array[i,j] = check_result[j]
        
    return equil_array