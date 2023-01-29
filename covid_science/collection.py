"""This module provides functions that helps data collection."""

import os
import datetime,requests,json
import time,re
import pandas as pd
from IPython.display import clear_output
import ctypes
import __main__
from seleniumwire import webdriver
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.utils import ChromeType

#------------------------------------------------------------------
def store_dataframe(input_df,folder_path,horizontal_sery=False,auto_process=False,
                    encoding=None,subset=None,conflict_ignore=False,**kwargs):
    """
    This function is used to store time sery dataframe into the directory.
    File name will be formatted as '%YYYY-%MM-%DD.csv' matching the invoked date.
    The function applied for time sery data, normal dataframe will cause error 
    and not saved. The limitation of this function is that it only compare 
    the dataframe with latest file in the folder with correct name format
    and only creates data for today,and suggestion to save the input dataframe 
    into the folder or not.
    
    Warning
    ----------
    This function can't be used for mutiple index DataFrame. so it only work 
    for the first time if the folder is empty.
    File name is based on process date, not latest date of data.
    New file will override the file created within the same day.
    
    Parameters
    ----------
    input_df: pd.DatFrame
        Dataframe that need to be stored.
        
    folder_path: str
        Absolute/Relative path to the stored folder
        
    horizontal_sery: bool 
        The purpose is to make sure the dataframe is formatted correctly 
        for comparision (vertical or horizone time sery)
    
    auto_process: bool 
        If True, new data will override previous version if changes exist, and 
        new file will be created even though no there might be no changes.
    
    encoding: str 
        Reading/saving encoding of csv file.
        
    subset: list of str 
        Columns of input dataframe that use to check for duplication if necessary.
        
    conflict_ignore: bool
        Save DataFrame even if it conflicts with current database. This is a 
        shortcut for saving data without resolving errors.
        Errors will still be raised but ignored.
        
    **kwargs: 
        Extra input from pandas.DataFrame.to_csv() function
    
    Returns
    -------
    None
        CSV file will be stored in the destination folder with datetime.date.today 
        as the file name.
    """
    assert isinstance(input_df,pd.DataFrame),"Type should be DataFrame."
    assert type(folder_path)==str,"Type should be string."
    
    try:
        if subset==None: #check for duplication since some day might be repeated in data
            pass
        else:
            if horizontal_sery==True: #this doesn't happen with horizontal time sery, since day is column name
                pass
            else:
                if input_df.shape == input_df.drop_duplicates(subset).shape:
                    pass
                else:
                    raise Exception("Duplication of data found, please process data before continuing.")
        if os.path.exists(str(folder_path)):
            pass
        else:
            os.makedirs(str(folder_path))
            print("New folder created.")
        file_list = [
            file for file in os.listdir(folder_path) if (file.endswith('.csv') and 
            re.match(r"[0-9]{1,4}-(1[0-2]|0[1-9])-(0[1-9]|[1-2][0-9]|3[0-1])\.csv",file))
            ]
        datetime_list = [
            datetime.datetime.strptime(re.sub(r".csv$","",file),"%Y-%m-%d").date() 
            for file in os.listdir(folder_path) if (file.endswith('.csv') 
            and re.match(r"[0-9]{1,4}-(1[0-2]|0[1-9])-(0[1-9]|[1-2][0-9]|3[0-1])\.csv",file))
            ]
        
        if file_list ==[] and datetime_list==[]: #if folder doesn't any previous data, just save the new csv file
            input_df.to_csv(folder_path + "\\" + 
                            datetime.date.today().strftime("%Y-%m-%d") +".csv",
                            encoding=encoding,index=False,**kwargs)
            print("New file added.")
        elif max(datetime_list)>datetime.date.today():
            compare_file_index = datetime_list.index(max(datetime_list))
            raise Warning(f"Latest file is '{file_list[compare_file_index]}'"+
                          " and the date is in the future. Please check the data again.")
        else: #check the new dataframe
            if datetime.date.today() in datetime_list: #check if data for today is collected
                compare_file_index = datetime_list.index(datetime.date.today())
                today_collected = True
            else:
                compare_file_index = datetime_list.index(max(datetime_list))
                today_collected = False
            if horizontal_sery: #check if data is time sery
                df = input_df.transpose().reset_index()
                df2 = pd.read_csv(folder_path + "\\" + file_list[compare_file_index],
                                  encoding=encoding)
                compare_df = df2.transpose().reset_index()
                #float value actually cause error to drop_duplicates, so round up to fix it
                merge_df=pd.concat([df,compare_df]).round(3).drop_duplicates(keep=False) 
            else:
                df = input_df
                df2 = pd.read_csv(folder_path + "\\" + file_list[compare_file_index],
                                  encoding=encoding)
                compare_df = df2
                merge_df=pd.concat([df,compare_df]).round(3).drop_duplicates(keep=False)
            
            #compare difference between new dataframe and newest dataframe in folder
            if (merge_df.size>(compare_df.size+df.size) or 
                df.shape[1]!=compare_df.shape[1] or 
                df.shape[0]<compare_df.shape[0]):
                if horizontal_sery:
                    print("Time sery data")
                print("Input DataFrame shape: ",input_df.shape)
                print("Latest DataFrame shape: ",df2.shape)
                print("Drop duplicated Merge DataFrame shape: ",merge_df.shape)
                raise Exception("Input dataframe has some errors/format problems. File will not be saved.")
            elif (merge_df.shape[0]==0) and today_collected:
                print("Data are up to date. No file created.")
            elif today_collected:
                if auto_process:
                    answer = 6
                else:
                    answer = message_box(
                        "Data storage setting",
                        "There seems to be some changes comparing to the the "+
                        "previous version of data collected today. "+
                        "Do you want to override the file?",
                        4+32+4096)
                if answer == 6: #default for yes
                    input_df.to_csv(folder_path + "\\" + 
                                    datetime.date.today().strftime("%Y-%m-%d") +".csv",
                                    encoding=encoding,index=False,**kwargs)
                    print("Previous file version replaced.")
                else:
                    print("Previous file version for today is not updated.")
            elif (merge_df.shape[0]==0) and today_collected==False:
                if auto_process:
                    answer = 6
                else:
                    answer = message_box(
                        "Data storage setting",
                        "There seems to be no new information in the input DataFrame. "+
                        "Do you want to save the input DataFrame as new file?",
                        4+32+4096)
                if answer == 6: #default for yes
                    input_df.to_csv(folder_path + "\\" + 
                                    datetime.date.today().strftime("%Y-%m-%d") +".csv",
                                    encoding=encoding,index=False,**kwargs)
                    print("New file added.")
                else:
                    print("Data are up to date. No file created.")
            else:
                input_df.to_csv(folder_path + "\\" + 
                                datetime.date.today().strftime("%Y-%m-%d") +".csv",
                                encoding=encoding,index=False,**kwargs)
                print("New file added.")
    except Exception as error:
        if conflict_ignore==True:
            input_df.to_csv(folder_path + "\\" + 
                            datetime.date.today().strftime("%Y-%m-%d") +".csv",
                            encoding=encoding,index=False,**kwargs)
            print("DATA CONFLICTION!!! New file still created. Please process with caution.")
            raise error
        raise error
        
#------------------------------------------------------------------

def collect_api_response(website_link,
                         api,
                         search_scope=[],
                         driver_path="",
                         wait_time=15,
                         disable_encode=False,
                         encoding='utf-8'):
    """
    Collect API response from the desired website with chosen waiting time.
    
    Parameters
    ----------
    website_link: str
        Website link for data collection.
        
    api: str 
        API link(or pattern).
    
    search_scope: list of str 
        This accepts a list of patterns that will match the URLs to be captured.
    
    driver_path: str 
        PATH of 'chromedriver.exe'. There are two input options.
        1: just put in the desired path (relative/absolute), the module will 
        automatically download the chromedriver.exe and add '.wdm' folder to 
        the path which stores chromedriver.exe .
        2:Input the path to 'chromedriver.exe', the function will use this
        file directly to invoke the browser.
    
    wait_time: int 
        Desired waiting time before getting web traffic data.
    
    disable_encode: bool
        Ask the server not to compress the response.
    
    encoding: str
        The encoding with which to decode the bytes.
    
    Returns
    -------
    dict
        API response in JSON format.
    """
    
    assert type(website_link)==str,"Type should be string."
    assert type(search_scope)==list,"Type should be list."
    assert all(isinstance(x,str) for x in search_scope),"All scopes should be string."  
    assert type(api)==str,"Type should be string."
    assert type(driver_path)==str,"Type should be string."
    assert type(wait_time)==int,"Type should be integer."
    
    try:
        options = {
        'disable_encoding': disable_encode
        }
        chrome_option = webdriver.ChromeOptions()
        chrome_option.add_argument("--headless")
        if re.match(r'.*(chromedriver.exe)$',driver_path):
            path = driver_path
        else:
            path = ChromeDriverManager(path=driver_path,
                                       chrome_type=ChromeType.CHROMIUM).install()
        driver = webdriver.Chrome(path,
                                  options=chrome_option,
                                  seleniumwire_options=options)
        if search_scope==[]:
            pass
        else:
            driver.scopes=['.*'+re.escape(i)+'.*' for i in search_scope]
        driver.get(website_link)
        driver.wait_for_request(re.escape(api),timeout=wait_time)
        #driver.close() doesn't seem to work well with repeat
        response = None
        for request in driver.requests:
            if api in request.url:
                r = request.response
                response = json.loads(r.body.decode(encoding))
                break
        driver.quit()
        return response
    except TimeoutException:
        return None
    except Exception as error:
        raise error
        
#------------------------------------------------------
def refresh_web_traffic(input_object,
                        track_variable,
                        website_link,
                        api,
                        default_time=40,
                        repeat=3,
                        interval=5,
                        **kwargs):
    """
    This function is used to refresh the webpage and collect API response 
    again if possible.
    
    Parameters
    ----------
        
    input_object: dict 
        JSON object that need to be collected.
        
    track_variable: str 
        String name of a [global] variable uses to track collect time, 
        e.g. "wait_time".
        
    website_link: str
        Website link for data collection.
        
    api: str
        API link(or pattern).
        
    default_timer: int
        Cancel collection  after a set period.
        
    repeat: int
        Maximum number of repeat attempt.
        
    interval: int
        Ddded seconds between each attempt.
        
    **kwargs: 
        Optional inputs for covid_science.collection.collect_api_response 
        function. 
    
    Returns
    -------
    dict
        API response in JSON format if possible.
    """
    assert type(input_object)==dict or input_object is None,"input_object isn't in JSON format"
    assert type(track_variable)==str,"Require string name of the targeted variable."
    assert type(website_link)==str,"Type should be string."
    assert type(api)==str,"Type should be string."
    assert type(default_time)==int,"Type should be integer."
    assert type(repeat)==int,"Type should be integer."
    assert type(interval)==int,"Type should be integer."
    if input_object==None:
        clear_output(wait=False)
        for i in range(0,repeat):
            time.sleep(1)
            clear_output(wait=False)
            if __main__.__dict__[track_variable] >= (default_time):  
                print("API is no longer valid or website loading is too slow.")
                return None
            else:
                __main__.__dict__[track_variable]+=interval #this is use to modify the variable in the active module
                print("Data collection attempt No." + str(i+1) + ", please wait.")
                json_object = collect_api_response(website_link,
                                                   api,
                                                   wait_time=__main__.__dict__[track_variable],
                                                   **kwargs)
                if json_object != None:
                    print("Data collected after refreshing",(i+1),"time(s).")
                    return json_object
                else:
                    pass
        print("API is no longer valid or website loading is too slow.")
        return None
    else:
        return input_object
    
#------------------------------------------------------

def message_box(title, text, style):
    """
    This function is used to make pop-up message.
    
    Parameters
    ----------
    title: str
        pop-up title.
    
    text: str
        Body text.
    style: int
        MessageBox style, refer to MessageBox function from microsoft for 
        more inromation (note:convert hex to decimal before input)
    
    Guideline
    ----------
    The sum of each parameter is the input to style
    
    _Field:
    0 - The message box contains an OK button.
    1 - The message box contains OK and Cancel buttons.
    2 - The message box contains Abort, Retry, and Ignore buttons.
    3 - The message box contains Yes, No, and Cancel buttons.
    4 - The message box contains Yes and No buttons.
    5 - The message box contains Retry and Cancel buttons.
    6 - Specifies that the message box contains Cancel, Try Again, and Continue 
        buttons.
    
    _Icon:
    16 - A stop-sign icon appears in the message box.
    32 - A question-mark icon appears in the message box.
    48 - An exclamation-point icon appears in the message box.
    64 - An icon consisting of a lowercase letter i in a circle appears in the 
        message box.
    
    _Style:
    4096 - The message box is created with the WS_EX_TOPMOST window style.
    
    Output:
    Return ID code based on the chosen option.
    
    1 - Ok
    2 - Cancel
    3 - Abort
    4 - Retry
    5 - Ignore
    6 - Yes
    7 - No
    10- Try again
    11- Continue
    
    Returns
    -------
    None
        Notification message box pop up.
    """
    
    assert type(title)==str,"Type should be string"
    assert type(text)==str,"Type should be string"
    assert type(style)==int,"Type should be integer"
    
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

#------------------------------------------------------

class browser_request:
    def __init__(self,wait_time=20,attempt=3,interval=5):
        """
        A class build to get web response if normal get request from website
        doesn't work

        Parameters
        ----------
        wait_time : int, optional
            Maximum wait time until trying to get a response.
            
        attempt : int, optional
            Number of attempts to get web response.
            
        interval : int, optional
            Acummulated time added to 'wait_time' after each attemp.
        """
        
        assert type(wait_time)==int,"'wait_time' should be integer."
        assert wait_time>=1,"'wait_time' should be positive."
        assert type(attempt)==int,"'attempt' should be integer."
        assert attempt>=1,"'attempt' should be positive."
        assert type(interval)==int,"'interval' should be integer."
        assert interval>=1,"'interval' should be positive."
        
        self.wait_time = wait_time
        self.attempt = attempt
        self.interval = interval
    def get_web_response(self,website_link,api,process_time=False,**kwargs):
        """
        Get the website response from the given URL and their API

        Parameters
        ----------
        website_link : str
            The website URL.
            
        api : str
            THe website API.
        
        process_time: bool
            Print the processing time if True.
        
        **kwargs : dict, optional
            Extra keyword arguments to add to 
            covid_science.collection.collect_api_response function.

        Returns
        -------
        json_object : dict
            The web response in json format.

        """
        
        assert type(website_link)==str,"'website_link' should be string."
        assert type(api)==str,"'api' should be string."
        
        start = time.perf_counter()
        for i in range(self.attempt):
            if process_time:
                print(f"Attempt to collect data: {i+1} time(s)",end="\r")
            t = self.wait_time + i*self.interval
            json_object = collect_api_response(website_link,
                                               api,
                                               wait_time=t,
                                               **kwargs)
            if json_object is not None:
                break
        
        if json_object is None:
            end = time.perf_counter()
            if process_time:
                print(f"Couldn't collect data after {self.attempt} attempt(s),"
                      +" API is no longer valid or website loading is too slow.")
                print(f"Process time taken: {round(end-start,2)}s")
                print("----------")
            return None
        else:
            end = time.perf_counter()
            if process_time:
                print(f"Data collected after {i+1} attempt(s).")
                print(f"Process time taken: {round(end-start,2)}s")
                print("----------")
            return json_object
        
#------------------------------------------------------

def get_github_time_sery_url(api,**kwargs):
    """
    Use to catch the latest data from a github directory where csv files are 
    stored in "YYYY-MM-DD.csv" format.

    Parameters
    ----------
    api : str
        Github API use to get directory content.
    **kwargs : dict
        Extra parameters to requests.get.

    Returns
    -------
    latest_file : str
        Return the file URL.
    """
    
    r = requests.get(api,**kwargs)
    response = json.loads(r.content)
    df = pd.DataFrame(response)
    df['date']=pd.to_datetime(df['name'].str.replace(".csv","",regex=False))
    latest_file = df.loc[df['date']==df['date'].max(),'download_url'].values[0]
    return latest_file