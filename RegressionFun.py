#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    import xmltodict
    print("module 'xmltodict' is installed")
except ModuleNotFoundError:
  get_ipython().system('pip install xmltodict')

from sklearn.linear_model import LassoCV, RidgeCV, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
# import statsmodels.api as sm
from functools import reduce

import requests
import xmltodict
import datetime

# Web Call to Access and Download Data of a Single Station from AWDB Web Service (SOAP API)

def SOAP_Call(stationtriplets, elementCD, begindate, enddate):
  global xml, dict_of_xml, df
  # Create a dictionaries to store the data
  headers = {'Content-type': 'text/soap'}
  # current_dictionary = {}
  
  # Define Web Service URL
  URL = "https://wcc.sc.egov.usda.gov/awdbWebService/services?WSDL"

  # Define Parameters for SOAP Elements (getData:current and getCentralTendencyData:normals)
  SOAP_current = '''
  <?xml version="1.0" encoding="UTF-8"?>
  <SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:q0="http://www.wcc.nrcs.usda.gov/ns/awdbWebService" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <SOAP-ENV:Body>
      <q0:getData>
        <stationTriplets>STATIONTRIPLETS</stationTriplets>
        <elementCd>ELEMENTCD</elementCd>   
        <ordinal>1</ordinal>
        <duration>DAILY</duration>
        <getFlags>false</getFlags>
        <beginDate>BEGINDATE</beginDate>
        <endDate>ENDDATE</endDate>
        <alwaysReturnDailyFeb29>false</alwaysReturnDailyFeb29>   
      </q0:getData>
    </SOAP-ENV:Body>
  </SOAP-ENV:Envelope>

  '''.strip()
  #Read GetData documents - If <alwaysReturnDailyFeb29> is set to true, will set a null for every non leap year on the 29th,  
  #which breaks this request when selecting date ranges that include Feb 29.
  #Possible element codes: PREC, WTEQ (Water Equivalent/SWE)
  
  # Post SOAP Elements to AWDB Web Service and process results - getData
  SOAP_current = SOAP_current.replace("ELEMENTCD", elementCD)
  SOAP_current = SOAP_current.replace("STATIONTRIPLETS", stationtriplets)
  SOAP_current = SOAP_current.replace("BEGINDATE", begindate)
  SOAP_current = SOAP_current.replace("ENDDATE", enddate)

  #Send request to server and receive xml document
  xml = requests.post(URL, data=SOAP_current, headers=headers)

  #convert xml document to a dictionary, extract values putting them in a dataframe.  XML's aren't the easiest to parse and extract data from, so this is a nice work around.
  dict_of_xml = xmltodict.parse(xml.text)
  df = dict_of_xml['soap:Envelope']['soap:Body']['ns2:getDataResponse']['return']['values']

  #Null values are given as OrderedDictionaries with lots of text, while actual values are given as strings.  This converts all the OrderedDictionaries into actual null/none values, and converts all values that were given as strings into float numbers.
  df = pd.DataFrame(map(lambda i: float(i) if type(i) == str else None, df))

  #Since invidual dates aren't associated with the values in the xml document, have to create a range of dates bw the begindate and endate, which is then added to the dataframe.
  df['Date'] = pd.date_range(begindate,enddate,freq='d')
  df.columns = [f'{elementCD}','Date']
  df.set_index('Date', inplace=True)

  return df


# Function to download data from multiple stations at a time from AWDB Web Service
# Web service request will except multiple stations in a single request, so this is definitely not the best way to do this as it sends multiple requests combining each into a single pandas dataframe.  

def getData(stations, parameter_of_interest, begindate, enddate): 
  try:
    data_singleDF = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True, how='outer'), [SOAP_Call(stationtriplets=j,elementCD=parameter_of_interest,begindate=begindate,enddate=enddate) for j in stations])  
    data_singleDF.columns = [f'{j}' for j in stations]
    return data_singleDF
    
  except ValueError:
    print('ValueError occured. Data does not exist for the specific date of either (or both) the begindate or enddate.  Select a different date(s), or enter temporary value(s) into DMP')

  # except UnboundLocalError:
  #   print('Data does not exist for the specific date of either (or both) the begindate or enddate.  Select a different date(s), or enter temporary value(s) into DMP')

  except KeyError:
    print('KeyError occurred.')  
    print('Possible reasons for this error include:')
    print('\n')
    print('Stations are not in brackets. For example, \'401:OR:SNTL\' should instead be entered as [\'401:OR:SNTL\'].')
    print('Stations are incorrectly called or mispelled.  Check that the correct station ID, state, and network are correct.')
    print('Parameter was mispelled. For example, SWE was entered instead of WTEQ')
    print('Date is not in the correct DD/MM/YYYY format.')


#Function that assigns the regression model to be used and its parameters

def regressionModel(regression_model):
  if regression_model == 'Lasso':
    regr = LassoCV(alphas=(0.001,0.01,0.1,1,10,100,1000))
    return regr
  elif regression_model == 'Ridge':
    regr = RidgeCV(alphas=(0.001,0.01,0.1,1,10,100,1000))
    return regr
  elif regression_model == 'SVM': 
    regr = svm.SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    return regr
  elif regression_model == 'Huber':
    regr = HuberRegressor()
    return regr
  elif regression_model == 'GradientBoost':
    regr = GradientBoostingRegressor(random_state=0, n_estimators=100)
    return regr
  elif regression_model == 'AdaBoost':
    regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    return regr
  elif regression_model == 'Random Forest':
    regr = RandomForestRegressor(n_estimators=100)
    return regr
  else:
    print('Choose either Lasso, Ridge, SVM, Huber, GradientBoost, AdaBoost, or Random Forest')


class RegressionFun():

  '''
  stations: list of stationtriplets, with the response station in the first position followed by the predictor station(s).  Ex. ['817:WA:SNTL', '711:WA:SNTL', '975:WA:SNTL']
  parameter_of_interest: can choose 'WTEQ', 'SNWD', 'PREC', 'PRCP', 'TAVG', 'TOBS', and others (https://www.nrcs.usda.gov/wps/portal/wcc/home/dataAccessHelp/webService/webServiceReference/!ut/p/z1/jc9NDoIwEAXgs3CCPqCWuqxNBFIiqdCAsyFdkSaKLozn1xg2LmyY3STfmx9GbGS0-FeY_TPcF3_99BcSk5Sapy1HU576HFactTtWJkMKNnxBnUlVyQ6m1JxD7XXRN10BcwCjLXn8KbUxHwEUHz8w-l0BWWew2hre7pBDpyuIvRgFvVhB5IrHzbkRoZ5VkrwBnC-XyQ!!/?1dmy&current=true&urile=wcm%3apath%3a%2Fwcc%2Bcontent%2Fhome%2Fdata%2Baccess%2Bhelp%2Fweb%2Bservice%2Fweb%2Bservice%2Breference#elementCodes).
  '''

  def __init__(self, stations, parameter_of_interest): #, stations, parameter_of_interest):

    self.stations = stations
    self.parameter_of_interest = parameter_of_interest
    
  def check_model(self, begindate, enddate, regression_model, test_size):
    
    '''
    Function checks model fit on train and test sets.  Use to check which stations result in the best fitting model.  Once the best model is found, it can be used in the make_predictions function to predict null values.
    
    begindate: non-null date in 'mm/dd/yyyy' format
    enddate: non-null date in 'mm/dd/yyyy' format
    regression_model: can be 'Ridge', 'Lasso', 'Huber', 'SVM', 'Random Forest', 'AdaBoost', 'GradientBoost'
    test_size: value between 0 and 1 that defines the percentage of the data to be used as the test_size
    '''

    #Download Data from AWDB Web Service
    data = getData(self.stations, self.parameter_of_interest, begindate, enddate).dropna()

    #Define Targets and Features (e.g. Response and Predictor Variables)
    target = data.iloc[:,0]
    features = data.iloc[:,1:]

    #Split into training and test sets
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=test_size, shuffle=False)
  
    #Choose Regression Model:
    regr = regressionModel(regression_model)
  
    #Fit model on training set
    regr.fit(features_train, target_train)

    self.regr = regr

    #Run predictions on training features and test features
    target_train_pred = regr.predict(features_train)
    target_test_pred = regr.predict(features_test)

    #Print Root Mean Square Error for training and test sets:
    print('RMSE for training set', mean_squared_error(target_train, target_train_pred))
    print('RMSE for test set', mean_squared_error(target_test, target_test_pred))

    #Predictions plot

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y = target_train,
        x = target_train.index,
        mode = 'lines',
        name = 'Training Data'
    ))

    fig.add_trace(go.Scatter(
        y = target_train_pred,
        x = target_train.index,
        mode = 'lines',
        name = 'Model Predictions on Training Data'
    ))

    fig.add_trace(go.Scatter(
        y = target_test,
        x = target_test.index,
        mode = 'lines',
        name = 'Test Data'
    ))

    fig.add_trace(go.Scatter(
        y = target_test_pred,
        x = target_test.index,
        mode = 'lines',
        name = 'Model Predictions on Test Data'
    ))

    fig.update_yaxes(title_text = f"{self.parameter_of_interest}")
    fig.update_xaxes(title_text = 'Date') 

    fig.update_layout(
        showlegend=True,
        height=450,
        width=1200 
    )
    fig.show()

    # # # # # Regression Plot # # # # #

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
      x=features_train.iloc[:,0],
      y=target_train,
      mode='markers',
      hovertext = features_train.index,
      name =  'Response vs. Predictors'
    ))

    fig2.add_trace(go.Scatter(
      x=features_train.iloc[:,0],
      y=pd.DataFrame(target_train_pred.tolist()).iloc[:,0],
      mode='lines',
      hovertext = features_train.index,
      name = 'Model Fit'  
    ))
    
    fig2.update_yaxes(title_text = f'{self.stations[1:]} {self.parameter_of_interest}')
    fig2.update_xaxes(title_text = f'{self.stations[0]} {self.parameter_of_interest}')
    fig2.update_layout(
        showlegend=True,
        height=450,
        width=1200 
    )
    fig2.show()

    return regr
      
      
  def make_predictions(self, predict_begindate, predict_enddate):

    #Download Data from AWDB Web Service
    predict_data  = getData(self.stations, self.parameter_of_interest, predict_begindate, predict_enddate)

    predict_target = predict_data.iloc[:,0].fillna(0)
    predict_features = predict_data.iloc[:,1:].fillna(0)

    #Run predictions
    predictions = self.regr.predict(predict_features)
    
    #Plot predictions 
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y = predict_target,
        x = predict_target.index,
        mode = 'lines',
        name = f'{self.stations[0]} {self.parameter_of_interest}'
    )) 

    fig.add_trace(go.Scatter(
        y = predictions,
        x = predict_features.index,
        mode = 'lines',
        name = 'Model Predictions'
    )) 

    fig.update_layout(
      showlegend=True,
      height=650,
      width=1400,
      title={
          'text': "Model Predictions",
          'xanchor': 'center',
          'yanchor': 'top',
          'y':0.9,
          'x':0.4},
      xaxis_title = "Date",
      yaxis_title = f"{self.parameter_of_interest} (in)"
    )

    fig.show()  


# In[ ]:




