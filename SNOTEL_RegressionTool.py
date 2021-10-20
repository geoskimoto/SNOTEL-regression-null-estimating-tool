#!/usr/bin/env python
# coding: utf-8

# In[19]:


try:
    import xmltodict
    print("module 'xmltodict' is installed")
except ModuleNotFoundError:
  get_ipython().system('pip install xmltodict')

from sklearn.linear_model import LassoCV, RidgeCV, HuberRegressor, LinearRegression
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

def getData(stationparameter, begindate, enddate):   
 
    try:
        data = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True, how='outer'), 
              [SOAP_Call(stationtriplets = i[0], elementCD = i[1], begindate=begindate, enddate=enddate) for i in stationparameter])  
        data.columns = [f'{j}' for j in stationparameter]
        return data

    except ValueError:
        print('ValueError occured. Data does not exist for the specific date of either (or both) the begindate or enddate.  Select a different date(s), or enter temporary value(s) into DMP')

    except KeyError:
        print('KeyError occurred.')  


#Function that assigns the regression model to be used and its parameters

def regressionModel(regression_model):
    if regression_model == 'Lasso':
        regr = LassoCV(alphas=(0.001,0.01,0.1,1,10,100,1000))
        return regr
    elif regression_model == 'Ridge':
        regr = RidgeCV(alphas=(0.001,0.01,0.1,1,10,100,1000))
        return regr
    elif regression_model == 'Linear':
        regr = LinearRegression()
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
        print('Choose either Lasso, Ridge, Huber, SVM, GradientBoost, AdaBoost, or Random Forest')        
        
        
class RegressionFun():
    

    def __init__(self, stationparameterpairs, begindate, enddate): #, stations, parameter_of_interest):

        self.stationparameterpairs = stationparameterpairs
        self.stations = [i[0] for i in stationparameterpairs]
        self.parameters = [i[1] for i in stationparameterpairs]

        #Download Data from AWDB Web Service
        self.begindate = begindate
        self.enddate = enddate
        self.data = getData(self.stationparameterpairs, begindate, enddate).dropna()


    def train_model(self, regression_model, test_size):

        '''
        Function checks model fit on train and test sets.  Use to check which stations result in the best fitting model.  Once the best model is found, it can be used in the make_predictions function to predict null values.

        begindate: non-null date in 'mm/dd/yyyy' format
        enddate: non-null date in 'mm/dd/yyyy' format
        regression_model: can be 'Ridge', 'Lasso', 'Huber', 'SVM', 'Random Forest', 'AdaBoost', 'GradientBoost'
        test_size: value between 0 and 1 that defines the percentage of the data to be used as the test_size
        '''

        #Define Targets and Features (e.g. Response and Predictor Variables)
        target = self.data.iloc[:,0]
        self.target = target
        features = self.data.iloc[:,1:]
        self.features = features

        #Split into training and test sets
        features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=test_size, shuffle=False)

        self.features_train = features_train
        self.features_test = features_test
        self.target_train = target_train
        self.target_test = target_test

        #Choose Regression Model:
        regr = regressionModel(regression_model)

        #Fit model on training set
        regr.fit(features_train, target_train)

        self.regr = regr

        #Run predictions on training features and test features
        target_train_pred = regr.predict(features_train)
        target_test_pred = regr.predict(features_test)

        #Print Root Mean Square Error for training and test sets:
        RMSE_train = ('RMSE for training set:' + ' ' + str(f'{mean_squared_error(target_train, target_train_pred): .5}'))
        RMSE_test = ('RMSE for test set:' + ' ' + str(f'{mean_squared_error(target_test, target_test_pred): .5}'))

        self.RMSE_train = RMSE_train
        self.RMSE_test = RMSE_test

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
        
        fig.add_annotation(
            xref='paper',
            x=0.3,
            yref='paper',
            y=.8,
            text = RMSE_train,
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="black"
                ),
            align="left",
            bordercolor="black",
            borderwidth=2,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
#             textposition="top right",
        )
        
        fig.add_annotation(
            xref='paper',
            x=0.3,
            yref='paper',
            y=.7,
            text = RMSE_test,
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="black"
                ),
            align="left",
            bordercolor="black",
            borderwidth=2,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
               
#             textposition="top center",
        )
        
        fig.update_yaxes(title_text = f"{self.parameters[0]}")
        fig.update_xaxes(title_text = 'Date') 

        fig.update_layout(
            showlegend=True,
            height=500,
            width=950,
            title={
              'text': "Model Predictions on Training and Test Data",
              'xanchor': 'center',
              'yanchor': 'top',
              'y':0.9,
              'x':0.4
            },
        )
#         fig.show()
        self.traintest_fig = fig

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

        fig2.update_xaxes(title_text = f'{self.stations[0]} {self.parameters[0]}')
        fig2.update_yaxes(title_text = f'{self.stationparameterpairs[1:]}')
        fig2.update_layout(
            showlegend=True,
            height=500,
            width=950,
            title={
              'text': "Regression Model",
              'xanchor': 'center',
              'yanchor': 'top',
              'y':0.9,
              'x':0.4
            },
        )
#         fig2.show()
        self.modelfit_fig = fig2


    def make_predictions(self, predict_begindate, predict_enddate):

        #Download Data from AWDB Web Service
        predict_data  = getData(self.stationparameterpairs, predict_begindate, predict_enddate)

        predict_target = predict_data.iloc[:,0].fillna(0)
        predict_features = predict_data.iloc[:,1:].fillna(0)

        #Run predictions
        predictions = self.regr.predict(predict_features)

        #Plot predictions 
        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            y = predict_target,
            x = predict_target.index,
            mode = 'lines',
            name = f'{self.stations[0]} {self.parameters[0]}'
        )) 

        fig3.add_trace(go.Scatter(
            y = predictions,
            x = predict_features.index,
            mode = 'lines',
            name = 'Model Predictions'
        )) 

        fig3.update_layout(
          showlegend=True,
          height=500,
          width=950,
          title={
              'text': "Model Predictions",
              'xanchor': 'center',
              'yanchor': 'top',
              'y':0.9,
              'x':0.4
          },
          xaxis_title = "Date",
          yaxis_title = f"{self.stationparameterpairs[0]}"
        )
        
#         fig3.show()    
        self.predictions_fig = fig3

# In[25]:


# apple = RegressionFun([('302:OR:SNTL','WTEQ'),('302:OR:SNTL','SNWD'),('302:OR:SNTL','TAVG'),('653:OR:SNTL','WTEQ')], '01/01/2018','02/01/2020')


# In[26]:


# apple.train_model('SVM', 0.3)


# In[ ]:


# b


# In[ ]:


# a


# In[ ]:


# apple.model_fit_fig


# In[ ]:


# apple.train_test_fig


# In[ ]:


# apple.make_predictions('02/01/2020','07/01/2020')


# In[ ]:




