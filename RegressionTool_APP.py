#!/usr/bin/env python
# coding: utf-8

# Author: Nicholas Steele
# Snow Survey Hydrologist
# USDA - Natural Resources Conservation Service
# 1201 NE Lloyd Blvd, Suite #900
# Portland, OR  97232
# Email: nick.steele@usda.gov
# Cell: 503-819-5880
# 

# In[16]:




#------------------Import libraries----------------------------------------------


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
# from functools import reduce

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from datetime import date
import SNOTEL_RegressionTool as RegressionTool

import requests
import xmltodict

#-----------------variables to be used in different places within app-------------------

heading={'color':'black', 'font-size':50,'text-Align':'center','font-weight': 'bold'}
subheading={'color':'black', 'font-weight': 'bold', 'text-Align':'left'}

Stations = pd.read_csv('SNTL Triplets.csv')
# Stations = pd.read_excel('StationTriplets.xlsx')
station_names = Stations.loc[:,'Extended Name'].tolist()
triplets = Stations.loc[:, 'Station Triplet'].tolist()

options = []
for i in range(len(Stations.index)):
  options.append({'label': station_names[i], 'value': triplets[i]})
           
parameter_options =  [{'label': 'Snow Water Equivalent', 'value': 'WTEQ'},
                      {'label': 'Accumulative Precipitation', 'value': 'PREC'},
                      {'label': 'Precipitation', 'value': 'PRCP'},
                      {'label': 'Snow Depth', 'value': 'SNWD'},
                      {'label': 'Average Temperature', 'value': 'TAVG'},
                      {'label': 'Observed Temperature', 'value': 'TOBS'},
                      {'label': 'Max Temperature', 'value': 'TMAX'},
                      {'label': 'Min Temperature', 'value': 'TMIN'}]

model_options = [{'label':'Linear Regressor', 'value': 'Linear'},
                {'label':'Ridge Regressor', 'value': 'Ridge'},
                {'label':'Lasso Regressor', 'value': 'Lasso'},
                {'label':'Huber Regressor', 'value': 'Huber'},
                {'label':'Support Vector Machines Regressor', 'value': 'SVM'},
                {'label':'Random Forest Regressor', 'value': 'Random Forest'},
                {'label':'AdaBoost Regressor', 'value': 'AdaBoost'},
                {'label':'GradientBoost Regressor', 'value': 'GradientBoost'}]

#--------------------Initialize the app class----------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
#https://stackoverflow.com/questions/50844844/python-dash-custom-css


#--------------------Layout Design ---------------------------------------------

controls = dbc.Card(
    [
        dbc.Form(
            [
                        
                        html.Br(),
                        html.Div(children = [
                            html.H5(['Select the station and parameter to be used as the response:'], style=subheading),
                            dcc.Dropdown(id = 'response-station', 
                                         options = options, 
                                         multi=False,
                                         value = '301:CA:SNTL',
                    #                      style = station_dropdown_style
                                        ),
                            dcc.Dropdown(id = 'response-parameter', 
                                        options = parameter_options,
                                        multi=False,
                                        value = "WTEQ",
                    #                     style = parameter_dropdown_style
                                        )]),    
                        html.Br(),
                        html.Div(children = [
                            html.H5(['Select station parameter pairs to be used as predictors:'], style=subheading),
                            dcc.Dropdown(id = 'predictor-station1', 
                                         options = options, 
                                         multi = False,
                                         value = '391:CA:SNTL',
                    #                      style = station_dropdown_style,
                                        ),
                            dcc.Dropdown(id = 'predictor-parameter1', 
                                        options = parameter_options,
                                        multi = False,
                                        value="WTEQ",
                    #                     style = parameter_dropdown_style,
                                        ),  
                            html.Br(),
                            dcc.Dropdown(id = 'predictor-station2', 
                                         options = options, 
                                         multi = False,
                                         value = None,
                    #                      style = station_dropdown_style
                                        ),
                            dcc.Dropdown(id = 'predictor-parameter2', 
                                        options = parameter_options,
                                        multi = False,
                                        value = None,
                    #                     style = parameter_dropdown_style,
                                        ),

                            html.Br(),
                            dcc.Dropdown(id = 'predictor-station3', 
                                         options = options, 
                                         multi = False,
                                         value = None,
                    #                      style = station_dropdown_style
                                        ),
                            dcc.Dropdown(id = 'predictor-parameter3', 
                                        options = parameter_options,
                                        multi = False,
                                        value=None,
                    #                     style = 'station_dropdown_style'
                                        ),
                            html.Br(),
                            dcc.Dropdown(id = 'predictor-station4', 
                                         options = options, 
                                         multi = False,
                                         value = None,
                    #                      style = station_dropdown_style
                                        ),
                            dcc.Dropdown(id = 'predictor-parameter4', 
                                        options = parameter_options,
                                        multi = False,
                                        value=None,
                    #                     style = 'station_dropdown_style'
                                        ),

                        ]),
                        html.Br(),
                        html.Div(children = [
                        
                        html.H5(['Select date range to train the regression model: '], style=subheading),        

                        dcc.DatePickerSingle(
                            id='startdate_picker',
                            min_date_allowed=date(1970, 10, 1),
                    #         max_date_allowed=date(2017, 9, 19),
                    #         initial_visible_month=date(2017, 8, 5),
                            date = date(2020,11,1)
                    #         date = '09/01/2021'
                        ),

                        dcc.DatePickerSingle(
                            id='enddate_picker',
                            min_date_allowed=date(1970, 10, 1),
                    #         max_date_allowed=date(2017, 9, 19),
                    #         initial_visible_month=date(2017, 8, 5),
                            date = date(2021,2,1) #'09/20/2021',
                        ),

                        html.Br(),
                        html.Br(),
                        html.H5(['Select Regression Model to be used:']),
                        dcc.Dropdown(id = 'model_selection',
                                    options = model_options,
                                    value = 'Ridge'
                        ),    


                        html.Br(),
                        html.Button(id='submit-button-training', 
                                   children = 'Train Model and Run Predictions',
                    #                    style={'fontsize:24'}
                            )
                        ]),

                        html.Br(),
                        html.Br(),
                        html.Div(children = [

                            html.H5(['Select date range to run predictions after strong fitting model has been found:'], style=subheading),
                            dcc.DatePickerSingle(id='predict_startdate_picker',
                            date = date(2021,2,1)
                            ),
                            dcc.DatePickerSingle(id='predict_enddate_picker',
                            date = date(2021,3,1)
                            ),

                    #             html.Button(id='submit-button-predictions', 
                    # #                 value={},
                    #                children = 'Run Predictions',
                    # #                    style={'fontsize:24'}
                    #             ),
                         ]),
                
                        html.Br(),
                        html.Br(),
                        html.Details(
                            children=[
                                html.Summary('Useful Tibits'),
                                html.P(children=[
                                    'Tool can be used to estimate missing or bad data for any sensor on any SNOTEL or SNOLITE station in AWDB.',
                                    html.Br(),
                                    html.Br(),
                                    'If you are having issues with figures updating, likely reasons include:',
                                    html.Br(),
                                    '1. Data does not exist for the date range you are trying to train the model - Use Report Generator to determine the length of record for the interested stations.',
                                    html.Br(),
                                    '2. Similar to one, the specific four dates selected above can NOT be null in AWDB.  If estimating current data and no end date is available, go into DMP and add a temporary value for the end date. Change this temporary value afterwards using the regression model that you create.',
                                    html.Br(),
                                    html.Br(),
                                    'When evaluating models, pay attention to the Root Mean Square Error (RMSE) between the training and test data sets.',
                                    html.Br(),
                                    'As a general rule you want to minimize the error of both, but they should be similar in value.',
                                    html.Br(),
                                    'RMSE of test > RMSE of train => OVER FITTING of the data.',
                                    html.Br(),
                                    'RMSE of test < RMSE of train => UNDER FITTING of the data.',
                                    html.Br(),
                                    html.Br(),
                                    'When choosing regression model to make real world estimates, stick to linear models (linear, ridge, lasso, and huber).  These models are easy to explain and defend.  Additionally, the more advanced models have major drawbacks (primarily overfitting) that require either limiting the dataset or fine tuning parameters that are not available in this app.',
                                ])
                        

                                ],
#                             rows=4,
#                             cols=55
#                                              )],
                            title = 'Tool Description and Useful Info:',
                            id = 'Tool Description and Useful Info:',
                            key = 'Tool Description and Useful Info:',
                            style = subheading
                        )
            ])])


figures = dbc.Card(
    html.Div(children = [
        dbc.Row(
            dcc.Graph(id='training vs test plot', 
                      figure={}, 
#                                                       responsive=True,
#                                                       style={'width': '80%'}#,'padding-left':'2px','float':'right'}
             ), 
        ), #style={'width': '80%','padding-left':'2px','float':'right'}
        dbc.Row(
            dcc.Graph(id='regression model plot', 
                      figure={}, 
#                                                       responsive=True,
#                                                       style={'width': '80%'}#,'padding-left':'2px','float':'right'}
             ), 
        ),
        dbc.Row(
            dcc.Graph(id='predictions_figure', 
                      figure={}, 
    #                                                       responsive=True,
    #                                                       style={'width': '80%'}#,'padding-left':'2px','float':'right'}
             ), 
        ),
    ]))
             
              

                
                
app.layout = dbc.Container([
                        html.Div(children = [
                            dbc.Row([
                                html.Br(),
                                html.H5('SNOTEL Regression Tool', style = heading),
                                html.H4('Estimate missing or bad data using Regression', style = subheading)
                            ],  justify="center", align="center"),
                            dbc.Row(
                            [
                                    dbc.Col(controls, width=3),
                                    dbc.Col(figures),
                            ])
                        ])
                                
            ], fluid=True)
                
                     

#----------------All-in-one callback-----------------


@app.callback(
    [Output('training vs test plot', 'figure'),
    Output('regression model plot','figure'),
    Output('predictions_figure','figure')
    ],

#     [Input('submit-button-predictions', component_property='n_clicks')],
    [Input('submit-button-training', component_property='n_clicks')],
    
    [State('response-station','value'),
    State('response-parameter','value'),
    State('predictor-station1','value'),
    State('predictor-parameter1','value'),
    State('predictor-station2','value'),
    State('predictor-parameter2','value'),
    State('predictor-station3','value'),
    State('predictor-parameter3','value'),
    State('predictor-station4','value'),
    State('predictor-parameter4','value'),
    State('startdate_picker', 'date'),
    State('enddate_picker', 'date'),
    State('model_selection', 'value'),
    State('predict_startdate_picker', 'date'),
    State('predict_enddate_picker', 'date')
    ])


    
def predictions_figure(
    n_clicks,
    responsestation, 
    responseparameter, 
    predictorstation1, 
    predictorparameter1,
    predictorstation2, 
    predictorparameter2,
    predictorstation3, 
    predictorparameter3,
    predictorstation4, 
    predictorparameter4, 
    startdate, 
    enddate,
    modelselection,
    predict_startdate,
    predict_enddate):
    

# Filter out Nones in case not all station parameter dropdowns are used
    station_param_pairs2 = [
        (responsestation, responseparameter), 
        (predictorstation1, predictorparameter1), 
        (predictorstation2, predictorparameter2), 
        (predictorstation3, predictorparameter3),
        (predictorstation4, predictorparameter4)
    ]

    stationparampairs2 = []
    for i in station_param_pairs2:
        if i[0] is not None and i[1] is not None:
            stationparampairs2.append(i)
        
#Run the regression analysis 
    model_P = RegressionTool.RegressionFun(
        stationparampairs2, 
        str(startdate), 
        str(enddate)
    )
    
    model_P.train_model(modelselection, 0.3)
    model_P.make_predictions(predict_startdate, predict_enddate)

    return model_P.traintest_fig, model_P.modelfit_fig, model_P.predictions_fig
    
#----------------For predictions - Re-fit model to entire training/test set of data 

#Not sure why needing to filter out Nones again...
# Filter out Nones in case not all station parameter dropdowns are used
#     station_param_pairs0 = [
#         (responsestation, responseparameter), 
#         (predictorstation1, predictorparameter1), 
#         (predictorstation2, predictorparameter2), 
#         (predictorstation3, predictorparameter3),
# #         (predictorstation4, predictorparameter4)
#     ]
    
#     stationparampairs2 = []
#     for i in station_param_pairs0:
#         if i[0] is not None and i[1] is not None:
#             stationparampairs2.append(i)
        
#Run the regression analysis 
    model_pred = RegressionTool.RegressionFun(
        stationparampairs2, 
        str(startdate), 
        str(enddate)
    )

#Retrain model on all (or at least near all) of the data
    model_pred.train_model(modelselection, 0.01)
    model_pred.make_predictions(predict_startdate, predict_enddate)
    
    return model_pred.traintest_fig #model_pred.predictions_fig


#----------------------------------------------------------------------------------------------------------
#Run the app

if __name__ == '__main__':
              app.run_server()

