#!/usr/bin/env python
# coding: utf-8

# In[ ]:




#------------------Import libraries----------------------------------------------


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import scipy.stats as stats

import dash
from dash import dcc
from dash import html
import dash_auth
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from datetime import date
import snotel_regressiontool as RegressionTool


#-----------------variables to be used in different places within app-------------------

heading={'color':'black', 'font-size':50,'text-Align':'center','font-weight': 'bold'}
subheading={'color':'black', 'font-weight': 'bold', 'text-Align':'left'}
button_style = {'background-color': 'grey' , 'border': 'none', 'color': 'white', 'padding': '15px',  'text-align': 'center',
  'text-decoration': None,  'display': 'inline-block',  'font-size': '19px', 'border-radius': '12px',} 
#   'hover':{'background-color': '#4CAF50', 'color': 'white'},'transition-duration': '0.4s', 'cursor': 'pointer'}
radio_style = {'color':'black', 'font-weight': 'bold', 'padding': '12px 12px'}


# ORDCO_stations = pd.read_csv('ORDCO_SNTL_Triplets.csv')
# All_SNOTEL_stations = pd.read_csv('SNTL Triplets.csv')
# # Stations = pd.read_excel('StationTriplets.xlsx')
# station_names = stations.loc[:,'Extended Name'].tolist()
# triplets = stations.loc[:, 'Station Triplet'].tolist()

# options = []
# for i in range(len(Stations.index)):
#   options.append({'label': station_names[i], 'value': triplets[i]})
           
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

#MAKE SURE TO UNCOMMENT THIS LINE BEFORE TRYING TO DEPLOY!!!
server = app.server

app.config.suppress_callback_exceptions = True

auth = dash_auth.BasicAuth(
    app,
    {'ORDCO': 'SNOTEL'}
)

#--------------------Layout Design ---------------------------------------------

controls = dbc.Card(
    [
        dbc.Form(
            [
                        
                        html.Br(),
                        html.Div(children = [
                            html.H5(['Select grouping:'], style=subheading),
                            dcc.RadioItems(
                                id='station_selector',
                                options=[{'label': ' ORDCO ', 'value': 'ORDCO_stations'},
                                         {'label': ' All SNOTEL Stations ', 'value': 'All_SNOTEL_stations'}],
                                value='ORDCO_Stations',
                                style = radio_style,
                                labelStyle={'display': 'block'}
                            ),

                            html.Br(),
                            html.H5(['Select the station and parameter to be used as the response:'], style=subheading),
                            dcc.Dropdown(id = 'response-station', 
                                         options = [], 
                                         multi=False,
#                                          value=[]
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
                                         options = [], 
                                         multi = False,
#                                          value = []
                                         value = '301:CA:SNTL',
                    #                      style = station_dropdown_style,
                                        ),
                            dcc.Dropdown(id = 'predictor-parameter1', 
                                        options = parameter_options,
                                        multi = False,
                                        value="SNWD",
                    #                     style = parameter_dropdown_style,
                                        ),  
                            html.Br(),
                            dcc.Dropdown(id = 'predictor-station2', 
                                         options = [], 
                                         multi = False,
                                         value = '391:CA:SNTL',
                    #                      style = station_dropdown_style
                                        ),
                            dcc.Dropdown(id = 'predictor-parameter2', 
                                        options = parameter_options,
                                        multi = False,
                                        value = 'WTEQ',
                    #                     style = parameter_dropdown_style,
                                        ),

                            html.Br(),
                            dcc.Dropdown(id = 'predictor-station3', 
                                         options = [], 
                                         multi = False,
#                                          value = [],
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
                                         options = [], 
                                         multi = False,
#                                          value = [],
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
                            min_date_allowed=date(1950, 10, 1),
                    #         max_date_allowed=date(2017, 9, 19),
                    #         initial_visible_month=date(2017, 8, 5),
                            date = date(2015,10,1)
                    #         date = '09/01/2021'
                        ),

                        dcc.DatePickerSingle(
                            id='enddate_picker',
                            min_date_allowed=date(1950, 10, 1),
                    #         max_date_allowed=date(2017, 9, 19),
                    #         initial_visible_month=date(2017, 8, 5),
                            date = date(2021,2,1) #'09/20/2021',
                        ),

                        html.Br(),
                        html.Br(),
                        html.H5(['Select Regression Model to be used:']),
                        dcc.Dropdown(id = 'model_selection',
                                    options = model_options,
                                    value = 'SVM'
                        ),    


                        html.Br(),
                            html.H5(['First click returns model fit figure(s).  Second click returns training vs. test predictions figure.']),                        html.Div([
                            html.Button(
                                id='submit-button-training', 
                                children = 'Train Model',
                                style=button_style
                                )
                        ], style={
                              'display': 'block',
                              'margin-left': 'auto',
                              'margin-right': 'auto',
                              'width': '40%'}
                        )
                        ]),

                        html.Br(),
                        html.Br(),
                        html.Div([

                            html.H5(['Select date range to run predictions after strong fitting model has been found:'], style=subheading),
                            dcc.DatePickerSingle(id='predict_startdate_picker',
                            date = date(2021,2,1)
                            ),
                            dcc.DatePickerSingle(id='predict_enddate_picker',
                            date = date(2021,3,1)
                            ),
                            ]),
                            html.Br(),
                            html.Div([
                                html.Button(
                                    id='submit-button-predictions', 
                                    children = 'Run Predictions',
                                    style=button_style
                                )
                            ], style={
                                  'display': 'block',
                                  'margin-left': 'auto',
                                  'margin-right': 'auto',
                                  'width': '40%'}
                            ),
#                          ]),
                
                        html.Br(),
                        html.Br(),
                        html.Details(
                            children=[
                                html.Summary('Useful Tidbits'),
                                html.P([
                                    html.Ul([
                                    html.Li('Tool can be used to estimate missing or bad data for any sensor on any SNOTEL or SNOLITE station in    .'),
                                    html.Br(),
                                    html.Li('Use the NRCS IMAP (https://www.nrcs.usda.gov/wps/portal/wcc/home/quicklinks/imap) to find suitable predictor stations based on proximity, elevation, aspect, etc. for the response station of interest.'),
                                    html.Br(),
                                    html.Li('When evaluating models, pay attention to the Root Mean Square Error (RMSE) between the training and test data sets.'),
                                    html.Ul([
                                        html.Br(),
                                        html.Li('As a general rule you want to minimize the error of both and they should be similar in value.'),
#                                     html.Br(),
                                        html.Li('RMSE test > RMSE train => OVERFITTING'),
                                        html.Li('RMSE test < RMSE train => UNDERFITTING'),
                                        html.Li('Better to have an underfitting than an overfitting model.')
                                    ]),
                                    html.Br(),
                                    html.Br(),
                                     html.Li('When choosing a regression model type to make real world estimates, stick to linear models (linear, ridge, lasso, and huber).  These models are easy to explain and defend.  Additionally, the more advanced models have major drawbacks (primarily overfitting) that require either tinkering with the length of the training dataset or fine tuning parameters that are not available in this app.'),
                                    ])])
                            ]
                        )
            ])]),



# traintest = dbc.Card(
#     html.Div([
        
#         html.Div([
#             dcc.Graph(id='traintest plot', 
#                       figure={}, 
#                       responsive=True,
# #                       style={'width':12, 'height': 5},
#              ),
#         ]),
                    

#         html.Div([
#             dcc.Graph(id='modelfit plot', 
#                       figure={}, 
#                       responsive=True,
# #                       style={'width':12, 'height': 5},
#             ),
#         ]),
#         html.Div([
#             dbc.Row(
#                 html.Div(id='pred plots', children=[])              
#         )])
#     ])
#     )
            
training = dbc.Card(
    html.Div([
        html.Div(id='traintest plots', children=[]),
        ])),          

modelfit = dbc.Card(
    html.Div([
        html.Div(id='modelfit_plots', children=[])
        ])), 
             
predic = dbc.Card(
    html.Div([
        html.Div(id='pred plots', children=[])              
        ])),
                
                
app.layout = dbc.Container([
                        html.Div([

                            html.Br(),
                            html.P([                                   
                                html.Br(),
                                html.H5('SNOTEL Regression Tool', style = heading),
                                html.H4('Estimate missing or bad data using regression', style = subheading)
                            ], style = {'text-align': "center"}),
                            ]),
                            html.Br(),
                            dbc.Row([
                                    dbc.Col(controls, width=3),
#                                     dbc.Row(
                                    dbc.Col(
                                        [dbc.Row(
                                            dbc.Col(training)
                                        ),
                                         dbc.Row(
                                            dbc.Col(modelfit)
                                        ),
                                        dbc.Row(
                                            dbc.Col(predic),
                                        )],
                                    ),
#                                     dbc.Col(
#                                         [dbc.Row(
#                                             dbc.Col(modelfit)
#                                         )]
#                                     )

                                    
                                    
                            ])

                                        
# ]),
#                             dbc.Row([
#                                     dbc.Col(predic, style = heading)
#                             ])
#                         ])
                                
            ], fluid=True)
                
                     

#--------------------------Callbacks--------------------------------

#Call back to populate dropdown lists with selected stations:

@app.callback(
    Output('response-station','options'),
    Output('predictor-station1','options'),
    Output('predictor-station2','options'),
    Output('predictor-station3','options'),
    Output('predictor-station4','options'),
    
    Input('station_selector', 'value'))

def populate_dropdowns(chosen_network):
    if chosen_network == 'ORDCO_stations':
        Stations = pd.read_csv('ORDCO_SNTL_Triplets.csv')
    else: # chosen_network == 'All_SNOTEL_stations':  #unbound local variable error when using elif for some reason
        Stations = pd.read_csv('SNTL Triplets.csv')

    station_names = Stations.loc[:,'Extended Name'].tolist()
    triplets = Stations.loc[:, 'Station Triplet'].tolist()

    station_list = []
    for i in range(len(Stations.index)):
          station_list.append({'label': station_names[i], 'value': triplets[i]})
    
    #Have to return a value for every Output in the callback...reason for this weirdness
    return station_list, station_list, station_list, station_list, station_list 






#----------------------------
@app.callback(
    Output('traintest plots', 'children'),
    Output('modelfit_plots','children'),

#     Output('traintest plot','figure'),
#     Output('modelfit plot','figure'),
    Input('submit-button-training', component_property='n_clicks'),
    State('response-station','value'),
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
    State('predict_enddate_picker', 'date'),
)


    
def train_figures(
    n_clicks,
#     n_clicks2,
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
    predict_enddate
):
    

# Filter out Nones in case not all station parameter dropdowns are used
    station_param_pairs = [
        (responsestation, responseparameter), 
        (predictorstation1, predictorparameter1), 
        (predictorstation2, predictorparameter2), 
        (predictorstation3, predictorparameter3),
        (predictorstation4, predictorparameter4)
    ]

    stationparampairs = []
    for i in station_param_pairs:
        if i[0] is not None and i[1] is not None:
            stationparampairs.append(i)
       
#Run the regression analysis 
    model = RegressionTool.RegressionFun(
        stationparampairs, 
        str(startdate), 
        str(enddate)
    )
    
    model.train_model(modelselection, 0.3)
    model.make_predictions(predict_startdate, predict_enddate)


            
#     return model.traintest_fig, model.modelfit_fig[0] #, model.modelfit_fig[1] #,model.traintest_fig, model_P.predictions_fig



    traintest_graph = html.Div(
        children=[
            dcc.Graph(
#                 id={
#                     'type': 'dynamic-graph',
#                     'index': n_clicks
#                 },
                figure=model.traintest_fig,
                responsive=True
            )            
        ]),

 

    children = []
#     if len(model.stations) == 3:
    children.append(
        dcc.Graph(
            figure=model.modelfit_fig,
            responsive=True,
            style={'width': '100%', 'height': '100vh'}
        )
    )   
    modelfit_graphs = html.Div(children)

#     else:
#         for i in range(len(model.modelfit_fig)):
#                 children.append(
#                     dcc.Graph(
#         #                 id={
#         #                     'type': 'dynamic-graph',
#         #                     'index': n_clicks
#         #                 },

#                         figure=model.modelfit_fig[i],
#                         responsive=True
#                     )
#                 )


#         modelfit_graphs = html.Div(children),

    #     modelfit_graph = html.Div(
    #             children=[
    #             dcc.Graph(
    # #                 id={
    # #                     'type': 'dynamic-graph',
    # #                     'index': n_clicks
    # #                 },
    #                 figure=model.modelfit_fig,
    #                 responsive=True
    #             )            
    #         ])

    return traintest_graph, modelfit_graphs
    
    
    
    
    
    
    
    
    
    
    
    
#----------------For predictions - Re-fit model to entire training/test set of data 

@app.callback(
    [Output('pred plots', 'children'),
#     Output('regression model plot','figure'),
#     Output('predictions_figure','figure')
    ],

#     [Input('submit-button-training', component_property='n_clicks')],
    [Input('submit-button-predictions', component_property='n_clicks')],
    
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
    ],
    prevent_initial_call=True)


    
def train_figures(
    n_clicks,
#     n_clicks2,
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
    model_Pr = RegressionTool.RegressionFun(
        stationparampairs2, 
        str(startdate), 
        str(enddate)
    )
    
    model_Pr.train_model(modelselection, 0.01)
    model_Pr.make_predictions(predict_startdate, predict_enddate)

    
    predictions_graph = html.Div(
#         style={'width': '80%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
        children=[
#             html.H3('Predictions'),
            dcc.Graph(
#                 id={
#                     'type': 'dynamic-graph',
#                     'index': n_clicks
#                 },
                figure=model_Pr.predictions_fig,
                responsive=True
            )            
        ]),
    
    
            
    return predictions_graph,  #model_P.traintest_fig,# model_P.modelfit_fig, model_P.predictions_fig

#----------------------------------------------------------------------------------------------------------
#Run the app

if __name__ == '__main__':
              app.run_server()