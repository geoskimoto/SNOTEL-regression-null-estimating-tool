#!./.venv/bin/python
# coding: utf-8

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import svm
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split #cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.linear_model import LassoCV, RidgeCV, HuberRegressor, LinearRegression
from functools import reduce
# Function that assigns the regression model to be used and its parameters
# import _pickle as cPickle
from _pickle import dumps
from utils import get_multiplestation_data


def regressionModel(regression_model):
    if regression_model == "Linear":
        regr = LinearRegression()
        return regr
    elif regression_model == "Lasso":
        regr = LassoCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000))
        return regr
    elif regression_model == "Huber":
        regr = HuberRegressor()
        return regr    
    elif regression_model == "SVM":
        regr = svm.SVR(
            kernel="rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            tol=0.001,
            C=1.0,
            epsilon=0.1,
            shrinking=True,
            cache_size=200,
            verbose=False,
            max_iter=-1,
        )
        return regr
    elif regression_model == "Random Forest":
        regr = RandomForestRegressor(n_estimators=100)
        return regr
    elif regression_model == "AdaBoost":
        regr = AdaBoostRegressor(random_state=0, n_estimators=100)
        return regr
    elif regression_model == 'XGBoost':
        regr = XGBRegressor(objective='reg:squarederror')
        return regr
    else:
        print('Choose either Linear, Lasso, Huber, SVM, Random Forest, AdaBoost, or XGBoost')



class RegressionFun:
    def __init__(
        self, stationparameterpairs, begindate, enddate, orient="records"
    ):  # , stations, parameter_of_interest):

        self.stationparameterpairs = stationparameterpairs
        self.stations = [i[0] for i in stationparameterpairs]
        self.parameters = [i[1] for i in stationparameterpairs]

        # Download Data from AWDB Web Service
        self.begindate = begindate
        self.enddate = enddate
        self.data = get_multiplestation_data(
            self.stationparameterpairs, begindate, enddate, orient
        ).dropna()

    def train_model(self, regression_model, test_size):

        """
        Function checks model fit on train and test sets.  Use to check which
        stations result in the best fitting model.  Once the best model is found,
        it can be used in the make_predictions function to predict null values.

        begindate: non-null date in 'mm/dd/yyyy' format
        enddate: non-null date in 'mm/dd/yyyy' format
        regression_model: can be 'Ridge', 'Lasso', 'Huber', 'SVM', 'Random Forest',
        'AdaBoost', 'GradientBoost'
        test_size: value between 0 and 1 that defines the percentage of the data
        to be used as the test_size
        """

        # Define Targets and Features (e.g. Response and Predictor Variables)
        target = self.data.iloc[:, 0]
        self.target = target
        features = self.data.iloc[:, 1:]
        self.features = features


        # Split into training and test sets
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=test_size, shuffle=False
        )

        #Scale the training data:
        #Changing the scale of the variable will lead to a corresponding change in the scale of the coefficients and standard errors, but no change in the significance or interpretation.
        #http://www2.kobe-u.ac.jp/~kawabat/ch06.pdf
        # scaler = MinMaxScaler()
        # scaler = StandardScaler()
        # features_train = scaler.fit_transform(features_train)
        # features_test = scaler.fit_transform(features_test)


        self.features_train = features_train
        self.features_test = features_test
        self.target_train = target_train
        self.target_test = target_test

        # Choose Regression Model:
        regr = regressionModel(regression_model)

        # Fit model on training set
        regr.fit(features_train, target_train)

        
        self.regr = regr
        self.regressor_type = regression_model
        self.regr_data_string = dumps(regr) #dict(_since_beginning), f, -1)
    

        
        # Run predictions on training features and test features
        target_train_pred = regr.predict(features_train)
        target_test_pred = regr.predict(features_test)

        self.target_train_pred = target_train_pred
        self.target_test_pred = target_test_pred

        # Print Root Mean Square Error for training and test sets:
        RMSE_train = mean_squared_error(target_train, target_train_pred)        
        RMSE_test = mean_squared_error(target_test, target_test_pred)
        
        RMSE_train_text = (
            "RMSE for training set:"
            + f"{RMSE_train: .5}"
        )
        RMSE_test_text = (
            "RMSE for test set:"
            + f"{RMSE_test: .5}"
        )

        self.RMSE_train = RMSE_train
        self.RMSE_test = RMSE_test

        # Predictions plot

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=target_train, x=target_train.index, mode="lines", name="Training Data"
            )
        )

        fig.add_trace(
            go.Scatter(
                y=target_train_pred,
                x=target_train.index,
                mode="lines",
                line_dash="dash",
                name="Model Predictions on Training Data",
            )
        )

        fig.add_trace(
            go.Scatter(
                y=target_test, x=target_test.index, mode="lines", name="Test Data"
            )
        )

        fig.add_trace(
            go.Scatter(
                y=target_test_pred,
                x=target_test.index,
                mode="lines",
                line_dash="dash",
                name="Model Predictions on Test Data",
            )
        )

        fig.add_annotation(
            xref="paper",
            x=0.5,
            yref="paper",
            y=1.12,
            text=f"{RMSE_train_text} - {RMSE_test_text}",
            font=dict(family="Courier New, monospace", size=14, color="black"),
            align="center",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="lightgrey",
            opacity=0.7,
            showarrow=False,
        )

        fig.update_yaxes(title_text=f"{self.parameters[0]}")
        fig.update_xaxes(title_text="Date")

        fig.update_layout(
            showlegend=True,
            height=500,
            width=950,
            title={
                "text": "Model Predictions on Training and Test Data",
                "xanchor": "center",
                "yanchor": "top",
                "y": 0.9,
                "x": 0.4,
            },
        )
        self.traintest_fig = fig

        if len(self.stations) == 3:
            customdata = self.data.reset_index()
            trace1 = go.Scatter3d(
                x=features_train.iloc[:, 0],
                y=features_train.iloc[:, 1],
                z=target_train,
                mode="markers",
                customdata=customdata,
                hovertemplate="Date: %{customdata[0]: .2f}"
                + "<br> x: %{customdata[1]: .2f}</br>"
                + "y: %{customdata[2]: .2f}"
                + "<br>z: %{customdata[3]: .2f}</br>",
                name="Response vs. Predictors",
            )

            trace2 = go.Scatter3d(
                x=features_train.iloc[:, 0],
                y=features_train.iloc[:, 1],
                z=pd.DataFrame(target_train_pred.tolist()).iloc[:, 0],
                mode="lines",
                name="Model Fit",
            )

            data = [trace1, trace2]
            layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))

            fig2 = go.Figure(data=data, layout=layout)

            fig2.update_layout(
                showlegend=True,
                legend={"orientation": "h"},
                height=500,
                width=950,
                title={
                    "text": "Regression Model",
                    "xanchor": "center",
                    "yanchor": "top",
                    "y": 0.9,
                    "x": 0.4,
                },
                scene=dict(
                    xaxis_title=f"{self.stationparameterpairs[0]} x",
                    yaxis_title=f"{self.stationparameterpairs[1]} y",
                    zaxis_title=f"{self.stationparameterpairs[2]} z",
                ),
            )

            self.modelfit_fig = fig2

        else:

            fig3 = make_subplots(
                rows=len(self.stations),
                cols=1,
                vertical_spacing=0.1,
            )
            for i in range(0, len(self.stations) - 1):
                fig3.append_trace(
                    go.Scatter(
                        x=features_train.iloc[:, i],
                        y=target_train,
                        mode="markers",
                        marker_color="#1f77b4",
                        hovertext=features_train.index,
                        name="Response vs. Predictors",
                    ),
                    row=i + 1,
                    col=1,
                )

                fig3.append_trace(
                    go.Scatter(
                        x=features_train.iloc[:, i],
                        y=pd.DataFrame(target_train_pred.tolist()).iloc[:, 0],
                        mode="lines",
                        marker_color="#ff7f0e",
                        hovertext=features_train.index,
                        name="Model Fit",
                    ),
                    row=i + 1,
                    col=1,
                )

                fig3.update_xaxes(
                    title_text=f"{self.stationparameterpairs[i]}", row=i + 1, col=1
                )
                fig3.update_yaxes(
                    title_text=f"{self.stationparameterpairs[0]}", row=i + 1, col=1
                )
            fig3.update_layout(
                showlegend=True,
                height=950,
                width=950,
                title={
                    "text": "Regression Model Slice(s)",
                    "xanchor": "center",
                    "yanchor": "top",
                    "x": 0.4,
                },
                font=dict(
                    family="Courier New, monospace",
                    size=12,
                ),
            )

            self.modelfit_fig = fig3

    def make_predictions(self, predict_begindate, predict_enddate):

        # Download Data from AWDB Web Service
        predict_data = get_multiplestation_data(
            self.stationparameterpairs, predict_begindate, predict_enddate
        )

        #Convert null values to 0, so they are obvious in the plot
        predict_target = predict_data.iloc[:, 0].fillna(0)
        predict_features = predict_data.iloc[:, 1:].fillna(0)

        # Run predictions
        predictions = self.regr.predict(predict_features)

        # Combine predictions into a df with rest of data
        column_name = 'Predictions - ' + f'{self.stations[0]}' + '(' + f'{self.parameters[0]}' + ')'
        predict_data.insert(0, column_name, list(predictions))
        predict_data[column_name] = predict_data[column_name].astype(float).round(2)
        predict_data.reset_index(inplace=True)
        self.predict_data = predict_data

        # Plot predictions
        fig_P = go.Figure()

        fig_P.add_trace(
            go.Scatter(
                y=predict_target,
                x=predict_target.index,
                mode="lines",
                name=f"{self.stations[0]} {self.parameters[0]}",
            )
        )

        fig_P.add_trace(
            go.Scatter(
                y=predictions,
                x=predict_features.index,
                mode="lines",
                name="Model Predictions",
            )
        )

        fig_P.update_layout(
            showlegend=True,
            height=500,
            width=950,
            title={
                "text": "Model Predictions",
                "xanchor": "center",
                "yanchor": "top",
                "y": 0.9,
                "x": 0.4,
            },
            xaxis_title="Date",
            yaxis_title=f"{self.stationparameterpairs[0]}",
        )

        self.predictions_fig = fig_P

