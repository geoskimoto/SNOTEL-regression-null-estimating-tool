
import dbs
from dash import dash_table
import pandas as pd
import views.regression as RegressionTool
from dash.dependencies import Input, Output, State


app = dbs.app

@app.callback(
    Output('db-datatable', 'children'),
    Input("datatable-button", "n_clicks"),
    [
    State("response-station", "value"),
    State("response-parameter", "value"),
    State("predictor-station1", "value"),
    State("predictor-parameter1", "value"),
    State("predictor-station2", "value"),
    State("predictor-parameter2", "value"),
    State("predictor-station3", "value"),
    State("predictor-parameter3", "value"),
    State("predictor-station4", "value"),
    State("predictor-parameter4", "value"),
    State("startdate_picker", "date"),
    State("enddate_picker", "date"),
    State("model_selection", "value"),
    ],
    prevent_initial_call=True
)

def datatable(
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
    modelselection
):

    station_param_pairs = [
        (responsestation, responseparameter),
        (predictorstation1, predictorparameter1),
        (predictorstation2, predictorparameter2),
        (predictorstation3, predictorparameter3),
        (predictorstation4, predictorparameter4),
    ]

    station_param_pairs[:] = [i for i in station_param_pairs if i[0] and i[1]]
    
    # Run the regression analysis
    model = RegressionTool.RegressionFun(
        station_param_pairs, str(startdate), str(enddate)
    )

    model.train_model(modelselection, 0.01)
    
    df = pd.DataFrame(
        model.stationparameterpairs, 
        model.begindate, 
        model.enddate, 
        model.regressor_type, 
        model.RMSE_train, 
        model.RMSE_test, 
        model.regr_data_string
    )
    
    dashtable = [
            dash_table.DataTable(
                id='our-table',
                columns=[{
                            'name': str(x),
                            'id': str(x),
                            'deletable': False,
                        } 
                        for x in df.columns],
                data=df.to_dict('records'),
                editable=True,
                row_deletable=True,
                filter_action="native",
                sort_action="native",  # give user capability to sort columns
                sort_mode="single",  # sort across 'multi' or 'single' columns
                page_action='none',  # render all of the data at once. No paging.
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
            ),
        ]
    
    return dashtable



if __name__ == "__main__":
    print('Hello there.')
