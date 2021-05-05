# SNOTEL-regression-null-estimating-tool
## Tool to evaluate regression models and estimate missing data at non-reporting SNOTEL stations using nearby stations.

Connected to the US Department of Agriculture's Air - Water Database (AWDB), so estimates can be done with most up-to-date data.

Training and checking model fit with train test split and RMSE.  Function allows for the evaluation of mutiple models choosing different SNOTEL stations as predictors, or any combination of.
![check_model](/check_model.png)

After a good model is found using appropriate nearby stations as predictors, predications can be made on null data.
![make_predictions](/make_predictions.png)
