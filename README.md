# The impact of heat pumps on day-ahead energy community load forecasting

We publish most forecasting methods, our benchmarking pipeline, all feature-engineered and selected data, as well as our final results of the paper "The impact of heat pumps on day-ahead energy community load forecasting" (Semmelmann et al., 2024) in this repository. We encourage fellow researchers and practitioners to use our pipeline to benchmark novel load forecasting methods for households, heat pumps and energy communuities featuring heat pumps and regular load. 


### How to benchmark your own models against the results

1. **Data Preparation**: Use the dataframes `Feature Selection aggregated energy community load.pkl`, `Feature Selection aggregated heat pump load.pkl`, and `Feature Selection aggregated household load.pkl` to predict whole energy community (`Comb`), heat pump (`HP`), and household (`HH`) loads. In the folder data/traintest, a train/test representation for tabular methods, like XGBoost, can be found.

2. **Model Training and Prediction**:
    - Train with the data from the 365 days before the first of every month in 2020.
    - Predict the day-ahead load based on data from the two previous days.
    - Re-train before every new month.

3. **Results Saving**:
    - Save the results (for the whole year) in the results folder with the following naming convention: `{inputdata} {modelname} Results.pkl`, e.g., `HH LSTM Results.pkl`.
    - In the file, there should be a `Agg Load` column for the real values, and a `Predict` column for the forecasted values, for every hour in the year.

4. **Evaluation Updates**:
    - Edit the `02 Evaluate` notebook; append below the `Individual Aggregated vs. directly combined` section the `methods` list and append the `{modelname}`.
    - You will find the benchmarked results in the following.
