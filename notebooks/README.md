# Probabilistic forecasting for Belgian Day Ahead Market and Imbalance Market using TSFMs

## Introduction
This directory contains code for gathering results and evaluating performance of point forecast for Belgian Day Ahead Market and Imbalance Market, using the TSFMs, i.e. Chronos-2 model and TimesFM 2.5 model, compare with baseline data-driven models (DNN and LEAR). 

## Point forecasts
Point forecast results summary for each model and market are stored in the following CSV files, in which "price" column contains the true values:
- dam_summary_point.csv (hourly forecasts for whole year of 2024 - DAM)
- imb_summary_point.csv (15-min window forecasts, each window contains 8-step-ahead forecasts, for whole year of 2023 - IMB)

## Forecasting intervals
Beside point forecasts, TSFMs provided zero-shot quantile forecasts, which we choose 0.1 and 0.9 quantiles saved in the following CSV files:
- dam_summary_upper.csv (0.9 quantile hourly forecasts for whole year of 2024 - DAM)
- dam_summary_lower.csv (0.1 quantile hourly forecasts for whole year of 2024 - DAM)
- imb_summary_upper.csv (0.9 quantile 15-min window forecasts, each window contains 8-step-ahead forecasts, for whole year of 2023 - IMB)
- imb_summary_lower.csv (0.1 quantile 15-min window forecasts, each window contains 8-step-ahead forecasts, for whole year of 2023 - IMB)

## Point forecast evaluation
The point forecasts are evaluated using the following metrics: Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
DM test also perform to compare the performance of different models. Notebook file: `comparison.ipynb`

## Forecast interval evaluation
We evaluate the forecast intervals using the following metrics: Winkler Score, Coverage and Mean Width of the interval.
### Phase 1: TSFMs zeroshot quantile forecasts evaluation
Working notebook: conformal_prediction.ipynb (Part 3: Native QR from TSFMs)

Basically, we can read files from dam_summary_upper.csv and dam_summary_lower.csv for DAM, imb_summary_upper.csv and imb_summary_lower.csv for IMB, and calculate the above metrics to evaluate the performance of the forecast intervals.

### Phase 2: TSFMs point forecasts + Conformal Prediction (WCP and ACI), with hyperparameter tuning evaluation 
- Calibaration set: 2023-01-01 to 2023-12-31 for DAM (in which further split to first 9 months and later 3 months are for hyperparameter tuning of WCP and ACI, respectively), 2023-01-01 to 2023-06-30 for IMB (in which further split to first 3 months and later 3 months  are for hyperparameter tuning of WCP and ACI, respectively)
- Test set: 2024-01-01 to 2024-12-31 for DAM, 2023-07-01 to 2023-12-31 for IMB
- Data for calibration and test sets are dam_summary_point.csv and imb_summary_point.csv, which contain the true values and point forecasts from TSFMs.
- Working notebook: conformal_prediction.ipynb (Part 1: CP for DAM, Part 2: CP for IMB)
### Phase 3: Quantile regression averaging for IMB baselines 
We also perform quantile regression averaging (QRA) for IMB, which is a post-processing method to combine the quantile forecasts from different models, and evaluate the performance of the combined forecasts. 
- Working notebook: QRA_imb_2023.ipynb 
- 9 baselines models in `imb_summary_point.csv` will be used for QRA: `['DNN_2023_large', 'XGB_2023_medium','DNN_2023_small','LEAR_2023_medium', 'DNN_2023_medium','LEAR_2023_large', 'LEAR_2023_small', 'XGB_2023_large', 'XGB_2023_small']`
- Model training: first 6 months of 2023
- Test set: 2023-07-01 to 2023-12-31
- Each "QH" is a model, totalling 8 models

### Phase 4: Quantile regression averaging for all models
Working notebook: QRA_all_2023.ipynb (to be created)

## Error analysis:
We also perform error analysis 
- Working notebook: error_analysis.ipynb (to be created). Part 1, analysis of errors of point forecasts; Part 2, analysis of errors (miscoverage) of forecast intervals.
- Realiability diagram for forecast intervals, to see if there are any patterns in the miscoverage of the intervals.
- Heat map of errors for each hour of day, day of week, and month of year (for DAM), to see if there are any patterns in the errors.
- Heat map of errors for each 15-min window and month of year (for IMB), to see if there are any patterns in the errors.

