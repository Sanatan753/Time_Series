# Time_Series

# README

## Overview

This project demonstrates time series forecasting using various statistical and machine learning techniques. The primary goal is to predict future values of a time series dataset and evaluate the performance of the models. The dataset used is `Lynx.csv`, which contains historical data.

## Prerequisites

Ensure you have the following packages installed:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install these packages using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Project Structure

- `Lynx.csv`: The dataset containing historical time series data.
- `main.py`: The main Python script that includes all the functions and execution logic for the time series forecasting.
- `LinearRegression_Accuracy.xlsx`: Excel file that stores the accuracy metrics of the Linear Regression model.
- `LinearRegression_Forecasts.xlsx`: Excel file that stores the forecasted values of the Linear Regression model.

## Functions

### `get_Patterns(TSeries, n_inputs, h)`
Splits a univariate time series into patterns for supervised learning.

**Parameters:**
- `TSeries`: Time series data.
- `n_inputs`: Number of input features (lags).
- `h`: Forecast horizon.

**Returns:**
- `X`: Input features.
- `y`: Output labels.

### `minmaxNorm(originalData, lenTrainValidation)`
Applies Min-Max normalization to the dataset.

**Parameters:**
- `originalData`: Original time series data.
- `lenTrainValidation`: Length of the training and validation data.

**Returns:**
- Normalized data as a DataFrame.

### `minmaxDeNorm(originalData, forecastedData, lenTrainValidation)`
Applies Min-Max de-normalization to the forecasted data.

**Parameters:**
- `originalData`: Original time series data.
- `forecastedData`: Forecasted normalized data.
- `lenTrainValidation`: Length of the training and validation data.

**Returns:**
- De-normalized forecasted data as a DataFrame.

### `findRMSE(Timeseries_Data, forecasted_value, lenTrainValidation)`
Calculates the Root Mean Squared Error (RMSE) for both training and test sets.

**Parameters:**
- `Timeseries_Data`: Original time series data.
- `forecasted_value`: Forecasted values.
- `lenTrainValidation`: Length of the training and validation data.

**Returns:**
- Train RMSE and Test RMSE.

### `findMAE(Timeseries_Data, forecasted_value, lenTrainValidation)`
Calculates the Mean Absolute Error (MAE) for both training and test sets.

**Parameters:**
- `Timeseries_Data`: Original time series data.
- `forecasted_value`: Forecasted values.
- `lenTrainValidation`: Length of the training and validation data.

**Returns:**
- Train MAE and Test MAE.

### `Find_Fitness(x, y, lenValid, lenTest, model)`
Trains the model and predicts values.

**Parameters:**
- `x`: Input features.
- `y`: Output labels.
- `lenValid`: Length of validation data.
- `lenTest`: Length of test data.
- `model`: Machine learning model.

**Returns:**
- Predicted normalized values as a DataFrame.

## Execution

1. Read the time series dataset `Lynx.csv`.
2. Normalize the dataset.
3. Transform the time series into patterns using a sliding window.
4. Train a Linear Regression model.
5. Evaluate the model using RMSE and MAE.
6. Save the accuracy metrics and forecasted values to Excel files.

## Usage

Run the main script `main.py` to execute the time series forecasting and evaluation:

```bash
python main.py
```

## Visualization

The script also includes visualization for the autocorrelation plot and a rug plot for the time series data.

## Author

Sanatan Kisku

## License

This project is licensed under the MIT License.

## Contact

For any queries, please contact Sanatan Kisku.

---

Feel free to customize this README file as per your requirements.
