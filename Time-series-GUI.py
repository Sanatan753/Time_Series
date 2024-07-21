import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def get_Patterns(TSeries, n_inputs, h):
    X, y = pd.DataFrame(np.zeros((len(TSeries) - n_inputs - h + 1, n_inputs))), pd.DataFrame()
    for i in range(len(TSeries)):
        end_ix = i + n_inputs + h - 1
        if end_ix > len(TSeries) - 1:
            break
        for j in range(n_inputs):
            X.loc[i, j] = TSeries.iloc[i + j, 0]
        i = i + n_inputs
        y = pd.concat([y, TSeries.iloc[end_ix]], ignore_index=True)
    return pd.DataFrame(X), pd.DataFrame(y)

def minmaxNorm(originalData, lenTrainValidation):
    max2norm = max(originalData.iloc[0:lenTrainValidation, 0])
    min2norm = min(originalData.iloc[0:lenTrainValidation, 0])
    lenOriginal = len(originalData)
    normalizedData = [(originalData.iloc[i] - min2norm) / (max2norm - min2norm) for i in range(lenOriginal)]
    return pd.DataFrame(normalizedData)

def minmaxDeNorm(originalData, forecastedData, lenTrainValidation):
    max2norm = max(originalData.iloc[0:lenTrainValidation, 0])
    min2norm = min(originalData.iloc[0:lenTrainValidation, 0])
    lenOriginal = len(originalData)
    denormalizedData = [(forecastedData.iloc[i] * (max2norm - min2norm)) + min2norm for i in range(lenOriginal)]
    return pd.DataFrame(denormalizedData)

def findRMSE(Timeseries_Data, forecasted_value, lenTrainValidation):
    l = Timeseries_Data.shape[0]
    lenTest = l - lenTrainValidation
    trainRMSE = np.sqrt(np.mean((forecasted_value.iloc[:lenTrainValidation, 0] - Timeseries_Data.iloc[:lenTrainValidation, 0]) ** 2))
    testRMSE = np.sqrt(np.mean((forecasted_value.iloc[lenTrainValidation:, 0] - Timeseries_Data.iloc[lenTrainValidation:, 0]) ** 2))
    return trainRMSE, testRMSE

def findMAE(Timeseries_Data, forecasted_value, lenTrainValidation):
    l = Timeseries_Data.shape[0]
    lenTest = l - lenTrainValidation
    trainMAE = np.mean(np.abs(forecasted_value.iloc[:lenTrainValidation, 0] - Timeseries_Data.iloc[:lenTrainValidation, 0]))
    testMAE = np.mean(np.abs(forecasted_value.iloc[lenTrainValidation:, 0] - Timeseries_Data.iloc[lenTrainValidation:, 0]))
    return trainMAE, testMAE

def Find_Fitness(x, y, lenValid, lenTest, model):
    NOP = y.shape[0]
    lenTrain = NOP - lenValid - lenTest
    xTrain = x.iloc[0:lenTrain, :]
    xValid = x.iloc[lenTrain:(lenTrain + lenValid), :]
    xTest = x.iloc[(lenTrain + lenValid):NOP, :]
    yTrain = y.iloc[0:lenTrain, 0]
    yValid = y.iloc[lenTrain:(lenTrain + lenValid), 0]
    yTest = y.iloc[(lenTrain + lenValid):NOP, 0]
    model.fit(xTrain, yTrain)
    yhatNorm = model.predict(x).flatten().reshape(x.shape[0], 1)
    return pd.DataFrame(yhatNorm)

class TimeSeriesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time Series Analysis and Forecasting")
        self.root.geometry("600x400")
        self.root.configure(bg="white")

        self.timeseries_data = None
        self.model = LinearRegression()

        self.create_widgets()

    def create_widgets(self):
        self.upload_button = tk.Button(self.root, text="Upload Data", command=self.upload_data, font=("Arial", 12))
        self.upload_button.pack(pady=10)

        self.analyze_button = tk.Button(self.root, text="Analyze", command=self.analyze, font=("Arial", 12))
        self.analyze_button.pack(pady=10)

        self.forecast_label = tk.Label(self.root, text="Enter the number of steps to forecast:", font=("Arial", 12), bg="white")
        self.forecast_label.pack(pady=10)

        self.forecast_entry = tk.Entry(self.root, font=("Arial", 12))
        self.forecast_entry.pack(pady=10)

        self.forecast_button = tk.Button(self.root, text="Forecast", command=self.forecast, font=("Arial", 12))
        self.forecast_button.pack(pady=10)

        self.text = tk.Text(self.root, font=("Arial", 12), height=10, width=70)
        self.text.pack(pady=10)

    def upload_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.timeseries_data = pd.read_csv(file_path, header=None)
            self.text.insert(tk.END, f"Data loaded from {file_path}\n")
        else:
            messagebox.showwarning("Warning", "No file selected")

    def analyze(self):
        if self.timeseries_data is not None:
            plt.title("Autocorrelation Plot")
            plt.xlabel("Lags")
            plt.acorr(np.array(self.timeseries_data.iloc[:, 0], dtype=float), maxlags=20)
            plt.grid(True)
            plt.show()

            sns.rugplot(data=self.timeseries_data, height=.03, color='darkblue')
            sns.histplot(data=self.timeseries_data, kde=True)
            plt.show()

            LagLength = 10
            h = 1
            lt = self.timeseries_data.shape[0]
            lenTrain = int(round(lt * 0.7))
            lenValidation = int(round(lt * 0.15))
            lenTest = int(lt - lenTrain - lenValidation)

            normalizedData = minmaxNorm(self.timeseries_data, lenTrain + lenValidation)
            X, y = get_Patterns(normalizedData, LagLength, h)
            model = LinearRegression()
            name = 'LinearRegression'
            file1 = './' + str(name) + "_Accuracy.xlsx"
            file2 = './' + str(name) + "_Forecasts.xlsx"
            Forecasts = pd.DataFrame()
            Accuracy = pd.DataFrame()

            ynorm1 = Find_Fitness(X, y, lenValidation, lenTest, model)
            ynorm = pd.DataFrame(normalizedData.iloc[0:(LagLength + h - 1), 0])
            ynorm = pd.concat([ynorm, ynorm1], ignore_index=True)
            yhat = minmaxDeNorm(self.timeseries_data, ynorm, lenTrain + lenValidation)
            train_rmse, test_rmse = findRMSE(self.timeseries_data, yhat, lenTrain + lenValidation)
            train_mae, test_mae = findMAE(self.timeseries_data, yhat, lenTrain + lenValidation)
            Accuracy.loc[0, 'Train RMSE'], Accuracy.loc[0, 'Test RMSE'] = train_rmse, test_rmse
            Accuracy.loc[0, 'Train MAE'], Accuracy.loc[0, 'Test MAE'] = train_mae, test_mae
            Forecasts = pd.concat([Forecasts, yhat.T], ignore_index=True)
            Accuracy.to_excel(file1, sheet_name='Accuracy', index=False)
            Forecasts.to_excel(file2, sheet_name='Forecasts', index=False)

            self.text.insert(tk.END, f"Accuracy Metrics:\n{Accuracy}\n")
        else:
            messagebox.showwarning("Warning", "Please load data first")

    def forecast(self):
        try:
            steps = int(self.forecast_entry.get())
            if self.timeseries_data is None:
                messagebox.showwarning("Warning", "Please load data first")
                return

            LagLength = 10
            h = 1
            lt = self.timeseries_data.shape[0]
            lenTrain = int(round(lt * 0.7))
            lenValidation = int(round(lt * 0.15))
            lenTest = int(lt - lenTrain - lenValidation)

            normalizedData = minmaxNorm(self.timeseries_data, lenTrain + lenValidation)
            X, y = get_Patterns(normalizedData, LagLength, h)
            model = LinearRegression()
            name = 'LinearRegression'
            Forecasts = pd.DataFrame()

            ynorm1 = Find_Fitness(X, y, lenValidation, lenTest, model)
            ynorm = pd.DataFrame(normalizedData.iloc[0:(LagLength + h - 1), 0])
            ynorm = pd.concat([ynorm, ynorm1], ignore_index=True)
            yhat = minmaxDeNorm(self.timeseries_data, ynorm, lenTrain + lenValidation)

            forecast_values = yhat.iloc[-steps:].values.flatten()
            self.text.insert(tk.END, f"Forecast for next {steps} steps: {forecast_values}\n")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of steps")

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeSeriesApp(root)
    root.mainloop()
