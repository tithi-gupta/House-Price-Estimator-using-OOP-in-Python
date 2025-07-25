
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class HouseData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X = None
        self.y = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        if "Address" in self.df.columns:
            self.df.drop(columns=["Address"], inplace=True)
        self.X = self.df.drop("Price", axis=1)
        self.y = self.df["Price"]
        return self.X, self.y

class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train):
        return self.scaler.fit_transform(X_train)

    def transform(self, X_test):
        return self.scaler.transform(X_test)

class HousePriceModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class ModelEvaluator:
    def evaluate(self, y_test, y_pred):
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # fixed here
        r2 = r2_score(y_test, y_pred)
        print("Model Evaluation:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
  