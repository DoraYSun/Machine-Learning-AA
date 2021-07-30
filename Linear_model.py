#%%
import numpy as np
from AADatabase import AADatabase
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# %%
class LinearRegressionModel():

    def __init__(self):
        """split data into train, validation, test sets"""
        self.X, self.y = AADatabase().feature_selector(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        # self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_train, self.y_train, test_size=0.55)
        self._standardize_features() # call the function to standardize features
        # return  self.X_train, self.X_validation, self.X_test, self.y_train, self.y_validation, self.y_test
    
    def _standardize_features(self):
        """standerdize features in dataset"""
        scaler = StandardScaler()
        self.X_train = scaler.fit(self.X_train).transform(self.X_train)
        # self.X_validation = scaler.fit(self.X_validation).transform(self.X_validation)
        self.X_test = scaler.fit(self.X_test).transform(self.X_test)
        # return self.X_train, self.X_validation, self.X_test
        # return self.X_train, self.X_test

    def fit_linear_regression_model(self):
        regr = LinearRegression()
        regr.fit(self.X_train, self.y_train)
        print ('Coefficients: ', regr.coef_)
        print ('Intercept: ', regr.intercept_)

    
    def evaluate_linear_regression_model(self):
      
        y_test_pred = LinearRegression().fit(self.X_test, self.y_test).predict(self.X_test)
        linear_regression_model_MAE = mean_absolute_error(self.y_test, y_test_pred)
        linear_regression_model_MSE = mean_squared_error(self.y_test, y_test_pred)
        linear_regression_model_R2 = r2_score(self.y_test, y_test_pred)
        print("MAE of linear regression model is : ", linear_regression_model_MAE)
        print("MSE of linear regression model is : ", linear_regression_model_MSE)
        print("R2 Score of linear regression model is : ", linear_regression_model_R2)


# %%
model_1 = LinearRegressionModel()
model_1.fit_linear_regression_model()
model_1.evaluate_linear_regression_model()
# %%
