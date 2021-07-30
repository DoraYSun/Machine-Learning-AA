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
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = AADatabase().split_data(standardize_features=True)
        self.regr = LinearRegression()
    

    def fit_linear_regression_model(self):
        """fit training data into linear regression model"""     
        self.regr.fit(self.X_train, self.y_train)
        print ('Coefficients: ', self.regr.coef_)
        print ('Intercept: ', self.regr.intercept_)

    
    def evaluate_linear_regression_model(self):
      
        y_test_pred = self.regr.fit(self.X_test, self.y_test).predict(self.X_test)
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
