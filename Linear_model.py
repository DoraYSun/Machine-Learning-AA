#%%
import time
from AADatabase import AADatabase
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score



# %%
class LinearRegressionModel():

    def __init__(self):
        """split data into train, validation, test sets"""
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = AADatabase().split_data(standardize_features=True)
        self.lr = LinearRegression()
        self.output = {'model':'Linear Regression'}
        

    def _linear_regression_hyperparameters_turning(self):
        """turning hyperparameters of linear regression model for best performance"""
        start_time = time.time()
        lr_parameters = {'fit_intercept': [True, False], 
                    'normalize': [True, False]
                    }
        
        lr_grid = GridSearchCV(estimator=self.lr, param_grid = lr_parameters)
        lr_grid.fit(self.X_validation, self.y_validation)
        self.best_params = lr_grid.best_params_
        end_time = time.time()
        self.output['time'] = end_time - start_time
        self.output['best_params'] = self.best_params
    
    def evaluate_linear_regression_model(self):
        """fit training data into linear regression model and reflect how well linear regression model performs on validationset"""
        self._linear_regression_hyperparameters_turning()
        lr_best = LinearRegression(fit_intercept=self.best_params['fit_intercept'], normalize=self.best_params['normalize'])
        y_train_pred = lr_best.fit(self.X_train, self.y_train).predict(self.X_train)
        y_val_pred = lr_best.fit(self.X_validation, self.y_validation).predict(self.X_validation)
        y_test_pred = lr_best.fit(self.X_test, self.y_test).predict(self.X_test)
        linear_regression_train_R2 = r2_score(self.y_train, y_train_pred)
        linear_regression_val_R2 = r2_score(self.y_validation, y_val_pred)
        linear_regression_test_R2 = r2_score(self.y_test, y_test_pred)
        self.output['train_score'] = linear_regression_train_R2
        self.output['validation_score'] = linear_regression_val_R2
        self.output['test_score'] = linear_regression_test_R2
        return self.output

# %%
model_1 = LinearRegressionModel()
# model_1.fit_linear_regression_model()
model_1.evaluate_linear_regression_model()
# %%
