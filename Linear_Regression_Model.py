# %%
import time
from AADatabase import AADatabase
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


class LinearRegressionModel():
    """find the best fit linear regression model for AA dataset, return fitting time, best hyperparameters, R2 scores on train/validation/test sets"""

    def __init__(self):
        """split data into train, validation, test sets"""
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = AADatabase().split_data(standardize_features=True)
        self.lr = LinearRegression()
        self.output = {'model':'Linear Regression'}
        self._evaluate_linear_regression_model()


    def _linear_regression_hyperparameters_tuning(self):
        """tuning hyperparameters of linear regression model for best performance"""
        start_time = time.time()
        lr_parameters = {'fit_intercept': [True, False], 
                    'normalize': [True, False]
                    }
        
        lr_grid = GridSearchCV(estimator=self.lr, param_grid = lr_parameters)
        lr_grid.fit(self.X_validation, self.y_validation)
        self.best_params = lr_grid.best_params_
        end_time = time.time()
        self.output['time'] = end_time - start_time # time for finding the best model
        self.output['best_params'] = self.best_params
    
    def _evaluate_linear_regression_model(self):
        """fit training data into linear regression model and reflect how well linear regression model performs on validation set"""
        self._linear_regression_hyperparameters_tuning()
        lr_best = LinearRegression(fit_intercept=self.best_params['fit_intercept'], normalize=self.best_params['normalize']) # use best hyperparameters from tuning
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

