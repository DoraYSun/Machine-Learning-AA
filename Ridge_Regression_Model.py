import time
from AADatabase import AADatabase
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


class RidgeRegressionModel():
    """add l2 regularisation on linear regression, return fitting time, best hyperparameters, R2 scores on train/validation/test sets  """

    def __init__(self):
        """split data into train, validation, test sets"""
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = AADatabase().split_data(standardize_features=True)
        self.l2 = Ridge()
        self.output = {'model':'Ridge Regression'}
        self._evaluate_ridge_regression_model()
        

    def _ridge_regression_hyperparameters_tuning(self):
        """tuning hyperparameters of ridge regression model for best performance"""
        start_time = time.time()
        l2_parameters = {'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000],
                        'normalize':[True,False]
                        }
        
        l2_grid = GridSearchCV(estimator=self.l2, param_grid = l2_parameters)
        l2_grid.fit(self.X_validation, self.y_validation)
        self.best_params = l2_grid.best_params_
        end_time = time.time()
        self.output['time'] = end_time - start_time # time for finding the best model
        self.output['best_params'] = self.best_params
    
    def _evaluate_ridge_regression_model(self):
        """fit training data into ridge regression model and reflect how well lasso regression model performs on validation set"""
        self._ridge_regression_hyperparameters_tuning()
        l2_best = Ridge(alpha=self.best_params['alpha'], normalize=self.best_params['normalize']) # use best hyperparameters from tuning
        y_train_pred = l2_best.fit(self.X_train, self.y_train).predict(self.X_train)
        y_val_pred = l2_best.fit(self.X_validation, self.y_validation).predict(self.X_validation)
        y_test_pred = l2_best.fit(self.X_test, self.y_test).predict(self.X_test)
        ridge_regression_train_R2 = r2_score(self.y_train, y_train_pred)
        ridge_regression_val_R2 = r2_score(self.y_validation, y_val_pred)
        ridge_regression_test_R2 = r2_score(self.y_test, y_test_pred)
        self.output['train_score'] = ridge_regression_train_R2
        self.output['validation_score'] = ridge_regression_val_R2
        self.output['test_score'] = ridge_regression_test_R2
        return self.output
        
