import time
from AADatabase import AADatabase
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# %%
class LassoRegressionModel():
    """add l1 regularisation on linear regression, return fitting time, best hyperparameters, R2 scores on train/validation/test sets  """

    def __init__(self):
        """split data into train, validation, test sets"""
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = AADatabase().split_data(standardize_features=True)
        self.l1 = Lasso()
        self.output = {'model':'Lasso Regression'}
        

    def _lasso_regression_hyperparameters_turning(self):
        """turning hyperparameters of lasso regression model for best performance"""
        start_time = time.time()
        l1_parameters = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000],
                        'normalize':[True,False]
                        }]
        
        l1_grid = GridSearchCV(estimator=self.l1, param_grid = l1_parameters)
        l1_grid.fit(self.X_validation, self.y_validation)
        self.best_params = l1_grid.best_params_
        end_time = time.time()
        self.output['time'] = end_time - start_time
        self.output['best_params'] = self.best_params
    
    def evaluate_lasso_regression_model(self):
        """fit training data into lasso regression model and reflect how well lasso regression model performs on validation set"""
        self._lasso_regression_hyperparameters_turning()
        l1_best = Lasso(alpha=self.best_params['alpha'], normalize=self.best_params['normalize'])
        y_train_pred = l1_best.fit(self.X_train, self.y_train).predict(self.X_train)
        y_val_pred = l1_best.fit(self.X_validation, self.y_validation).predict(self.X_validation)
        y_test_pred = l1_best.fit(self.X_test, self.y_test).predict(self.X_test)
        lasso_regression_train_R2 = r2_score(self.y_train, y_train_pred)
        lasso_regression_val_R2 = r2_score(self.y_validation, y_val_pred)
        lasso_regression_test_R2 = r2_score(self.y_test, y_test_pred)
        self.output['train_score'] = lasso_regression_train_R2
        self.output['validation_score'] = lasso_regression_val_R2
        self.output['test_score'] = lasso_regression_test_R2
        return self.output

# %%
model_1 = LassoRegressionModel()
# model_1.fit_linear_regression_model()
model_1.evaluate_lasso_regression_model()