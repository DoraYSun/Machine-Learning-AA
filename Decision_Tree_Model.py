import time
from AADatabase import AADatabase
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# %%
class DecisionTreeModel():
    """find the best fit Decision Tree model return fitting time, best hyperparameters, R2 scores on train/validation/test sets"""

    def __init__(self):
        """split data into train, validation, test sets"""
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = AADatabase().split_data(standardize_features=True)
        self.dt = DecisionTreeRegressor()
        self.output = {'model':'Decision Tree'}
        

    def _Decision_Tree_hyperparameters_turning(self):
        """turning hyperparameters of Decision Tree model for best performance"""
        start_time = time.time()
        dt_parameters = {'splitter': ('best', 'random'),
                         'max_depth' : [1, 3, 5, 7, 9, 11, 12]
                        }
        
        dt_grid = GridSearchCV(estimator=self.dt, param_grid = dt_parameters)
        dt_grid.fit(self.X_validation, self.y_validation)
        self.best_params = dt_grid.best_params_
        end_time = time.time()
        self.output['time'] = end_time - start_time
        self.output['best_params'] = self.best_params
    
    def evaluate_Decision_Tree_model(self):
        """fit training data into Decision Tree model and reflect how well lasso regression model performs on validation set"""
        self._Decision_Tree_hyperparameters_turning()
        dt_best = DecisionTreeRegressor(splitter=self.best_params['splitter'], max_depth=self.best_params['max_depth'])
        y_train_pred = dt_best.fit(self.X_train, self.y_train).predict(self.X_train)
        y_val_pred = dt_best.fit(self.X_validation, self.y_validation).predict(self.X_validation)
        y_test_pred = dt_best.fit(self.X_test, self.y_test).predict(self.X_test)
        dt_train_R2 = r2_score(self.y_train, y_train_pred)
        dt_val_R2 = r2_score(self.y_validation, y_val_pred)
        dt_test_R2 = r2_score(self.y_test, y_test_pred)
        self.output['train_score'] = dt_train_R2
        self.output['validation_score'] = dt_val_R2
        self.output['test_score'] = dt_test_R2
        return self.output
    
# %%
model_dt = DecisionTreeModel()
# model_1.fit_linear_regression_model()
model_dt.evaluate_Decision_Tree_model()
# %%
