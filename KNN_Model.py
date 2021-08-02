import time
from AADatabase import AADatabase
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# %%
class KNNModel():
    """add l2 regularisation on linear regression, return fitting time, best hyperparameters, R2 scores on train/validation/test sets  """

    def __init__(self):
        """split data into train, validation, test sets"""
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = AADatabase().split_data(standardize_features=True)
        self.knn = KNeighborsRegressor()
        self.output = {'model':'KNN'}
        

    def _KNN_hyperparameters_turning(self):
        """turning hyperparameters of ridge regression model for best performance"""
        start_time = time.time()
        knn_parameters = [{'n_neighbors': list(range(1, 30)),
                        'weights':['uniform', 'distance']
                        }]
        
        knn_grid = GridSearchCV(estimator=self.knn, param_grid = knn_parameters)
        knn_grid.fit(self.X_validation, self.y_validation)
        self.best_params = knn_grid.best_params_
        end_time = time.time()
        self.output['time'] = end_time - start_time
        self.output['best_params'] = self.best_params
    
    def evaluate_KNN_model(self):
        """fit training data into ridge regression model and reflect how well lasso regression model performs on validation set"""
        self._KNN_hyperparameters_turning()
        knn_best = KNeighborsRegressor(n_neighbors=self.best_params['n_neighbors'], weights=self.best_params['weights'])
        y_train_pred = knn_best.fit(self.X_train, self.y_train).predict(self.X_train)
        y_val_pred = knn_best.fit(self.X_validation, self.y_validation).predict(self.X_validation)
        y_test_pred = knn_best.fit(self.X_test, self.y_test).predict(self.X_test)
        knn_train_R2 = r2_score(self.y_train, y_train_pred)
        knn_val_R2 = r2_score(self.y_validation, y_val_pred)
        knn_test_R2 = r2_score(self.y_test, y_test_pred)
        self.output['train_score'] = knn_train_R2
        self.output['validation_score'] = knn_val_R2
        self.output['test_score'] = knn_test_R2
        return self.output
    
# %%
model_knn = KNNModel()
# model_1.fit_linear_regression_model()
model_knn.evaluate_KNN_model()
# %%
