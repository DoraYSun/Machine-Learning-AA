import time
from AADatabase import AADatabase
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# %%
class SVMModel():
    """find the best fit SVM model return fitting time, best hyperparameters, R2 scores on train/validation/test sets"""

    def __init__(self):
        """split data into train, validation, test sets"""
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = AADatabase().split_data(standardize_features=True)
        self.svc = svm.SVC()
        self.output = {'model':'SVM'}
        

    def _SVM_hyperparameters_turning(self):
        """turning hyperparameters of SVM model for best performance"""
        start_time = time.time()
        svm_parameters = {'C': (0.1, 1, 10, 100, 1000),
                        'gamma': ('scale', 'auto')
                        }
        
        svm_grid = GridSearchCV(estimator=self.svc, param_grid = svm_parameters)
        svm_grid.fit(self.X_validation, self.y_validation)
        self.best_params = svm_grid.best_params_
        end_time = time.time()
        self.output['time'] = end_time - start_time
        self.output['best_params'] = self.best_params
    
    def evaluate_SVM_model(self):
        """fit training data into SVM model and reflect how well lasso regression model performs on validation set"""
        self._SVM_hyperparameters_turning()
        svm_best = svm.SVC(C=self.best_params['C'], kernel='linear', gamma=self.best_params['gamma'])
        y_train_pred = svm_best.fit(self.X_train, self.y_train).predict(self.X_train)
        y_val_pred = svm_best.fit(self.X_validation, self.y_validation).predict(self.X_validation)
        y_test_pred = svm_best.fit(self.X_test, self.y_test).predict(self.X_test)
        svm_train_R2 = r2_score(self.y_train, y_train_pred)
        svm_val_R2 = r2_score(self.y_validation, y_val_pred)
        svm_test_R2 = r2_score(self.y_test, y_test_pred)
        self.output['train_score'] = svm_train_R2
        self.output['validation_score'] = svm_val_R2
        self.output['test_score'] = svm_test_R2
        return self.output
    
# %%
model_svm = SVMModel()
# model_1.fit_linear_regression_model()
model_svm.evaluate_SVM_model()