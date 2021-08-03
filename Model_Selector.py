# %%
import time
from Linear_Regression_Model import LinearRegressionModel
from Lasso_Regression_Model import LassoRegressionModel
from Ridge_Regression_Model import RidgeRegressionModel
from KNN_Model import KNNModel
from Decision_Tree_Model import DecisionTreeModel
from SVM_Model import SVMModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.externals import joblib
# %%
class ModelSelector():
    """find best model for AAdatabase to predict price and save the model"""

    def __init__(self):
        self.outputs = []
        self._get_models()
        self.best_model_info = self.outputs[0]
        self._save_best_model()
    
    def _get_models(self):
        """compare performance of each model and select the best"""
        main_start_time = time.time()
        models = [LinearRegressionModel(), LassoRegressionModel(), 
                 RidgeRegressionModel(), KNNModel(), SVMModel(), DecisionTreeModel()]
        for model in models:
            self.outputs.append(model.output)
        main_end_time = time.time()
        self.outputs = sorted(self.outputs, key = lambda i: i['test_score'], reverse=True)
        self.model_evaluation_time = main_end_time - main_start_time
        return self.outputs, self.model_evaluation_time

    def _save_best_model(self):
        """save the best performance model"""
        if self.outputs[0]['model'] == 'Linear Regression':
            best_model = LinearRegression(fit_intercept=self.best_model_info['best_params']['fit_intercept'], normalize=self.best_model_info['best_params']['normalize'])

        elif self.outputs[0]['model'] == 'Lasso Regression':
            best_model = Lasso(alpha=self.best_model_info['best_params']['alpha'], normalize=self.best_model_info['best_params']['normalize'])
        
        elif self.outputs[0]['model'] == 'Ridge Regression':
            best_model = Ridge(alpha=self.best_model_info['best_params']['alpha'], normalize=self.best_model_info['best_params']['normalize'])
        
        elif self.outputs[0]['model'] == 'KNN':
            best_model = KNeighborsRegressor(n_neighbors=self.best_model_info['best_params']['n_neighbors'], weights=self.best_model_info['best_params']['weights'])
            
        elif self.outputs[0]['model'] == 'Decision Tree':
            best_model = DecisionTreeRegressor(splitter=self.best_model_info['best_params']['splitter'], max_depth=self.best_model_info['best_params']['max_depth'])
        
        else:
            best_model = svm.SVC(C=self.best_model_info['best_params']['C'], kernel='linear', gamma=self.best_model_info['best_params']['gamma'])
        
        #save best model
        joblib.dump(best_model, 'best_AA_model.model')
        print('>>>>>>>>Best model has been saved.')

# %%
ModelSelector()
