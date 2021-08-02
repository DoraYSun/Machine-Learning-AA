#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%
class AADatabase():

    def __init__(self):
        self.df = pd.read_csv('Pre-processed Data.csv', index_col=0)
    
    def feature_selector(self, return_X_y):
        """Select lable and features from AA database""" 
        # one hot encoding catigorical features
        dummy_df = self.df[['fuel_type', 'transmission', 'body_type', 'car_tier', 'engine_tier', 'year_band']] 
        Feature = pd.concat([self.df['mileage'], pd.get_dummies(dummy_df)], axis=1)

        # add label and features to dataframe
        car_df = pd.concat([self.df['price'], Feature], axis=1)

        # feature selection
        self.X = Feature
        self.y = car_df['price'].values
        
        #return lable and features
        if return_X_y == True:
            return self.X, self.y
        else:
            return car_df
    
    
    def split_data(self, standardize_features, random_seed=27):
        """split data into train, validation, test sets"""
        # set one seed for all model
        rd_seed = np.random.seed(seed=random_seed)
        X, y = self.feature_selector(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rd_seed)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.3, random_state=rd_seed)

        # standerdize features in dataset
        if standardize_features == True:
            scaler = StandardScaler()
            X_train = scaler.fit(X_train).transform(X_train)
            X_validation = scaler.fit(X_validation).transform(X_validation)
            X_test = scaler.fit(X_test).transform(X_test)

        return X_train, X_test, X_validation, y_train, y_test, y_validation
      
#%%
if __name__ == '__main__':
    data = AADatabase()
    X_train, X_test, X_validation, y_train, y_test, y_validation = data.split_data(standardize_features = True)
    print(X_train, X_test, X_validation, y_train, y_test, y_validation)



# %%
