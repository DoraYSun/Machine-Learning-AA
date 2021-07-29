#%%
import pandas as pd
import numpy as np
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
        X = Feature
        y = car_df['price'].values
        
        #return lable and features
        if return_X_y == True:
            return X, y
        else:
            return car_df
#%%
if __name__ == '__main__':
    data = AADatabase()
    data.feature_selector(return_X_y=True)


# %%
