import pandas as pd
import numpy as np
import os, dill
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score


    
class DataCleaner:
    def __init__(self, data):
        self.data = data
        
    def clean_data_trainer(self):
        self.data.drop(['Crime ID', 'LSOA name', 'Reported by', 'Location', 'LSOA code'], 
                       axis = 1, inplace = True, errors = 'ignore')
        
        if self.data['Month'] is not None:
            self.data['Month'] = pd.to_datetime(self.data['Month'])
        
        # self.data['Month'] = pd.to_datetime(self.data['Month'])
        
        self.data.loc[:, 'Outcome type'] = self.data.loc[:, 'Outcome type'].apply(lambda x: 'prosecuted' if 'charged' in x else 'not-prosecuted')
        
        outcome_map = {'not-prosecuted':0, 'prosecuted':1}
        
        self.data['Outcome type'] = self.data['Outcome type'].map(outcome_map)
        
        self.data['year'] = self.data.loc[:, 'Month'].dt.year
        self.data['month'] = self.data.loc[:, 'Month'].dt.month
        
        self.data.drop(['Month'], axis = 1, inplace = True, errors = 'ignore')
        
        return self.data


class LoadSaveObject:
    
    def load_object(file_path):
        with open(file_path, 'rb') as f:
            return dill.load(f)
        
    def save_object(file_path, file_object):
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            return dill.dump(file_object, f)
        

def save_object(file_path, file_object):
    dir_path=os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        return dill.dump(file_object, f)


# def evaluate_model(x_test, y_test, y_pred):

# def save_object(file_path, file_preprocessor):
#     dir_path=
