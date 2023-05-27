import pandas as pd
import numpy as np


    
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
    
        
        
        
        
        
        
        