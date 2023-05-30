import pandas as pd
import numpy as np
import os, dill


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils import LoadSaveObject, evaluate_model
from src.components.data_transformer import DataTransformation
from src.components.data_ingestion import DataIngestion

from dataclasses import dataclass


MODEL_DIR = 'artifacts\model'

@dataclass
class ModelTrainerConfig:
    model_path:str=os.path.join(MODEL_DIR, 'model.pkl')
    
class ModelTrainer(ModelTrainerConfig):
    def __init_subclass__(cls) -> str:
        return super().__init_subclass__()
    
    def data_features_transformation(self, preprocessor_path, train_path, test_path):
        train_df = pd.read_csv(train_path)
        X_train = train_df.drop(['Outcome type'], axis = 1)
        y_train = train_df['Outcome type'].copy()
        
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop(['Outcome type'], axis = 1)
        y_test = test_df['Outcome type'].copy()
        
        preprocessor = LoadSaveObject().load_object(preprocessor_path)
        
        X_train_arr = preprocessor.fit_transform(X_train).toarray()
        X_test_arr = preprocessor.transform(X_test).toarray()
        
        return X_train_arr, X_test_arr, y_train, y_test
    
    def initiate_model_training(self):
        
        X_train_arr, X_test_arr, y_train, y_test = self.data_features_transformation(
            preprocessor_path=DataTransformation.preprocessor_path,
            train_path=DataIngestion.train_data_path,
            test_path=DataIngestion.test_data_path
        )
        
        models = {
            'RandomForestClassifier':RandomForestClassifier(n_jobs=-1),
            'LogisticRegression':LogisticRegression(n_jobs=-1)
        }
        
        model_predictions = {}
        
        model_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        
        report = {}
        
        for model in models.keys():
            
            # fit model
            models[model].fit(X_train_arr, y_train)
            
            # perform precistions
            y_pred = models[model].predict(X_test_arr)
            # storing values in dictionary
            model_predictions['y_pred_'+model] = y_pred
            
            # evalauting model performance
            accuracy, precision, recall, f1 = evaluate_model(y_test=y_test, y_pred=y_pred)
            
            
            
        
        
        