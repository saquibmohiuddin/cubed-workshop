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
    
    def initiate_model_training(self, X_train_arr, X_test_arr, y_train, y_test):
        
        # X_train_arr, X_test_arr, y_train, y_test = self.data_features_transformation(
        #     preprocessor_path=DataTransformation.preprocessor_path,
        #     train_path=DataIngestion.train_data_path,
        #     test_path=DataIngestion.test_data_path
        # )
        
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
        
        for model in models.keys():
            
            # fit model
            models[model].fit(X_train_arr, y_train)
            
            # perform precistions
            y_pred = models[model].predict(X_test_arr)
            # storing values in dictionary
            model_predictions['y_pred_'+model] = y_pred
            
            # evalauting model performance
            accuracy, precision, recall, f1 = evaluate_model(y_test=y_test, y_pred=y_pred)
            
            model_list.append(model)
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            
            
        report = pd.DataFrame(list(zip(model_list, accuracy_list, 
                                       precision_list, recall_list, f1_list)), 
                              columns=['model_name', 'accuracy', 'precision', 'recall', 'f1-score']).sort_values(by=['f1-score'], ascending=False)
        
        best_model_name = report['model_name'].iloc[0]
        best_model_score = report['f1-score'].iloc[0]
        
        best_model = models[best_model_name]
        
        LoadSaveObject().save_object(file_object=best_model,
                                     file_path=ModelTrainer.model_path)
        
        return f'The best model noted is {best_model_name} with F1-Score {best_model_score}'
        
if __name__=='__main__':
    model_trainer = ModelTrainer()
    X_train_arr, X_test_arr, y_train, y_test = model_trainer.data_features_transformation(preprocessor_path=DataTransformation.preprocessor_path,
                                                                                          train_path=DataIngestion.train_data_path,
                                                                                          test_path=DataIngestion.test_data_path)
    
    score = model_trainer.initiate_model_training(X_train_arr, X_test_arr, y_train, y_test)
    print(score)
    
    
        
        
        
            
            
            
            
            
            
            
            
            
            
        
        
        