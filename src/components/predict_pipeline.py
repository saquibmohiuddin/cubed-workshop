import pandas as pd
import numpy as np
import os, dill

from src.utils import DataCleaner, LoadSaveObject, f1_metric
from src.components.data_transformer import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer_tensorflow import TFModelTrainer
from keras.utils import custom_object_scope


import tensorflow as tf
from tensorflow import keras

class PredictPipeline:
    def __init__(self) -> int:
        pass
    
    def predict(self, features):
        preprocessor_path:str=DataTransformation.preprocessor_path
        model_path=TFModelTrainer.tf_model_path
        
        
        preprocessor = LoadSaveObject().load_object(preprocessor_path)
        
        classifier = tf.keras.models.load_model('artifacts/model/model_tf.h5', custom_objects={'f1_metric':f1_metric})
        
        X_inputs = preprocessor.transform(features)
        
        result = np.round(classifier.predict(X_inputs))[0][0].astype(int)
        
        return result
        
        
class CustomData:
    def __init__(self, falls_within:str, longitude:float, latitude:float,
                 crime_type:str, year:int, month:int) -> None:
        
        self.falls_within = falls_within
        self.longitude = longitude
        self.latitude = latitude
        self.crime_type = crime_type
        self.year = year
        self.month = month
        
    def get_dataframe(self):
        data_input_dict = {
            'Falls_within':self.falls_within,
            'Longitude':self.longitude,
            'Latitude':self.latitude,
            'Crime_type':self.crime_type,
            'year':self.year,
            'month':self.month
        }
        
        return pd.DataFrame(data_input_dict, index=[0])

