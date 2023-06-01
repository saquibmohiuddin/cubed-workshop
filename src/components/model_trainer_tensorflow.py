import pandas as pd
import numpy as np
import os, dill

from src.utils import LoadSaveObject, evaluate_model, my_model, F1Score, f1_metric
from src.components.data_transformer import DataTransformation
from src.components.data_ingestion import DataIngestion
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
from keras.utils import custom_object_scope


from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import datetime, os

from dataclasses import dataclass

# keras.utils.custom_object_scope['f1_metric'] = f1_metric




MODEL_DIR = 'artifacts\model'

@dataclass
class TFModelTrainerConfig:
    tf_model_path:str=os.path.join(MODEL_DIR, 'model_tf.h5')
    
class TFModelTrainer(TFModelTrainerConfig):
    def __init_subclass__(cls) -> None:
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
        
        class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        model = my_model()
        
        es = EarlyStopping(monitor = "val_f1_metric", mode = "max", min_delta= 0.0001, patience = 5, verbose=1)
        
        mc = ModelCheckpoint(filepath="artifacts/weights", 
                             monitor="val_f1_metric", verbose=1, save_best_only= True, mode="max")
        
        
        history = model.fit(X_train_arr, y_train, epochs= 200, validation_data = (X_test_arr, y_test), 
                    callbacks = [es,mc], class_weight = class_weight_dict)
        
        print(history)
        
        acc = model.evaluate(X_test_arr, y_test, verbose = 1)
        print(acc)
        
        model.save(TFModelTrainer.tf_model_path)
        
        return TFModelTrainer.tf_model_path
    
if __name__=='__main__':
    tf_model_trainer = TFModelTrainer()
    X_train_arr, X_test_arr, y_train, y_test = tf_model_trainer.data_features_transformation(preprocessor_path=DataTransformation.preprocessor_path,
                                                                                          train_path=DataIngestion.train_data_path,
                                                                                          test_path=DataIngestion.test_data_path)
    
    score = tf_model_trainer.initiate_model_training(X_train_arr, X_test_arr, y_train, y_test)
    print(score)
    
    
    
