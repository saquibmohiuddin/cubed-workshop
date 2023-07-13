import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass
from src.utils import LoadSaveObject
from src.components.data_ingestion import DataIngestion


ARTIFACTS_DIR = 'artifacts'

@dataclass
class DataTransformationConfig:
    preprocessor_path:str=os.path.join(ARTIFACTS_DIR, 'preprocessor', 'preprocessor.pkl')
    
class DataTransformation(DataTransformationConfig):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    
    def get_transformer_object(self):
        categorical_cols = ['Falls_within', 'Crime_type']
        numerical_cols = ['Longitude', 'Latitude', 'year', 'month']
        
        categorical_pipeline = Pipeline(
            steps=[
                ('imputation_cat', SimpleImputer(strategy='most_frequent')),
                ('LabelEncoding', OneHotEncoder(handle_unknown='ignore'))
            ]
        )
        
        numerical_pipeline = Pipeline(
            steps=[
                ('imputation_median', SimpleImputer(missing_values=np.nan, strategy='median')),
                ('Scaler', MinMaxScaler())
            ]
        )
        
        # Column transformer
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical_pipeline', categorical_pipeline, categorical_cols),
                ('numerical_pipeline', numerical_pipeline, numerical_cols)
            ],
            n_jobs=-1, remainder='passthrough'
        )
        
        return preprocessor
    
    
    def save_preprocessor_object(self):

        preprocessor = self.get_transformer_object()
        
        train_df = pd.read_csv(DataIngestion.train_data_path)
        X_train = train_df.drop(['Outcome_type'], axis = 1)
        
        preprocessor.fit_transform(X_train)

        load_save_object = LoadSaveObject()
        
        load_save_object.save_object(file_path=DataTransformation.preprocessor_path,
                                     file_object=preprocessor)
        
        
        return DataTransformation.preprocessor_path
    
    
if __name__=='__main__':
    transformation_object=DataTransformation()
    preprocessor_path = transformation_object.save_preprocessor_object()
    print(preprocessor_path)
    
        
        
        
        
        
        
