import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
from src.utils import DataCleaner


ARTIFACTS_DIR = 'artifacts'

@dataclass
class DataTransformationConfig:
    preprocessor_path:str=os.path.join(ARTIFACTS_DIR, 'preprocessor', 'preprocessor.pkl')
    
class DataTransformation(DataTransformationConfig):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    
    def get_transformer_object(self):
        categorical_cols = ['Falls within', 'Crime type']
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
                ('numerical_pipeline', numerical_cols, numerical_cols)
            ],
            n_jobs=-1, remainder='passthrough'
        )
        
        
