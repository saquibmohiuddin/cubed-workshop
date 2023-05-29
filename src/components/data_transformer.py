import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
from src.utils import DataCleaner, LoadSaveObject, save_object


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
                ('numerical_pipeline', numerical_pipeline, numerical_cols)
            ],
            n_jobs=-1, remainder='passthrough'
        )
        
        return preprocessor
    
    
    def initiate_features_transformation(self):
        train_path:str=DataIngestion.train_data_path
        train_df=pd.read_csv(train_path)
        
        train_cleaner = DataCleaner(data=train_df)
        train_df_clean = train_cleaner.clean_data_trainer()
        
        
        test_path:str=DataIngestion.test_data_path
        test_df = pd.read_csv(test_path)
        
        test_cleaner = DataCleaner(data=test_df)
        test_df_clean = test_cleaner.clean_data_trainer()
        
        
        preprocessor = self.get_transformer_object()
        
        x_train_arr = preprocessor.fit_transform(train_df_clean).toarray()
        x_test_arr = preprocessor.transform(test_df_clean).toarray()
        
        # load_save_object = LoadSaveObject()
        
        # load_save_object.save_object(file_path=DataTransformation.preprocessor_path,
        #                              file_object=preprocessor)
        
        save_object(file_path=DataTransformation.preprocessor_path,
                    file_object=preprocessor)
        
        return x_train_arr, x_test_arr
    
    
# if __name__=='__main__':
#     transformation_object=DataTransformation()
#     train_arr, test_arr = transformation_object.initiate_features_transformation()
#     print(train_arr, test_arr)
    
        
        
        
        
        
        
