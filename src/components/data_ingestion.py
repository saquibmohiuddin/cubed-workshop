import pandas as pd 
from dataclasses import dataclass
import os
from sklearn.model_selection import train_test_split
from src.utils import DataCleaner


DATA_DIR = 'artifacts\data'

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join('clean_data', 'clean_data_v2.csv')
    train_data_path:str=os.path.join(DATA_DIR, 'train.csv')
    test_data_path:str=os.path.join(DATA_DIR, 'test.csv')
    
class DataIngestion(DataIngestionConfig):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    
    def initiate_data_ingestion(self):
        df = pd.read_csv(DataIngestion.raw_data_path)
        
        data_cleaner = DataCleaner(data=df)
        df = data_cleaner.clean_data_trainer()
        
        # train test splits
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        
        train_set.to_csv(DataIngestion.train_data_path, index=False, header=True)
        test_set.to_csv(DataIngestion.test_data_path, index=False, header=True)
        
        return (DataIngestion.train_data_path, DataIngestion.test_data_path)
        
     
if __name__=='__main__':
    ingestion=DataIngestion()
    train_data, test_data=ingestion.initiate_data_ingestion()
    print(train_data, test_data)