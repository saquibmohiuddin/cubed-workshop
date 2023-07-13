import streamlit as st
import datetime
import tensorflow as tf
from tensorflow import keras
from keras.utils import custom_object_scope

from src.components.predict_pipeline import CustomData, PredictPipeline

import pandas as pd
import numpy as np

st.set_page_config(page_title = 'Education Cubed Workshop', layout = 'wide')
def main():
    st.title('Education Cubed Crime Data Workshop')
    # front-end elements of the web page
    
    # code for interactive elements of App
    st.subheader('Hello ! :smile: Welcome to the App.')
    with st.sidebar:
        st.header('Select crime inputs')
        
        falls_within=st.selectbox('Police Jurisdiction :round_pushpin:', [
        'West Yorkshire Police', 'West Midlands Police', 'Surrey Police',
       'Kent Police', 'Metropolitan Police Service',
       'North Yorkshire Police', 'Durham Constabulary',
       'Norfolk Constabulary', 'South Wales Police', 'Northumbria Police',
       'South Yorkshire Police', 'Northamptonshire Police',
       'Merseyside Police', 'Cheshire Constabulary',
       'Hampshire Constabulary', 'Bedfordshire Police',
       'Wiltshire Police', 'Suffolk Constabulary', 'Staffordshire Police',
       'Gwent Police', 'Sussex Police', 'Derbyshire Constabulary',
       'North Wales Police', 'Leicestershire Police',
       'Nottinghamshire Police', 'Warwickshire Police',
       'West Mercia Police', 'Cambridgeshire Constabulary',
       'Cleveland Police', 'Essex Police', 'Lancashire Constabulary',
       'Avon and Somerset Constabulary', 'Dyfed-Powys Police',
       'Humberside Police', 'Lincolnshire Police',
       'Hertfordshire Constabulary', 'Cumbria Constabulary',
       'Thames Valley Police', 'Devon & Cornwall Police', 'Dorset Police',
       'Gloucestershire Constabulary', 'City of London Police'
        ])
        
        crime_type=st.selectbox('Type of Crime ', [
        'Violence and sexual offences', 'Vehicle crime',
       'Theft from the person', 'Public order', 'Shoplifting', 'Drugs',
       'Burglary', 'Criminal damage and arson', 'Other theft',
       'Bicycle theft', 'Other crime', 'Possession of weapons', 'Robbery'
        ])
        
    col1, col2=st.columns(2)
    
    with col1:
        try:
            longitude=st.number_input('Enter longitude: ', value=-2.36)
        except:
            print('The selection is invalid')
            
        month = st.number_input('Enter the month of crime: ', min_value = 1, max_value = 12)
        
        
    with col2:
        try:
            latitude=st.number_input('Enter Latitude: ', value=51.38)
        except:
            print('The selection is invalid')
            
        year=st.number_input('Enter year of crime ', min_value=2020)
    
    map_data = {
        'longitude':longitude,
        'latitude':latitude
    }
            
    st.map(data=pd.DataFrame(map_data, index=[0]))
        

    
    
    
    result=''
    
    
    # making predictions 
    if st.button('Classifiy'):
        data=CustomData(falls_within=falls_within, longitude=longitude, latitude=latitude, 
                        crime_type=crime_type, year=year, month=month)
        
        result_df=data.get_dataframe()
        st.write(result_df)
        
        result_pipeline=PredictPipeline()
        prosecution_result=result_pipeline.predict(result_df)
        if prosecution_result == 1:
            st.subheader(f'The crime can be prosecuted')
        else:
            st.subheader(f'The crime cannot be prosecuted')
        
        
if __name__=='__main__':
    main()