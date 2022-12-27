from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 


with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Auto ML Regression APP")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download", "Predict"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
   
if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download":
    st.write("Download best model for future purpose")
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")
        
if choice == "Predict":
    st.title("Upload Your Test Dataset")
    test_file = st.file_uploader("Upload Your Test Dataset")
    if test_file: 
        test_df = pd.read_csv(test_file, index_col=None)
        st.dataframe(test_df)
        model = load_model('best_model')
        st.write(model.predict(test_df))