import streamlit as st
import pandas as pd
from pycaret.regression import RegressionExperiment, compare_models


st.sidebar.file_uploader("Choose a CSV file", key='csv')

if st.session_state.csv != None:
    st.session_state.df = pd.read_csv(st.session_state.csv)
    st.session_state.target_name = st.selectbox('Choose target', st.session_state.df.columns)
    st.session_state.df.drop(st.multiselect('Choose feature to drop', st.session_state.df.columns), axis=1, inplace=True)
    st.experimental_data_editor(st.session_state.df, height=210)
    
    # st.session_state.model = RegressionExperiment()
    # st.session_state.model.setup(st.session_state.df, target = st.session_state.target_name, session_id = 123)
    # best = compare_models()

st.session_state