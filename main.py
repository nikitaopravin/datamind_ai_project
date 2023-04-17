import streamlit as st
from streamlit_option_menu import option_menu
# from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# import os
from utils.clf_funcs import pretrain_clf
# import test_clf 
import pandas as pd
from pycaret.classification import *
# from utils.classification import prep_and_train_clf
# from utils.regression import prep_and_train_regr
from datetime import datetime
from utils.get_profile import get_profile
# from utils.classification import tuning_clf, pred_clf
# from utils.regression import tuning_regr, setup_regr, pred_regr
from utils.clf_funcs import *
from utils.regr_funs import *
import numpy as np
# from utils.plot_regr import plot_graph_regr
# from utils.plot_clf import plot_graph_clf
st.set_page_config(layout="wide")

classif_dictionary = np.load('data/classif_dic.npy',allow_pickle='TRUE').item()
regr_dic = np.load('data/regr_dic.npy',allow_pickle='TRUE').item()


with st.sidebar: #Side bar 
    selected = option_menu(menu_title=None,options=["Home", 'Classification','Regression', 'Time Series'], 
        icons=['house', 'file-binary','graph-up','bezier2'], menu_icon="cast", default_index=0)
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        st.session_state.df = pd.read_csv(file, index_col=None)
    

if selected == 'Home':
    with st.sidebar:
        if st.button('Load Classification example'):
            st.session_state.df = pd.read_csv('examples/Titanic.csv')
            st.write('Titanic.csv loaded')
        if st.button('Load regression example'):
            st.session_state.df = pd.read_csv('examples/House_prices.csv')
            st.write('House_pricing.csv loaded')

    nones = ['None' for i in range(5)]
    section= option_menu(None, ["Info", "Data profile",'AutoML'], 
    default_index=0, icons=nones,orientation="horizontal")

    if section== 'Info':
        st.title('Main info about service')
        st.write('Some Text about service')

    if section == 'Data profile':
        st.title('This section will give you main information about uploaded dataset')
        st.write('Simply click "Generate new profile" if you want to generate new profile data and click "View old report to load previous profile"')
        if st.checkbox('Huge Dataset'):
            speedup = 'data/config_minimal.yaml'
        else:
            speedup= 'data/config_default.yaml'
        try:
            st.session_state.df
        except AttributeError:
            st.warning('Please Load Dataset')
        if st.button('Generate report'):
            try:
                report = get_profile(st.session_state.df, speedup)
                export=report.to_html()
                st.download_button(label="Download Full Report", data=export, file_name=f'report-{datetime.now().strftime("%Y_%m_%d")}.html')
                st_profile_report(report)
            except NameError:
                st.error('Please upload dataset first')



if selected == 'Classification':
    with st.sidebar:
        if st.button('Load Classification example'):
            st.session_state.df = pd.read_csv('examples/Titanic.csv')
            st.write('Titanic.csv loaded')
    section = option_menu(None, ["Prep & Train",'Tune & Analyse','Predict'], 
    default_index=0,icons=['1-square','1-square','1-square'],orientation="horizontal")

    if section == 'Prep & Train':
        pretrain_clf(classif_dictionary)

    if section == 'Tune & Analyse':
        tunalyse_clf()

    if section == 'Predict':
        predict()


            
if selected == 'Regression':
    with st.sidebar:
        if st.button('Load regression example'):
            st.session_state.df = pd.read_csv('examples/House_prices.csv')
            st.write('House_pricing.csv loaded') 
    section = option_menu(None, ["Prep & Train",'Tune & Analyse','Predict'], 
    default_index=0,icons=['1-square','1-square','1-square'],orientation="horizontal")

    if section == 'Prep & Train':
        pretrain_regr(regr_dic)

    if section == 'Tune & Analyse':
        tunalyse_regr()
              
    if section == 'Predict':
        predict_regr()

        




