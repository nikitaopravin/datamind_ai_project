import streamlit as st
from streamlit_option_menu import option_menu
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os 
import pandas as pd
from pycaret.classification import *
from classification import prep_and_train
import time
from get_profile import get_profile
from classification import tuning
import numpy as np
st.set_page_config(layout="wide")

classif_dictionary = np.load('classif_dic.npy',allow_pickle='TRUE').item()


with st.sidebar: #Side bar 
    selected = option_menu(menu_title=None,options=["Home", 'Classification','Regression', 'Time Series'], 
        icons=['house', 'file-binary','graph-up','bezier2'], menu_icon="cast", default_index=0)
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        st.session_state.df = pd.read_csv(file, index_col=None)



if selected == 'Home':
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
            speedup = 'config_minimal.yaml'
        else:
            speedup= 'config_default.yaml'
        if st.button('Generate report'):
            try:
                st_profile_report(get_profile(st.session_state.df, speedup))
            except NameError:
                st.error('Please upload dataset first')

if selected == 'Classification':
    section = option_menu(None, ["Prep & Train",'Tune & Analyse','Predict'], 
    default_index=0,icons=['1-square','1-square','1-square'],orientation="horizontal")



    if section == 'Prep & Train':
        with st.container():
            col1, col2 = st.columns([3,1.5])
            with col2:
                st.subheader('Choose Parameters')
                try:
                    st.session_state.targ = st.selectbox('Choose target', st.session_state.df.columns)
                except AttributeError:
                    pass
                with st.expander('Params'):
                    model = st.multiselect('Choose model',
                                            ['lr','ridge','lda','et','nb','qda','rf','gbc','lightgbm','catboost','ada','dt','knn','dummy','svm'],
                                            help='Blablabla', format_func=lambda x: classif_dictionary.get(x))
                with st.expander('Paramss'):
                    st.write('aaa')
                with st.expander('Paramsss'):
                    st.write('aaaa')
                with st.expander('Paramss'):
                    st.write('aaa')                  
                if st.button('Try model'):
                    try:
                            st.session_state.best, st.session_state.model_info, st.session_state.metrics_info = prep_and_train(st.session_state.targ, st.session_state.df, model)
                            # save_model(st.session_state.best, 'dt_pipeline')
                            with col1:
                                st.subheader('Actual Model')
                                st.session_state.model_info_last = st.session_state.model_info
                                st.session_state.metrics_info_last = st.session_state.metrics_info
                                col1, col2 = st.columns([3.5,1.8])
                                with col1:
                                    st.dataframe(st.session_state.metrics_info)
                                with col2:
                                    st.dataframe(st.session_state.model_info)     
                    except ValueError:
                                st.error('Please choose target with binary labels')
                else:
                    try:
                        with col1:
                            st.subheader('Your last teached model')
                            col1, col2 = st.columns([3.5,1.8])
                            with col1:
                                st.dataframe(st.session_state.metrics_info_last)
                            with col2:
                                st.dataframe(st.session_state.model_info_last)
                    except AttributeError: 
                        st.error('Teach your first model')
        st.divider()
        with st.container():
            col1,col2 = st.columns([2,1])
            with col2:
                st.subheader('Choose parameters for plots')
                st.session_state.plot_params = st.multiselect('Choose model',
                                            ['pr','auc','threshold','confusion_matrix','rfe'],max_selections=3,
                                            help='Blablabla')
                if st.button('Plot'):
                    with col1:
                        plot_model(st.session_state.best, plot = st.session_state.plot_params[0], display_format='streamlit')
                        plot_model(st.session_state.best, plot = st.session_state.plot_params[1], display_format='streamlit')
                        plot_model(st.session_state.best, plot = st.session_state.plot_params[2], display_format='streamlit')   



    if section == 'Tune & Analyse':
        st.title('Choose parameters to tune your model')
        st.subheader('Current model')
        st.table(st.session_state.metrics_info_last.head(1))
        col1, col2 = st.columns([2,4])
        with col2:
            option = st.selectbox(
            'Choose the tuning engine',
            ('scikit-learn', 'optuna', 'scikit-optimize'))
            st.session_state.optimize = st.selectbox('Choose metric to optimize', ('Accuracy','AUC','F1'))
            st.session_state.iters = st.slider('n_estimators', 5, 20, 5, 1)  
            if st.button('Tune'):
                clf1 = setup(data = st.session_state.df, target = st.session_state.targ)
                st.session_state.tuned_dt = tune_model(estimator=st.session_state.best,n_iter=st.session_state.iters,choose_better=True,optimize=st.session_state.optimize)
                st.session_state.info_df = pull()
            with col1:
                try:
                    st.dataframe(st.session_state.info_df)
                    st.write('Last best params')
                    st.code(st.session_state.tuned_dt)
                except AttributeError:
                    pass


