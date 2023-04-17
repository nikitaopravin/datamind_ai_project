from pycaret.regression import *
from utils.plot_regr import plot_graph_regr
from utils.regression import prep_and_train_regr, tuning_regr, pred_regr
import streamlit as st
import pandas as pd 

def pretrain_regr(regr_dic):
    with st.container():
            col1, col2 = st.columns([3,1.5])
            with col2:
                st.subheader('Choose Parameters', anchor=False)
                try:
                    st.session_state.targ_regr = st.selectbox('Choose target', st.session_state.df.columns)
                except AttributeError:
                    pass
                with st.expander('Params'):
                    st.session_state.model_regr = st.multiselect('Choose model',
                                            ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par','ransac', 'tr',
                                              'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada',
                                                'gbr', 'mlp', 'xgboost', 'lightgbm', 'dummy'],
                                            help='Blablabla', format_func=lambda x: regr_dic.get(x))
                    train_size = st.number_input('Training Size:', value=0.7)
                    data_split_stratify = st.checkbox("Controls Stratification during Split", value=False)
                    fold_strategy = st.selectbox('Choice of Cross Validation Strategy',options=['kfold','stratifiedkfold'])
                    fold = st.number_input('Number of Folds to be Used in Cross Validation',min_value=2,value=10)
                    remove_outliers = st.checkbox('Remove outliers', value=False)
                with st.expander('Inputation and Normalisation'):
                    numeric_imputation = st.selectbox('Missing Value for Numeric Columns', options=['mean','median','mode'])
                    # select numberical features preprocessing
                    normalize = st.checkbox('Normalization', value=False)
                    normalize_method = 'zscore'
                    if normalize:
                        normalize_method = st.selectbox('Method to be used for Normalization',options=['zscore','minmax','maxabs','robust'])
                    # fix_imbalance = st.checkbox('Fix Imbalance of Target Classes',value=False)    
                with st.expander('Feature selection'):
                    feature_selection = st.checkbox('Select a Subset of Features Using a Combination of various Permutation Importance', value=False)
                    feature_selection_method = 'classic'
                    if feature_selection:
                        feature_selection_method= st.selectbox('Algorithm for feature selection',options=['classic','univariate','sequential'])
                    remove_multicollinearity = st.checkbox('Remove Highly Linearly Correlated Features', value=False)
                    multicollinearity_threshold = 0.9
                    if remove_multicollinearity:
                        multicollinearity_threshold = st.number_input('Threshold Used for Dropping the Correlated Features', min_value=0.0, value=0.9)
                with col1.expander('Dataset'):
                    try:
                        st.dataframe(st.session_state.df)
                    except AttributeError:
                        pass
                if st.button('Try model'):
                    try:
                            # clf1 = setup_regr(data = st.session_state.df, target = st.session_state.targ_regr, session_id=2)
                            st.session_state.best_regr, st.session_state.model_info_regr, st.session_state.metrics_info_regr = prep_and_train_regr(
                                st.session_state.targ_regr, st.session_state.df, st.session_state.model_regr, 
                                train_size, data_split_stratify, fold_strategy, fold, numeric_imputation,
                                normalize,normalize_method,feature_selection,feature_selection_method,
                                remove_multicollinearity,multicollinearity_threshold, remove_outliers
                                )
                            
                            # save_model(st.session_state.best, 'dt_pipeline')
                            with col1:
                                st.subheader('Actual Model')
                                st.session_state.model_info_last_regr = st.session_state.model_info_regr
                                st.session_state.metrics_info_last_regr = st.session_state.metrics_info_regr
                                col1, col2 = st.columns([3.5,1.8])
                                with col1:
                                    st.dataframe(st.session_state.metrics_info_regr)
                                with col2:
                                    st.dataframe(st.session_state.model_info_regr)     
                    except ValueError:
                                st.error('Please choose target with binary labels')
                else:
                    try:
                        with col1:
                            st.subheader('Your last teached model')
                            col1, col2 = st.columns([3.5,1.8])
                            with col1:
                                st.dataframe(st.session_state.metrics_info_last_regr)
                            with col2:
                                st.dataframe(st.session_state.model_info_last_regr)
                    except AttributeError: 
                        st.error('Teach your first model')
    st.divider()
    with st.container():
        col1,col2 = st.columns([2,1])
        with col2:
            st.subheader('Choose parameters for plots')
            st.session_state.plot_params_regr = st.multiselect('Choose model',
                                        ['residuals','error','cooks'],
                                        help='Blablabla')
            if st.button('Plot'):
                with col1:
                    plot_graph_regr(st.session_state.best_regr, st.session_state.plot_params_regr)

def tunalyse_regr():
    st.title('Choose parameters to tune your model')
    st.subheader('Current model')
    try:
        st.table(st.session_state.metrics_info_last_regr.head(1))
    except AttributeError:
        pass
    col1, col2 = st.columns([2,4])
    with col2:
        option_regr = st.selectbox(
        'Choose the tuning engines',
        ('scikit-learn', 'optuna', 'scikit-optimize'))
        optimizer_regr = st.selectbox('Choose metric to optimize', ('MAE','MSE','RMSE'))
        st.session_state.iters_regr = st.slider('n_estimators', 5, 20, 5, 1)  
        if st.button('Tune'):
            # clf2 = setup(data = st.session_state.df, target = st.session_state.targ_regr, session_id=2)
            st.session_state.tuned_dt_regr, st.session_state.info_df_regr = tuning_regr(model=st.session_state.best_regr,n_iters=st.session_state.iters_regr,opti=optimizer_regr, search_lib=option_regr)
            st.write('Last best params')
            st.code(st.session_state.tuned_dt_regr)
        with col1:
            try:
                st.dataframe(st.session_state.info_df_regr)
                # st.write('Last best params')
                # st.code(st.session_state.tuned_dt)
            except (AttributeError, NameError):
                st.warning('Prepare and train model first')

def predict_regr():
    try:
        holdout_pred = pred_regr(st.session_state.best_regr)
    except AttributeError:
        st.warning('Teach model first')
    file_res = st.file_uploader("Upload Your Datasets")
    if file_res: 
        test_df = pd.read_csv(file_res, index_col=None)
    if st.button('predict on test data'):
        try:
            st.dataframe(holdout_pred)
        except UnboundLocalError:
            pass
    if st.button('predict for result'):
        result_pred = pred_regr(st.session_state.best_regr, data=test_df)
        result = result_pred
        result.to_csv('result.csv')
        st.dataframe(result_pred) 