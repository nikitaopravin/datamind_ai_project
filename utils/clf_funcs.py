import streamlit as st
from pycaret.classification import *
from utils.classification import *
from streamlit_option_menu import option_menu
import pandas as pd
from utils.classification import tuning_clf
from utils.plot_clf import plot_graph_clf


def pretrain_clf(classif_dictionary):
        with st.container():
            col1, col2 = st.columns([3,1.5])
            with col2:
                st.subheader('Choose Parameters', anchor=False)
                try:
                    st.session_state.targ_clf = st.selectbox('Choose target', st.session_state.df.columns)
                except AttributeError:
                    pass
                with st.expander('Params'):
                    st.session_state.model_clf = st.multiselect('Choose model',
                                            ['lr','ridge','lda','et','nb','qda','rf','gbc','lightgbm','catboost','ada','dt','knn','dummy','svm'],
                                            help='Blablabla', format_func=lambda x: classif_dictionary.get(x))
                    train_size = st.number_input('Training Size:', value=0.7)
                    data_split_stratify = st.checkbox("Controls Stratification during Split", value=False)
                    fold_strategy = st.selectbox('Choice of Cross Validation Strategy',options=['kfold','stratifiedkfold'])
                    fold = st.number_input('Number of Folds to be Used in Cross Validation',min_value=2,value=10)
                with st.expander('Inputation and Normalisation'):
                    numeric_imputation = st.selectbox('Missing Value for Numeric Columns', options=['mean','median','mode'])
                    # select numberical features preprocessing
                    normalize = st.checkbox('Normalization', value=False)
                    normalize_method = 'zscore'
                    if normalize:
                        normalize_method = st.selectbox('Method to be used for Normalization',options=['zscore','minmax','maxabs','robust'])
                    fix_imbalance = st.checkbox('Fix Imbalance of Target Classes',value=False)    
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
                            st.session_state.best_clf = None
                            st.session_state.best_clf, st.session_state.metrics_info_clf, st.session_state.model_info_clf = prep_and_train_clf(
                                st.session_state.targ_clf, st.session_state.df, st.session_state.model_clf, 
                                train_size, data_split_stratify, fold_strategy, fold, numeric_imputation,
                                normalize,normalize_method,fix_imbalance,
                                feature_selection,feature_selection_method,
                                remove_multicollinearity,multicollinearity_threshold
                                )
                            with col1:
                                st.subheader('Actual Model')
                                st.session_state.model_info_last_clf = st.session_state.model_info_clf
                                st.session_state.metrics_info_last_clf = st.session_state.metrics_info_clf
                                col1, col2 = st.columns([3.5,1.8])
                                with col1:
                                    st.dataframe(st.session_state.metrics_info_clf)
                                with col2:
                                    st.dataframe(st.session_state.model_info_clf)     
                    except ValueError:
                                st.error('Please choose target with binary labels')
                else:
                    try:
                        with col1:
                            st.subheader('Your last teached model')
                            col1, col2 = st.columns([3.5,1.8])
                            with col1:
                                st.dataframe(st.session_state.metrics_info_last_clf)
                            with col2:
                                st.dataframe(st.session_state.model_info_last_clf)
                    except AttributeError: 
                        st.error('Teach your first model')
        st.divider()
        with st.container():
            col1,col2 = st.columns([2,1])
            with col2:
                st.subheader('Choose parameters for plots')
                st.session_state.plot_params_clf = st.multiselect('Choose model',
                                            ['pr','auc','threshold','confusion_matrix','rfe'],max_selections=3,
                                            help='Blablabla')
                if st.button('Plot'):
                    with col1:
                        plot_graph_clf(st.session_state.best_clf, st.session_state.plot_params_clf) 

def tunalyse_clf():
    st.title('Choose parameters to tune your model')
    st.subheader('Current model')
    try:
        st.table(st.session_state.metrics_info_last_clf.head(1))
    except AttributeError:
        st.warning('Teach model first')    
    col1, col2 = st.columns([2,4])
    with col2:
        option = st.selectbox(
        'Choose the tuning engine',
        ('scikit-learn', 'optuna', 'scikit-optimize'))
        optimizer = st.selectbox('Choose metric to optimizee', ('Accuracy','AUC','F1'))
        st.session_state.iters_clf = st.slider('n_estimators', 5, 20, 5, 1)  
        if st.button('Tune'):
            # clf1 = setup(data = st.session_state.df, target = st.session_state.targ_clf, session_id=1)
            st.session_state.tuned_dt_clf = tuning_clf(model=st.session_state.best_clf,n_iters=st.session_state.iters_clf,search_lib=option,opti=optimizer)
            st.session_state.info_df_clf = pull()
            st.write('Last best params')
            st.code(st.session_state.tuned_dt_clf)
        with col1:
            try:
                st.dataframe(st.session_state.info_df_clf)
                # st.write('Last best params')
                # st.code(st.session_state.tuned_dt)
            except AttributeError:
                pass

def predict():
    try:
        holdout_pred = pred_clf(st.session_state.best_clf)
    except AttributeError:
        st.warning('Teach model first')
    file_res = st.file_uploader("Upload Your Datasetss")
    if file_res: 
        test_df = pd.read_csv(file_res, index_col=None)
    if st.button('predict on test data'):
        st.dataframe(holdout_pred)
    if st.button('predict for result'):
        try:
            result_pred = pred_clf(st.session_state.best_clf, data=test_df)
            result_pred.to_csv('result.csv')
            st.dataframe(result_pred) 
        except UnboundLocalError:
            st.warning('Please load dataset with unseen data')
    # st.image('streamlit-main-2023-04-14-21-04-62.gif')


