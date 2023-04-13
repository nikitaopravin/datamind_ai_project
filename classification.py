from pycaret.classification import *
import streamlit as st

@st.cache_data
def prep_and_train(targ, data, models=None):
    s = setup(data, target = targ, session_id = 12)
    s_df = pull()
    best = compare_models(include=models)
    best_df = pull()
    return best, s_df, best_df


@st.cache_resource
def tuning(_model, n_iters, search_lib):
    tuned_dt = tune_model(estimator=_model, n_iter=n_iters, search_library=search_lib, choose_better=True)
    info_df = pull()
    return tuned_dt, info_df



