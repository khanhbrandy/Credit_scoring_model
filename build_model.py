"""
Created on 2019-12-26
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
import brandy_interest
import brandy_profile
import brandy_preprocess
import brandy_model
import joblib
import warnings

def build_model(data, seed, method):
    model = brandy_model.Model()
    X, y, X_train, y_train, X_test, y_test = model.split_data(data, seed=seed, re=False)
    evc_meta = model.model_ensemble(X, y, method=method)
    model.model_predict(evc_meta,X_train,y_train,X_test,y_test, seed=seed)
    model.cross_validate(evc_meta, X, y, seed)
    print('Start dumping Meta classifier...')
    joblib.dump(evc_meta, 'meta_clf.pkl') 
    print('Done dumping Meta classifier ! \n')
    return evc_meta
if __name__=='__main__':
    print('***************************************************************************************') 
    print('***************************************************************************************')
    seed = 50
    print('With v represents VotingClassifier and s represents StackingClassifier. \n')
    method = input('Please specify preferred method (v or s): ')
    profile = brandy_profile.Profile()
    interest = brandy_interest.Interest()
    preprocess = brandy_preprocess.Preprocessor()
    profile_raw = profile.get_profile()
    interest_raw, ids, fbids_lv1, fbids_lv2, fbids_lv3, fbids_lv4, fbids_lv5 = interest.data_merge()
    # interest_raw.to_excel('interest.xlsx')
    data = preprocess.finalize_data(profile_raw, interest_raw)
    warnings.filterwarnings('ignore', category=FutureWarning)
    evc_meta = build_model(data, seed=seed, method=method)
    print('***************************************************************************************')
    print('***************************************************************************************')

