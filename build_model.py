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

def data_prep(seed):
    profile = brandy_profile.Profile()
    interest = brandy_interest.Interest()
    preprocess = brandy_preprocess.Preprocessor()
    profile_raw = profile.get_profile()
    interest_raw, ids = interest.data_merge()
    data = preprocess.finalize_data(profile_raw, interest_raw)
    X, y, X_train, y_train, X_test, y_test = preprocess.split_data(data, seed=seed, re=False)
    return X, y, X_train, y_train, X_test, y_test, ids

def build_model(X, y, X_train, y_train, X_test, y_test, seed, method, ):
    model = brandy_model.Model()
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
    print(' \n With v represents VotingClassifier and s represents StackingClassifier.')
    method = input('Please specify preferred method (v or s): ')
    warnings.filterwarnings('ignore', category=FutureWarning)
    X, y, X_train, y_train, X_test, y_test, ids = data_prep(seed)
    evc_meta = build_model(X, y, X_train, y_train, X_test, y_test, seed=seed, method=method)
    print('***************************************************************************************')
    print('***************************************************************************************')

