"""
Created on 2019-12-26
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
from sklearn import model_selection


class Preprocessor:
    def __init__(self):
        pass

    def finalize_data(self, profile_data, interest_strength):
        print('Start merging data for training...')
        start = time.time()
        data=pd.merge(interest_strength, profile_data[[
                                               'USER_ID',
                                               'AGE',
                                            #    'AGE_RANGE',
                                               'GENDER',
                                               'PRODUCT',
                                               'LOCATION_F',
                                               'RELATIONSHIP_F',
                                               'RELATIONSHIP_CHECK',
                                               'LOCATION_M',
                                               'LOCATION_CHECK',
                                               'RELATIONSHIP_M',
                                               'EDUCATION',
                                               'GB'
                                               ]], how='inner', on='USER_ID')
        data.dropna(inplace=True)
        print('Done merging data for training. Time taken = {:.1f}(s) \n'.format(time.time()-start))
        return data

    
