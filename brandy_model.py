"""
Created on 2019-12-26
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
from sklearn import model_selection
from sklearn import metrics
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from mlxtend.classifier import StackingClassifier, StackingCVClassifier, EnsembleVoteClassifier
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt



class Mymetrics():
    def __init__(self):
        pass
    def accuracy(self, y_test, y_pred):
        # Accuracy
        acc = metrics.accuracy_score(y_test, y_pred)
        print('Classifier Accuracy: {:.2f}%'.format(acc * 100))
        return acc
    def roc_curve(self, y_test, y_pred):
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
        return fpr, tpr, threshold
    def auc_score(self, fpr, tpr):
        roc_auc = metrics.auc(fpr, tpr)
        print('Classifier AUC: {:.2f}%'.format(roc_auc*100))
        return roc_auc
    def precision_score(self, y_test, y_pred):
        precision_scr = metrics.precision_score(y_test, y_pred)
        print('Precision score is {:.2f}'.format(float(precision_scr)))
        return precision_scr
    def recall_score(self, y_test, y_pred):
        recall_scr = metrics.recall_score(y_test, y_pred)
        print('Recall score is {:.2f}'.format(float(recall_scr)))
        return recall_scr

class Myvisualization(Mymetrics):
    def __init__(self):
        pass
    def roc_auc_viz(self, y_test,y_pred):
        fpr, tpr, threshold = self.roc_curve(y_test, y_pred)
        roc_auc = self.auc_score(fpr, tpr)    
        gini_score=2*roc_auc-1
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = {:.2f} and GINI = {:.2f}'.format(roc_auc,gini_score))
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        pass
class Model(Myvisualization):
    def __init__(self):
        self.clf_0 = xgb.XGBClassifier(
                    subsample= 0.8, 
                    silent= 1, 
                    seed= 50, 
                    reg_lambda= 40, 
                    reg_alpha= 10, 
                    objective= 'binary:logistic', 
                    n_estimators= 168, 
                    min_child_weight= 15, 
                    max_depth= 4, 
                    learning_rate= 0.05, 
                    gamma= 0.8, 
                    colsample_bytree= 0.4, 
                    class_weight= 'd',
                    verbose=2,
                    random_state=50)
        self.clf_1 = AdaBoostClassifier(
                    n_estimators= 800, 
                    learning_rate= 0.01,
                    random_state=50)
        self.clf_2 = LGBMClassifier(
                    reg_lambda= 20, 
                    reg_alpha= 20, 
                    num_leaves= 10, 
                    n_estimators= 512, 
                    max_depth= 5, 
                    learning_rate= 0.01, 
                    class_weight= 'balanced',
                    random_state=50)
        self.clf_3 = LogisticRegression(
                    penalty='l2', 
                    solver='liblinear', 
                    C=10000, 
                    class_weight='balanced'
                    )
        self.clf_4 = EnsembleVoteClassifier(clfs=[self.clf_0, self.clf_1, self.clf_2], weights=[1,1,1])
        self.clf_5 = StackingClassifier(classifiers=[self.clf_0, self.clf_2], meta_classifier=self.clf_3, use_probas=True)
        
    
    def split_data(self, data, seed, re=False):
        X, y = data.iloc[:,1:-1],data.iloc[:,-1]
        # Train-Test split
        test_size = 0.2
        X_train_o, X_test, y_train_o, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
        # Resampling
        if re:
            resam=SMOTE(random_state=seed)
            resam.fit(X_train_o, y_train_o)
            X_train, y_train = resam.fit_resample(X_train_o, y_train_o)
            X_train = pd.DataFrame(X_train, columns=X_train_o.columns)
            y_train = pd.Series(y_train)
        else:
            X_train, y_train = X_train_o,y_train_o
        return X, y, X_train, y_train, X_test, y_test

    def model_ensemble(self, X, y, method):
        def base_fit(X, y, clf_list, clf_name, meta_clf):
            clf_list.append(meta_clf)
            clf_name.append(meta_clf.__class__.__name__)
            enum = zip(clf_list, clf_name)
            print('Start fitting base classifiers with 3-fold cross validation...')
            start = time.time()
            for clf, label in enum:
                scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='roc_auc')
                print('Accuracy: {:.2f} (+/- {:.2f}) [{}]'.format(scores.mean(), scores.std(), label))
            print('Done fitting base classifiers. Time taken = {:.1f}(s) \n'.format(time.time()-start))
            return meta_clf
        if method == 'v':
            print('********** Voting method choosen **********')
            clf_list = [self.clf_0, self.clf_1, self.clf_2]
            clf_name = [self.clf_0.__class__.__name__, self.clf_1.__class__.__name__, self.clf_2.__class__.__name__]
            meta_clf = base_fit(X, y, clf_list, clf_name, self.clf_4)
            return meta_clf
        elif method == 's': 
            print('********** Stacking method choosen **********')
            clf_list = [self.clf_0, self.clf_2]
            clf_name = [self.clf_0.__class__.__name__, self.clf_2.__class__.__name__]
            meta_clf = base_fit(X, y, clf_list, clf_name, self.clf_5)
            return meta_clf
        else:
            print('********** Please specify preferred method ! **********')

    def model_predict(self, model, X_train, y_train, X_test, y_test, seed):
        if 'random_state' in model.get_params().keys():
            model.set_params(random_state=seed)
        print('Start fitting Meta classifier...')
        start = time.time()
        model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X_test)
        # Accuracy
        acc = self.accuracy(y_test, y_pred)
        fpr, tpr, threshold = self.roc_curve(y_test, model.predict_proba(X_test)[:,1])
        roc_auc = self.auc_score(fpr, tpr)
        precision_scr = self.precision_score(y_test, y_pred)
        recall_scr = self.recall_score(y_test, y_pred)
        # print('{} accuracy: {:.2f}% and AUC: {:.2f}% \n'.format(model.__class__.__name__, acc * 100, roc_auc*100))
        print('Done fitting meta classifier. Time taken = {:.1f}(s) \n'.format(time.time()-start))
        pass

    def cross_validate(self, model, X, y, seed):
        kfold = model_selection.StratifiedKFold(n_splits=4,shuffle=True, random_state=seed)
        print('Start cross validating Meta classifier...')
        results = model_selection.cross_val_score(model, X, y, scoring='roc_auc',cv=kfold)
        print('Done cross validatiion. Validated AUC: {:.2f} (+/- {})'.format(results.mean()*100, results.std()*100))
        pass 



