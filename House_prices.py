# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import optuna
import seaborn as sns
import lightgbm as lgb
import joblib

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit

# +
train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

cv_indicies = list(KFold().split(train_df))

# +
#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train_df.shape))
print("The test data size before dropping Id feature is : {} ".format(test_df.shape))

#Save the 'Id' column
train_ID = train_df['Id']
test_ID = test_df['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train_df.drop("Id", axis = 1, inplace = True)
test_df.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train_df.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test_df.shape))

#all_data = pd.concat([train_df,test_df])
# -

train_df = train_df.fillna('None')
test_df = test_df.fillna('None')

X_train = train_df.drop(['SalePrice'],axis=1, inplace = False)
y_train = train_df['SalePrice']

train_df.info()

display(train_df)

display(test_df)

print(train_df.shape)
print(test_df.shape)

# + code_folding=[0]
#modelセット
model = lgb.LGBMRegressor(colsample_bytree=0.68, 
                              learning_rate=0.1, 
                              min_child_samples=40, 
                              n_estimators=183, 
                              num_leaves=9064, 
                              reg_alpha=0.13, 
                              reg_lambda=0.0, 
                              subsample=0.83, 
                              subsample_for_bin=155127, 
                              subsample_freq=214953,
                              n_jobs=4, random_state=1)

#cross_val_score(estimator=model,X=df,y=df_target,cv=cv_indicies)

# +
def objective(trial):
    
    t_num_leaves = trial.suggest_int("num_leaves", 20, 10000)
    t_learning_rate = trial.suggest_float("learning_rate", 0.1, 0.7,step=0.1)
#    t_learning_rate = round(t_learning_rate,1)
    t_n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    t_subsample_for_bin = trial.suggest_int("subsample_for_bin", 0, 300000)
    t_subsample_freq = trial.suggest_int("subsample_freq", 0, 300000)
    t_reg_alpha = trial.suggest_float("reg_alpha", 0.1, 0.2,step = 0.01)
#    t_reg_alpha = round(t_reg_alpha,2)
    t_reg_lambda = trial.suggest_float("reg_lambda", 0, 0.2,step = 0.01)
#    t_reg_lambda = round(t_reg_lambda,2)
    t_colsample_bytree = trial.suggest_float("colsample_bytree", 0.01, 1,step = 0.01)
#    t_colsample_bytree = round(t_colsample_bytree,2)
    t_subsample = trial.suggest_float("subsample", 0.01, 1,step = 0.01)
#    t_subsample = round(t_subsample,2)
    t_min_child_samples = trial.suggest_int("min_child_samples", 0, 100)
    
    
    lgb_param = {"boosting_type":"gbdt",
                 "num_leaves":t_num_leaves,
                 "learning_rate":t_learning_rate,
                 "subsample_for_bin":t_subsample_for_bin,
                 "subsample_freq":t_subsample_freq,
                 "reg_alpha":t_reg_alpha,
                 "reg_lambda":t_reg_lambda,
                 "colsample_bytree":t_colsample_bytree,
                 "subsample":t_subsample,
                 "min_child_samples":t_min_child_samples,
                 "n_estimators":t_n_estimators
                }
    
    model = lgb.LGBMRegressor(
                **lgb_param,
                n_jobs=4, random_state=1)
    
    score = cross_val_score(estimator=model,X=X_train,y=y_train,cv=cv_indicies)[0]
    return score

study = optuna.create_study(direction='maximize',
                            study_name='Hyperparameters',
                            storage='sqlite:///optuna_studies.db',
                            load_if_exists=True)
study.optimize(objective, n_trials=100)
# -

param_b = study.best_params
param = {key: value for key, value in param_b.items()
            if key in {'colsample_bytree',
                       'learning_rate',
                       'min_child_samples',
                       'n_estimators',
                       'num_leaves',
                       'reg_alpha',
                       'reg_lambda',
                       'subsample',
                       'subsample_for_bin',
                       'subsample_freq'}
      }

# +
model = lgb.LGBMRegressor(
            **param,
            n_jobs=4, random_state=1)

model.fit(train_df, test_df)
joblib.dump(model, 'model.xz', compress=True)

importance = pd.DataFrame(sorted(zip(model.feature_importances_,df_head)), columns=['Value','Feature'])
display(importance)
