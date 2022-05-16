import pandas as pd
import numpy as np
import os
import sklearn as sk
import azureml.core
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer 
from azureml.core import Run
import joblib
from azureml.core import Dataset, Workspace
import argparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Normalizer

def read_file(file_name):
    cols = ['engine_number','time_in_cycles']
    op_set_columns = ['op_set{}'.format(i) for i in range(1,4)]
    sm_cols = ['sm{}'.format(i) for i in range(1,24)]
    columns = cols + op_set_columns + sm_cols
    train = run.input_datasets[file_name].to_pandas_dataframe()
    train.columns = columns
    return train

def generate_rul(df):
    df_rul = pd.DataFrame(df.groupby('engine_number')['time_in_cycles'].max().reset_index())
    df_rul.columns = ['engine_number','max_time']
    df = df.merge(df_rul,how="left",on="engine_number")
    df['rul'] = df['max_time'] - df['time_in_cycles'] #remaining useful life
    df.drop('max_time',axis=1,inplace=True)
    return df

def add_rul_test(df,rul1):
    rul1.rename(columns = {'Column1':'rul'},inplace=True)
    rul1['engine_number'] = rul1.index + 1
    test1_life = df.groupby('engine_number')['time_in_cycles'].max().reset_index()
    test1_life['max_life'] = test1_life['time_in_cycles'] + rul1['rul']
    test1_life.drop('time_in_cycles',axis=1,inplace=True)
    new_test = df.merge(test1_life,on="engine_number",how="left")
    new_test['rul'] = new_test['max_life'] - new_test['time_in_cycles']
    new_test.drop('max_life',axis=1,inplace=True)
    return new_test


parser = argparse.ArgumentParser()
parser.add_argument("--train-data", type=str, dest='train_data', help='raw train data')
parser.add_argument('--test-data',type=str,dest='test_data',help = "raw test data")
parser.add_argument('--rul-data',type=str,dest='rul_data',help="rul for test data")
parser.add_argument('--cleanedtrain-data', type=str, dest='cleaned_train_data', help='Folder for results')
parser.add_argument('--cleanedtest-data', type=str, dest='cleaned_test_data', help='Folder for results')
args = parser.parse_args()
save_train_folder = args.cleaned_train_data
save_test_folder = args.cleaned_test_data

run = Run.get_context()
train = read_file(file_name='raw_train_data')
train = generate_rul(train)
test = read_file(file_name = "raw_test_data")
rul = run.input_datasets['rul_data'].to_pandas_dataframe()
test = add_rul_test(test,rul)
run.log('Shape of training set:',train.shape)
run.log('Shape of test set:',test.shape)

train_data = train.copy()
train_data.drop(columns = ['op_set1','op_set2','op_set3','sm1','sm5','sm10','sm16','sm18','sm19','sm14','sm22','sm23'],inplace=True)

test_data = test.copy()
test_data.drop(columns = ['op_set1','op_set2','op_set3','sm1','sm5','sm10','sm16','sm18','sm19','sm14','sm22','sm23'],inplace=True)

cols_norm = train_data.columns.difference(['engine_number','rul'])
norm = Normalizer()
tmp_train = norm.fit_transform(train_data[cols_norm])

df_train = pd.DataFrame(tmp_train,columns=cols_norm)
train_df = pd.concat([df_train,train_data[['engine_number','rul']]],axis=1)

tmp_test = norm.transform(test_data[cols_norm])
df_test = pd.DataFrame(tmp_test,columns = cols_norm)
test_df = pd.concat([df_test,test_data[['engine_number','rul']]],axis=1)

os.makedirs(save_train_folder, exist_ok=True)
os.makedirs(save_test_folder,exist_ok=True)
train_path = os.path.join(save_train_folder,'transformed_train_data.csv')
test_path = os.path.join(save_test_folder,'transformed_test_data.csv')

train_df.to_csv(train_path,index=False,header=True)
test_df.to_csv(test_path,index=False,header=True)


