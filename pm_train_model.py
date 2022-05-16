import argparse
from azureml.core import Run,Model
import os
import pandas as pd
import joblib
import numpy as np
import keras
from xgboost import XGBRFRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

# function to reshape features into (samples, time steps, features) 
def gen_sequence(engine_df, seq_length):
    #splitter hele dataframen (rows) inn i sekvenser på 50. så først 0 til 50 så 1 til 51 ....N-50 til N.
    data_array = engine_df.values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

parser = argparse.ArgumentParser()
parser.add_argument("--training-folder", type=str, dest='training_folder', help='training data folder')
parser.add_argument('--test-folder',type=str,dest='test_folder',help = 'test data folder')
args = parser.parse_args()
training_folder = args.training_folder
test_folder = args.test_folder

run = Run.get_context()

print("Loading Data...")
train_path_transformed = os.path.join(training_folder,'transformed_train_data.csv')
clean_train_data = pd.read_csv(train_path_transformed)

test_path_transformed = os.path.join(test_folder,'transformed_test_data.csv')
clean_test_data = pd.read_csv(test_path_transformed)


X,y = clean_train_data.drop('rul',axis=1),clean_train_data['rul']
X_test,y_test = clean_test_data.drop('rul',axis=1),clean_test_data['rul']
window_length = 30

seq_gen = (list(gen_sequence(X[X['engine_number']==id], window_length)) 
           for id in X['engine_number'].unique())

seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

label_gen = [gen_labels(clean_train_data[clean_train_data['engine_number']==id], window_length, ['rul']) 
                 for id in clean_train_data['engine_number'].unique()]

label_array = np.concatenate(label_gen).astype(np.float32)

seq_gen_test = (list(gen_sequence(X_test[X_test['engine_number']==id], window_length)) 
           for id in X_test['engine_number'].unique())

seq_array_test = np.concatenate(list(seq_gen_test)).astype(np.float32)


label_gen_test = [gen_labels(clean_test_data[clean_test_data['engine_number']==id], window_length, ['rul']) 
                 for id in clean_test_data['engine_number'].unique()]


label_test_array = np.concatenate(label_gen_test).astype(np.float32)

best_network = keras.models.Sequential([keras.layers.LSTM(input_shape=(seq_array.shape[1],seq_array.shape[2]),activation='tanh',units=200,return_sequences=True,dropout=0.2),
keras.layers.LSTM(units=100,activation='tanh',return_sequences=False,dropout=.2),
keras.layers.Dense(1)])
best_network.compile(loss="mean_squared_error",optimizer='adam')
network_history = best_network.fit(seq_array,label_array,validation_split=.1,epochs=30,batch_size=200,callbacks = [keras.callbacks.EarlyStopping(patience=5)])

xgb_tree = XGBRFRegressor()
X.drop('engine_number',axis=1,inplace=True)

cv = cross_validate(xgb_tree,X,y,cv=10,scoring=['r2',"neg_root_mean_squared_error"])

lstm_train_loss = np.sqrt(min(network_history.history['loss']))
lstm_validation_loss = np.sqrt(min(network_history.history['val_loss']))

run.log('rmse for train:',(cv['test_neg_root_mean_squared_error']*-1).mean())
run.log('r2_score for train:',cv['test_r2'].mean())

run.log('rmse for lstm train data:',lstm_train_loss)
run.log('rmse for lstm validation data:',lstm_validation_loss)


xgb_tree.fit(X,y)

X_test.drop('engine_number',axis=1,inplace=True)
y_test_predict = xgb_tree.predict(X_test)

lstm_y_test_predict = best_network.predict(seq_array_test)

rmse_test = np.sqrt(mean_squared_error(y_test,y_test_predict))
run.log('rmse for test:',rmse_test)

lstm_rmse_test = np.sqrt(mean_squared_error(label_test_array,lstm_y_test_predict))
run.log('rmse for lstm test data:',lstm_rmse_test)

os.makedirs('outputs',exist_ok=True)
lstm_model_file = 'lstm_model.h5'
xgb_file_name = "xgbrfr_model.pkl"
best_network.save(filepath = lstm_model_file)
joblib.dump(value = xgb_tree,filename = xgb_file_name)
run.upload_file(name = 'outputs/' + lstm_model_file, path_or_stream = './' + lstm_model_file)
run.upload_file(name = 'outputs/' + xgb_file_name, path_or_stream = './' + xgb_file_name)

run.complete()
run.register_model(model_path='outputs/lstm_model.h5', model_name='lstm_model',
                   tags={'model':'lstm'},
                   properties={'rmse_train': run.get_metrics()['rmse for lstm train data:'],
                   'rmse for validation data:': run.get_metrics()['rmse for lstm validation data:'],
                   'rmse for test data:': run.get_metrics()['rmse for lstm test data:']})

run.register_model(model_path='outputs/xgbrfr_model.pkl', model_name='xgbrfr_model',
                   tags={'model':'xgboost random forest regressor'},
                   properties={'rmse_train': run.get_metrics()['rmse for train:'],'rmse_test':run.get_metrics()['rmse for test:'],'r2':run.get_metrics()['r2_score for train:']})






