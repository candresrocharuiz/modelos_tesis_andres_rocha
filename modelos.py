import glob
import pickle
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# Red neuronal
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Definir funciones
def create_dataset(dataset, look_back=1):
    	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# Crear modelo LSTM
def modelo_lstm(data, columna, rezagos, size=0.8):
  """
  data: DataFrame con las series de tiempo
  columna: Nombre de la columna de la serie de tiempo
  rezagos: Cuántos rezagos se quiere crear el dataset
  size: Proporción de datos de entrenamiento  
  """
  # Seleccionar la serie
  dataset = data[columna].values
  dataset = dataset.astype('float32')
  dataset = dataset.reshape(dataset.shape[0],1)
  # Escalarlo
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  train_size = int(len(dataset) * size)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  look_back = rezagos
  # Crear los datos
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)
  trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
  testX = testX.reshape(testX.shape[0], testX.shape[1], 1)
  # Modelo
  model = Sequential()
  model.add(LSTM(10, input_shape=(look_back, 1), return_sequences=True))
  model.add(Dropout(0.3))
  model.add(LSTM(10))
  model.add(Dropout(0.3))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
  # Predecir
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)
  # Pronosticos recursivos
  testPredict_r = model.predict(testX[0].reshape(1,12,1))
  recursivo = list(testX[:,:,0][0])
  for i in range(1,12):
    a = testPredict_r[[0]]
    recursivo = recursivo[1:len(recursivo)]
    recursivo.append(list(a[0])[0])
    recursivo = np.array(recursivo).reshape(1,12,1)
    testPredict_r = model.predict(recursivo)
    recursivo = list(recursivo.reshape(12))
  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict)
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict)
  testY = scaler.inverse_transform([testY])
  testPredict_r = scaler.inverse_transform(np.array(recursivo).reshape(1,12))
  # calculate root mean squared error
  trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
  print('Train Score: %.2f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
  print('Test Score: %.2f RMSE' % (testScore))
  testScore_r = math.sqrt(mean_squared_error(testY[0], testPredict_r[0][-11:]))
  print('Test Score_r: %.2f RMSE' % (testScore_r))
  # graficos
  trainPredictPlot = numpy.empty_like(dataset)
  trainPredictPlot[:, :] = numpy.nan
  trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
  # shift test predictions for plotting
  testPredictPlot = numpy.empty_like(dataset)
  testPredictPlot[:, :] = numpy.nan
  # testPredictPlot
  testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
  # plot baseline and predictions
  plt.rcParams['figure.dpi'] = 200 # default for me was 75
  plt.plot(scaler.inverse_transform(dataset))
  plt.plot(trainPredictPlot)
  plt.plot(testPredictPlot)
  plt.show()
  return trainScore, testScore, testPredict, testPredict_r, model

datos = pd.read_csv('datos/series_muestra.csv', sep = ';')

## Ejecutar el modelo LSTM
train_score = {}
test_score = {}
y_hat = {}
y_hat_r = {}
modelos = {}
for column in datos.columns:
  t1, t2, t3, t4, m1 = modelo_lstm(datos, column, 12, 0.95)
  train_score[column] = t1
  test_score[column] = t2
  y_hat[column] = t3
  y_hat_r[column] = t4
  modelos[column] = m1


with open('/content/drive/My Drive/U TADEO/Maestría/train_score_lstm.pkl', 'wb') as fid:
     pickle.dump(train_score, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/test_score_lstm.pkl', 'wb') as fid:
     pickle.dump(test_score, fid)


with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_lstm.pkl', 'wb') as fid:
     pickle.dump(y_hat, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_r_lstm.pkl', 'wb') as fid:
     pickle.dump(y_hat_r, fid)


## SVM

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def modelo_svm(data, columna, rezagos, size=0.8):
  """
  Mismos parámetros del LSTM para esta función
  """
  # Seleccionar la serie
  dataset = data[columna].values
  dataset = dataset.astype('float32')
  dataset = dataset.reshape(dataset.shape[0],1)
  # Escalarlo
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  train_size = int(len(dataset) * size)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  look_back = rezagos
  # Crear los datos
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)
  # Modelo
  regressor = SVR()
  k=['rbf', 'linear','poly','sigmoid']
  c= range(1,100)
  g=np.arange(1e-4,1e-2,0.0001)
  g=g.tolist()
  param_grid=dict(kernel=k, C=c, gamma=g)
  grid = GridSearchCV(regressor, param_grid, cv=5)
  grid.fit(trainX, trainY)  
  # Predecir
  trainPredict = grid.predict(trainX)
  testPredict = grid.predict(testX)
  # Pronosticos recursivos
  trainPredict = grid.predict(trainX)
  testPredict = grid.predict(testX)
  # Pronosticos recursivos
  testPredict_r = grid.predict(testX[0].reshape(1,12))
  recursivo = list(testX[0])
  for i in range(1,12):
    a = testPredict_r[0]
    recursivo = recursivo[1:len(recursivo)]
    recursivo.append([a][0])
    recursivo = np.array(recursivo).reshape(1,12)
    testPredict_r = grid.predict(recursivo)
    recursivo = list(recursivo.reshape(12))
  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict.reshape(trainPredict.shape[0], 1))
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict.reshape(testPredict.shape[0], 1))
  testY = scaler.inverse_transform([testY])
  testPredict_r = scaler.inverse_transform(np.array(recursivo).reshape(1,12))
  return testPredict, testPredict_r, grid


y_hat_svm = {}
y_hat_r_svm = {}
modelos_svm = {}
for column in datos.columns:
  print('Empezando ' + column)
  t3_svm, t4_svm, m1_svm = modelo_svm(datos, column, 12, 0.95)
  # train_score_svm[column] = t1_svm
  # test_score_svm[column] = t2_svm
  y_hat_svm[column] = t3_svm
  y_hat_r_svm[column] = t4_svm
  modelos_svm[column] = m1_svm
  print(column)



with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_svm.pkl', 'wb') as fid:
     pickle.dump(y_hat_svm, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_r_svm.pkl', 'wb') as fid:
     pickle.dump(y_hat_r_svm, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/modelos_svm.pkl', 'wb') as fid:
     pickle.dump(modelos_svm, fid)



## RF

def modelo_rf(data, columna, rezagos, size=0.8):
      # Seleccionar la serie
  dataset = data[columna].values
  dataset = dataset.astype('float32')
  dataset = dataset.reshape(dataset.shape[0],1)
  # Escalarlo
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  train_size = int(len(dataset) * size)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  look_back = rezagos
  # Crear los datos
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)
  # Modelo
  rf = RandomForestRegressor(random_state = 42)
  # Number of trees in random forest
  n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt']
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  max_depth.append(None)
  # Minimum number of samples required to split a node
  min_samples_split = [2, 5, 10]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 2, 4]
  # Method of selecting samples for training each tree
  bootstrap = [True, False]
  # Create the random grid
  random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
  # Use the random grid to search for best hyperparameters
  # First create the base model to tune
  rf = RandomForestRegressor()
  # Random search of parameters, using 3 fold cross validation, 
  # search across 100 different combinations, and use all available cores
  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
  # Fit the random search model
  rf_random.fit(trainX, trainY)
  # Predecir
  trainPredict = rf_random.predict(trainX)
  testPredict = rf_random.predict(testX)
  # Pronosticos recursivos
  trainPredict = rf_random.predict(trainX)
  testPredict = rf_random.predict(testX)
  # Pronosticos recursivos
  testPredict_r = rf_random.predict(testX[0].reshape(1,12))
  recursivo = list(testX[0])
  for i in range(1,12):
    a = testPredict_r[0]
    recursivo = recursivo[1:len(recursivo)]
    recursivo.append([a][0])
    recursivo = np.array(recursivo).reshape(1,12)
    testPredict_r = rf_random.predict(recursivo)
    recursivo = list(recursivo.reshape(12))
  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict.reshape(trainPredict.shape[0], 1))
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict.reshape(testPredict.shape[0], 1))
  testY = scaler.inverse_transform([testY])
  testPredict_r = scaler.inverse_transform(np.array(recursivo).reshape(1,12))
  return testPredict, testPredict_r, rf_random



y_hat_rf = {}
y_hat_r_rf = {}
modelos_rf = {}
for column in datos.columns:
  print('Empezando ' + column)
  t3_rf, t4_rf, m1_rf = modelo_rf(datos, column, 12, 0.95)
  # train_score_svm[column] = t1_svm
  # test_score_svm[column] = t2_svm
  y_hat_rf[column] = t3_rf
  y_hat_r_rf[column] = t4_rf
  modelos_rf[column] = m1_rf
  print(column)



with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_rf.pkl', 'wb') as fid:
     pickle.dump(y_hat_rf, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_r_rf.pkl', 'wb') as fid:
     pickle.dump(y_hat_r_rf, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/modelos_rf.pkl', 'wb') as fid:
     pickle.dump(modelos_rf, fid)

## KNN

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def modelo_knn(data, columna, rezagos, size=0.8):
      # Seleccionar la serie
  dataset = data[columna].values
  dataset = dataset.astype('float32')
  dataset = dataset.reshape(dataset.shape[0],1)
  # Escalarlo
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  train_size = int(len(dataset) * size)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  look_back = rezagos
  # Crear los datos
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)
  # Modelo
  model = KNeighborsRegressor()
  # Create 3 folds
  seed = 1995
  kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
  # Define our candidate hyperparameters
  hp_candidates = [{'n_neighbors': [2,3,4,5,6], 'weights': ['uniform','distance']}]
  # Search for best hyperparameters
  grid = GridSearchCV(estimator=model, param_grid=hp_candidates, cv=kfold, scoring='neg_root_mean_squared_error')
  grid.fit(trainX, trainY)
  # Predecir
  trainPredict = grid.predict(trainX)
  testPredict = grid.predict(testX)
  # Pronosticos recursivos
  trainPredict = grid.predict(trainX)
  testPredict = grid.predict(testX)
  # Pronosticos recursivos
  testPredict_r = grid.predict(testX[0].reshape(1,12))
  recursivo = list(testX[0])
  for i in range(1,12):
    a = testPredict_r[0]
    recursivo = recursivo[1:len(recursivo)]
    recursivo.append([a][0])
    recursivo = np.array(recursivo).reshape(1,12)
    testPredict_r = grid.predict(recursivo)
    recursivo = list(recursivo.reshape(12))
  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict.reshape(trainPredict.shape[0], 1))
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict.reshape(testPredict.shape[0], 1))
  testY = scaler.inverse_transform([testY])
  testPredict_r = scaler.inverse_transform(np.array(recursivo).reshape(1,12))
  return testPredict, testPredict_r, grid


# train_score_svm = {}
# test_score_svm = {}
y_hat_knn = {}
y_hat_r_knn = {}
modelos_knn = {}
for column in datos.columns:
  print('Empezando ' + column)
  t3_knn, t4_knn, m1_knn = modelo_knn(datos, column, 12, 0.95)
  # train_score_svm[column] = t1_svm
  # test_score_svm[column] = t2_svm
  y_hat_knn[column] = t3_knn
  y_hat_r_knn[column] = t4_knn
  modelos_knn[column] = m1_knn
  print(column)



with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_knn.pkl', 'wb') as fid:
     pickle.dump(y_hat_knn, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_r_knn.pkl', 'wb') as fid:
     pickle.dump(y_hat_r_knn, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/modelos_knn.pkl', 'wb') as fid:
     pickle.dump(modelos_knn, fid)


# Modelo lineal

from sklearn.linear_model import LinearRegression

def modelo_lineal(data, columna, rezagos, size=0.8):
      # Seleccionar la serie
  dataset = data[columna].values
  dataset = dataset.astype('float32')
  dataset = dataset.reshape(dataset.shape[0],1)
  # Escalarlo
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  train_size = int(len(dataset) * size)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  look_back = rezagos
  # Crear los datos
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)
  # Modelo
  model = LinearRegression()
  model.fit(trainX, trainY)
  # Predecir
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)
  # Pronosticos recursivos
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)
  # Pronosticos recursivos
  testPredict_r = model.predict(testX[0].reshape(1,12))
  recursivo = list(testX[0])
  for i in range(1,12):
    a = testPredict_r[0]
    recursivo = recursivo[1:len(recursivo)]
    recursivo.append([a][0])
    recursivo = np.array(recursivo).reshape(1,12)
    testPredict_r = model.predict(recursivo)
    recursivo = list(recursivo.reshape(12))
  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict.reshape(trainPredict.shape[0], 1))
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict.reshape(testPredict.shape[0], 1))
  testY = scaler.inverse_transform([testY])
  testPredict_r = scaler.inverse_transform(np.array(recursivo).reshape(1,12))
  return testPredict, testPredict_r, model



# train_score_svm = {}
# test_score_svm = {}
y_hat_lineal = {}
y_hat_r_lineal = {}
modelos_lineal = {}
for column in datos.columns:
  print('Empezando ' + column)
  t3_lineal, t4_lineal, m1_lineal = modelo_lineal(datos, column, 12, 0.95)
  # train_score_svm[column] = t1_svm
  # test_score_svm[column] = t2_svm
  y_hat_lineal[column] = t3_lineal
  y_hat_r_lineal[column] = t4_lineal
  modelos_lineal[column] = m1_lineal
  print(column)



with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_lineal.pkl', 'wb') as fid:
     pickle.dump(y_hat_lineal, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_r_lineal.pkl', 'wb') as fid:
     pickle.dump(y_hat_r_lineal, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/modelos_lineal.pkl', 'wb') as fid:
     pickle.dump(modelos_lineal, fid)


#XG boost
from xgboost import XGBRegressor

def modelo_xgboost(data, columna, rezagos, size=0.8):
      # Seleccionar la serie
  dataset = data[columna].values
  dataset = dataset.astype('float32')
  dataset = dataset.reshape(dataset.shape[0],1)
  # Escalarlo
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  train_size = int(len(dataset) * size)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  look_back = rezagos
  # Crear los datos
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)
  # Modelo
  param_tuning = {
          'learning_rate': [0.01, 0.1],
          'max_depth': [3, 5, 7, 10],
          'min_child_weight': [1, 3, 5],
          'subsample': [0.5, 0.7],
          'colsample_bytree': [0.5, 0.7],
          'n_estimators' : [100, 200, 500],
          'objective': ['reg:squarederror']
      }
  xgb_model = XGBRegressor()
  gsearch = GridSearchCV(estimator = xgb_model,
                            param_grid = param_tuning,                        
                            #scoring = 'neg_mean_absolute_error', #MAE
                            #scoring = 'neg_mean_squared_error',  #MSE
                            cv = 5,
                            n_jobs = -1,
                            verbose = 1)
  gsearch.fit(trainX,trainY)
  # Predecir
  trainPredict = gsearch.predict(trainX)
  testPredict = gsearch.predict(testX)
  # Pronosticos recursivos
  trainPredict = gsearch.predict(trainX)
  testPredict = gsearch.predict(testX)
  # Pronosticos recursivos
  testPredict_r = gsearch.predict(testX[0].reshape(1,12))
  recursivo = list(testX[0])
  for i in range(1,12):
    a = testPredict_r[0]
    recursivo = recursivo[1:len(recursivo)]
    recursivo.append([a][0])
    recursivo = np.array(recursivo).reshape(1,12)
    testPredict_r = gsearch.predict(recursivo)
    recursivo = list(recursivo.reshape(12))
  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict.reshape(trainPredict.shape[0], 1))
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict.reshape(testPredict.shape[0], 1))
  testY = scaler.inverse_transform([testY])
  testPredict_r = scaler.inverse_transform(np.array(recursivo).reshape(1,12))
  return testPredict, testPredict_r, gsearch



y_hat_xg = {}
y_hat_r_xg = {}
modelos_xg = {}
for column in datos.columns:
  print('Empezando ' + column)
  t3_xg, t4_xg, m1_xg = modelo_xgboost(datos, column, 12, 0.95)
  # train_score_svm[column] = t1_svm
  # test_score_svm[column] = t2_svm
  y_hat_xg[column] = t3_xg
  y_hat_r_xg[column] = t4_xg
  modelos_xg[column] = m1_xg
  print(column)


with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_xg.pkl', 'wb') as fid:
     pickle.dump(y_hat_xg, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_r_xg.pkl', 'wb') as fid:
     pickle.dump(y_hat_r_xg, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/modelos_xg.pkl', 'wb') as fid:
     pickle.dump(modelos_xg, fid)

# Arbol de decisión
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


def modelo_arbol(data, columna, rezagos, size=0.8):
      # Seleccionar la serie
  dataset = data[columna].values
  dataset = dataset.astype('float32')
  dataset = dataset.reshape(dataset.shape[0],1)
  # Escalarlo
  scaler = MinMaxScaler(feature_range=(0, 1))
  dataset = scaler.fit_transform(dataset)
  train_size = int(len(dataset) * size)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
  look_back = rezagos
  # Crear los datos
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)
  # Modelo
  dtc = DecisionTreeRegressor(min_samples_split=4, random_state=1995)
  sample_split_range = list(range(1, 50))
  param_grid = dict(min_samples_split=sample_split_range)
  grid = GridSearchCV(dtc, param_grid, cv=10, scoring='neg_mean_squared_error')
  grid.fit(trainX, trainY)
  # Predecir
  trainPredict = grid.predict(trainX)
  testPredict = grid.predict(testX)
  # Pronosticos recursivos
  trainPredict = grid.predict(trainX)
  testPredict = grid.predict(testX)
  # Pronosticos recursivos
  testPredict_r = grid.predict(testX[0].reshape(1,12))
  recursivo = list(testX[0])
  for i in range(1,12):
    a = testPredict_r[0]
    recursivo = recursivo[1:len(recursivo)]
    recursivo.append([a][0])
    recursivo = np.array(recursivo).reshape(1,12)
    testPredict_r = grid.predict(recursivo)
    recursivo = list(recursivo.reshape(12))
  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict.reshape(trainPredict.shape[0], 1))
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict.reshape(testPredict.shape[0], 1))
  testY = scaler.inverse_transform([testY])
  testPredict_r = scaler.inverse_transform(np.array(recursivo).reshape(1,12))
  return testPredict, testPredict_r, grid




y_hat_arbol = {}
y_hat_r_arbol = {}
modelos_arbol = {}
for column in datos.columns:
  print('Empezando ' + column)
  t3_arbol, t4_arbol, m1_arbol = modelo_arbol(datos, column, 12, 0.95)
  # train_score_svm[column] = t1_svm
  # test_score_svm[column] = t2_svm
  y_hat_arbol[column] = t3_arbol
  y_hat_r_arbol[column] = t4_arbol
  modelos_arbol[column] = m1_arbol
  print(column)



with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_arbol.pkl', 'wb') as fid:
     pickle.dump(y_hat_arbol, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/y_hat_r_arbol.pkl', 'wb') as fid:
     pickle.dump(y_hat_r_arbol, fid)

with open('/content/drive/My Drive/U TADEO/Maestría/modelos_arbol.pkl', 'wb') as fid:
     pickle.dump(modelos_arbol, fid)


