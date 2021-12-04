
### Time Series Prediction


from tensorflow.keras.models import Sequential ### linear stack of layers from the module tenserflow, A way of defining layer
from tensorflow.keras.layers import Dense, LSTM # Fully connected layer
import math
from sklearn.metrics import mean_squared_error
import numpy as np ## module for all the mathematical calulations
from sklearn.preprocessing import MinMaxScaler  ###  Transform features by scaling each feature to a given range
import matplotlib.pyplot as plt  ## Plotting the chart


### Section for prediction with lstm model
class StockMarket:
  def __init__(self,dataset,timestep=10):
    self.dataset = dataset.set_index('Date')['Close']
    self.timestep = timestep
    self.normalizer = None
    self.dataset_scaled = None
    self.user_date = None
    self.y_train_pred = None
    self.y_valid_pred = None

  def normalize(self,dataset): ## Normalise the data between 0 to 1
    self.normalizer = MinMaxScaler()
    return self.normalizer.fit_transform(np.array(dataset).reshape(-1,1))
  
  def get_DatasetScaled(self):
    return self.dataset_scaled

  def preprocessing_data(self):
    '''
      Data Preprocessing:
        1. Closing prices were scaled to the range (0,1) to improve model performance.
        2. Dataset: Train set size = 70%; Test set size = 30%
    '''
    self.dataset_scaled = self.normalize(self.dataset)
    train_size = int(len(self.dataset_scaled)*0.7) ## Assigning the value for train and test data
    valid_size = len(self.dataset_scaled)-train_size
    X_train, X_valid = self.dataset_scaled[:train_size],self.dataset_scaled[train_size:]  ## Train and test data

    def create_data(data, timestep=3):
      x_features, y = [],[]
      for i in range(len(data)-timestep):
        x_features.append(data[i:i+timestep,0])
        y.append(data[i+timestep,0])
      return np.array(x_features), np.array(y)
    
    X_train, y_train = create_data(X_train,timestep=self.timestep)
    X_valid, y_valid = create_data(X_valid,timestep=self.timestep)
    X_train = X_train.reshape(X_train.shape+(1,))
    X_valid = X_valid.reshape(X_valid.shape+(1,))
    return X_train, y_train, X_valid, y_valid

  def build_model(self):
    model = Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(self.timestep,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    return model

  def plot_graph(self):
    # shift train predictions for plotting
    trainPredPlot = np.empty_like(self.dataset_scaled)
    trainPredPlot[:, :] = np.nan
    trainPredPlot[self.timestep-1:len(self.y_train_pred)+self.timestep-1, :] = self.y_train_pred
    # shift test predictions for plotting
    testPredPlot = np.empty_like(self.dataset_scaled)
    testPredPlot[:, :] = np.nan
    testPredPlot[len(self.y_train_pred)+(self.timestep*2)-1:len(self.dataset_scaled)-1, :] = self.y_valid_pred
    
    # plot baseline and predictions
    plt.plot(self.normalizer.inverse_transform(self.dataset_scaled))
    plt.plot(trainPredPlot)
    plt.plot(testPredPlot)
    #plt.show()
    return plt

  def stockMarketPred(self,user_date='2020-02-02',number_of_days=30):
    X_train, y_train, X_valid, y_valid = self.preprocessing_data()

    model = self.build_model()
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train,y_train, validation_data=(X_valid,y_valid),epochs=5,batch_size=64)

    self.y_train_pred = self.normalizer.inverse_transform(model.predict(X_train))
    self.y_valid_pred = self.normalizer.inverse_transform(model.predict(X_valid))

    print(f'Model Training Successful!!!')
    print(f'RMSE train data: {math.sqrt(mean_squared_error(y_train,self.y_train_pred))}')
    print(f'RMSE validation data: {math.sqrt(mean_squared_error(y_valid,self.y_valid_pred))}')


    self.user_date = self.dataset.index[-1]
    index = self.dataset.index.get_loc(self.user_date)
    test_data = self.dataset.iloc[index-self.timestep:index]
    test_data = self.normalizer.transform(np.array(test_data).reshape(-1,1)).reshape(1,-1)
    temp_data = list(test_data)[0].tolist()
    lst_output=[]
    i=0
    while(i<number_of_days):
      if(len(temp_data)>100): ## Training the data from 100 back date
          test_data=np.array(temp_data[1:])
          test_data=test_data.reshape(1,-1)
          test_data = test_data.reshape((1, self.timestep, 1))
          yhat = model.predict(test_data, verbose=0)
          temp_data.extend(yhat[0].tolist())
          temp_data=temp_data[1:]
          lst_output.extend(yhat.tolist())
          i=i+1
      else:

          test_data = test_data.reshape((1, self.timestep,1))
          yhat = model.predict(test_data, verbose=0)
          temp_data.extend(yhat[0].tolist())
          lst_output.extend(yhat.tolist())
          i=i+1
        

    return self.normalizer.inverse_transform(lst_output),f'\t\t\t\tRMSE validation data: {math.sqrt(mean_squared_error(y_valid,self.y_valid_pred))}'


"""https://github.com/krishnaik06/Stock-MArket-Forecasting/blob/master/Untitled.ipynb"""



