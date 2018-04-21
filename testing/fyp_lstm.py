import math

import numpy
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential

from keras.metrics import binary_accuracy

from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import common


# def parse(time):
#     return datetime.strptime(time, '%H:%M:%S.%f')

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def clean_data():
    dataset = read_csv(common.ibmFull)
    dataset.drop('Ticker', axis=1, inplace=True)
    dataset.drop('Conditions', axis=1, inplace=True)
    dataset.columns = ['Timestamp', 'EventType', 'Price', 'Quantity', 'Exchange', 'Wash']
    dataset.to_csv(common.ibmFullPre, index=False)
    print(dataset.head(5))


# clean_data()



dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values

# encoder = LabelEncoder()
# dataset[:, 0] = encoder.fit_transform(dataset[:, 0])
# dataset[:, 3] = encoder.fit_transform(dataset[:, 3])

dataset = dataset.astype('float32')

# normalise
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(dataset)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# X = dataset[:,0:4]
# Y = dataset[:,4]


# Create train and test set
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))

# reshape into X=t and Y=t+1
lookback = 3

trainX, trainY = create_dataset(train, lookback)
testX, testY = create_dataset(test, lookback)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# print(trainX.shape, trainY.shape, testX.shape, testY.shape)

# design network
model = Sequential()
model.add(LSTM(5, input_shape=(1, lookback)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy', 'binary_accuracy'])
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# # fit network
# history = model.fit(train_X, trainY, epochs=100, batch_size=72, validation_data=(test_X, testY), verbose=2, shuffle=False)
# results = model.evaluate(test_X, testY)
#
# predict = model.predict(test_X)
# print(predict)
# print(binary_accuracy(predict, testY))
# print(history)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))



# def lstm_model():
# design network
#  model = Sequential()
#  model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#  model.add(Dense(1))
#  model.compile(loss='mae', optimizer='adam')
