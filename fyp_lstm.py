from datetime import datetime

from pandas import read_csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.metrics import binary_accuracy

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import precision_score


from numpy import array

import common

# def parse(time):
#     return datetime.strptime(time, '%H:%M:%S.%f')


def clean_data():
    dataset = read_csv(common.ibmFull)
    dataset.drop('Ticker', axis=1, inplace=True)
    dataset.drop('Conditions', axis=1, inplace=True)
    dataset.columns = ['Timestamp', 'EventType', 'Price', 'Quantity', 'Exchange', 'Wash']
    dataset.to_csv(common.ibmFullPre, index=False)
    print(dataset.head(5))


clean_data()

dataframe = read_csv(common.ibmFullPre, header=0, index_col=0)
dataset = dataframe.values

encoder = LabelEncoder()
dataset[:, 0] = encoder.fit_transform(dataset[:, 0])
dataset[:, 3] = encoder.fit_transform(dataset[:, 3])
# dataset = dataset.astype('float32')

#normalise
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(dataset)

X = dataset[:,0:4]
Y = dataset[:,4]


# Create train and test set
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))


lookback = 1
trainX, trainY = train[:, :-1], train[:, -1]
testX, testY = test[:, :-1], test[:, -1]

train_X = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
test_X = testX.reshape((testX.shape[0], 1, testX.shape[1]))

print(trainX.shape, trainY.shape, testX.shape, testY.shape)


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy', 'binary_accuracy'])


# fit network
history = model.fit(train_X, trainY, epochs=10, batch_size=72, validation_data=(test_X, testY), verbose=2, shuffle=False)
results = model.evaluate(test_X, testY)

predict = model.predict(test_X)
print(predict)
print(history)



# def lstm_model():
   # design network
   #  model = Sequential()
   #  model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
   #  model.add(Dense(1))
   #  model.compile(loss='mae', optimizer='adam')
