from math import sqrt

import numpy
from matplotlib import pyplot
from numpy import concatenate, math, array, delete, arange
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import ThresholdedReLU
from keras.layers import InputLayer
from datetime import datetime
import common


from keras import backend as K

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def remove_wash_feat(data, lookback, features):
    columns = []
    for i in range((lookback + 1) * features, 0, -features):
        columns.append(i - 1)
    data = delete(data, columns, axis=1)
    return data


# load data
def parse(x):
    dt_obj = datetime.strptime(x,
                               '%f')
    print(dt_obj)
    return dt_obj

def clean_data():
    dataset = read_csv('../datasets/IBM.csv')
    dataset.drop('Ticker', axis=1, inplace=True)
    dataset.drop('Conditions', axis=1, inplace=True)
    dataset.columns = ['EventType', 'Exchange', 'Price', 'Quantity', 'Timestamp', 'Wash']
    dataset.to_csv('../datasets/IBM_Full_pre.csv', index=False)
    print(dataset.head(5))

numpy.random.seed(7)

clean_data()


# load dataset
# dataset = read_csv('pollution.csv', header=0, index_col=0)
dataset = read_csv('../datasets/IBM_Full_pre.csv', index_col=4, header=0)
values = dataset.values

# Convert Event and Exchange
encoder = LabelEncoder()
values[:, 0] = encoder.fit_transform(values[:, 0])
values[:, 1] = encoder.fit_transform(values[:, 1])

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

lookback = 11
features = 5

# frame as supervised learning
reframed = series_to_supervised(scaled, lookback, 1)
print(reframed.head())

# split into train and test sets
values = reframed.values

# Training and Test set sizes
train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size

train, test = values[:train_size, :], values[train_size:, :]
# split into input and outputs

# Remove the wash trade feature from teh training sets
washless_train = remove_wash_feat(train, lookback, features)
washless_test = remove_wash_feat(test, lookback, features)

n_obs = lookback * features - 1

# Fill dataset with training and expected
train_X, train_y = washless_train[:, :], train[:, -1]
test_X, test_y = washless_test[:, :], test[:, -1]
print(train_X.shape, len(train_X), train_y.shape)


# reshape input to be 3D [samples, trades, features]
train_X = train_X.reshape((train_X.shape[0], lookback + 1, features - 1))
test_X = test_X.reshape((test_X.shape[0], lookback + 1, features - 1))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

def build_model():
    # design network
    model = Sequential()
    model.add(InputLayer(input_shape=(train_X.shape[1], train_X.shape[2])))

    model.add(LSTM(50, dropout=0.5, activation='linear', return_sequences=True))
    model.add(LSTM(10, dropout=0, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    return model


# fit network
model = build_model()
history = model.fit(train_X, train_y, epochs=10, batch_size=1000, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)


# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X[:, -4:], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,features - 1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, -4:], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, features - 1]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
