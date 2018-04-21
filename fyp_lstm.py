from math import sqrt

from matplotlib import pyplot

import numpy
from numpy import concatenate, math, array, delete, arange

from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import InputLayer

from datetime import datetime
import common
import model_configs


# convert series to supervised learning
def series_to_supervised(data, n_in=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
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
    dataset = read_csv(common.ibmMix30000CleanNoise)
    dataset.drop('Ticker', axis=1, inplace=True)
    dataset.drop('Conditions', axis=1, inplace=True)
    dataset.columns = ['EventType', 'Exchange', 'Price', 'Quantity', 'Timestamp', 'Wash']
    dataset.to_csv(common.ibmFullPre, index=False)
    print(dataset.head(5))


output_header = 'RMSE,Precision, Recall, TN, FP, FN, TP, Percent Correct\n'

clean_data()

# load dataset
dataset = read_csv(common.ibmFullPre, index_col=4, header=0)
raw_values = dataset.values

# Convert Event and Exchange
encoder = LabelEncoder()
raw_values[:, 0] = encoder.fit_transform(raw_values[:, 0])
raw_values[:, 1] = encoder.fit_transform(raw_values[:, 1])

# ensure all data is float
raw_values = raw_values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(raw_values)

for confId in range(0, len(model_configs.confs)):
    conf = model_configs.confs[confId]

    output_filename = conf['confId'] + '_output.csv'

    f = open(output_filename, 'w')
    f.write(output_header)
    f.close()

    for i in range(0, len(conf['config'])):
        config = conf['config'][i]

        lookback = config['lookback']
        features = 5

        # frame as supervised learning
        reframed = series_to_supervised(scaled, lookback)
        print(reframed.head())

        # split into train and test sets
        values = reframed.values

        # Training and Test set sizes
        train_size = int(len(dataset) * config['trainSize'])
        test_size = len(dataset) - train_size

        train, test = values[:train_size, :], values[train_size:, :]
        # split into input and outputs

        # Remove the wash trade feature from the training sets
        washless_train = remove_wash_feat(train, lookback, features)
        washless_test = remove_wash_feat(test, lookback, features)

        n_obs = lookback * features - 1

        # Fill dataset with training and expected
        train_X, train_y = washless_train[:, :], train[:, -1]
        test_X, test_y = washless_test[:, :], test[:, -1]
        print(train_X.shape, len(train_X), train_y.shape)


        # reshape input to be 3D [samples, trades, features]
        train_X = train_X.reshape((train_X.shape[0], lookback, features - 1))
        test_X = test_X.reshape((test_X.shape[0], lookback, features - 1))
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        def build_model():
            # design network
            model = Sequential()
            model.add(InputLayer(input_shape=(train_X.shape[1], train_X.shape[2])))

            layers_list = config['modelLayers']
            for j in range(0, len(layers_list)):
                model.add(layers_list[j])

            model.add(Dense(1))
            model.compile(loss=config['loss'], optimizer=config['optimizer'], metrics=['accuracy'])
            return model


        # fit network
        model = build_model()

        history = model.fit(train_X, train_y, epochs=config['epochs'], batch_size=config['batch_size'],
                            validation_data=(test_X, test_y), verbose=2,shuffle=False)
        # Important note: the validation data does not get used to train the model, it is just used to evaluate the loss
        # function. Alternatively a separate validation set could be used.


        model.save("model.h5")
        print("Saved model to disk")

        # plot history
        # pyplot.plot(history.history['loss'], label='train')
        # pyplot.plot(history.history['val_loss'], label='test')
        # pyplot.legend()
        # pyplot.show()

        # make a prediction
        # yhat = model.predict(test_X)
        yhat = model.predict_classes(test_X, config['batch_size'], verbose=1)
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
        precision = precision_score(inv_y, inv_yhat)
        print(precision)
        recall = recall_score(inv_y, inv_yhat)
        print(recall)
        tn, fp, fn, tp = confusion_matrix(inv_y, inv_yhat).ravel()
        print(confusion_matrix(inv_y, inv_yhat))

        percent_correct = (tn + tp) / test_y.shape[0] * 100
        print(percent_correct)

        csv_row = str(rmse) + ',' + str(precision) + ',' + str(recall) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(tp) + ',' + str(percent_correct) + '\n'
        f = open(output_filename, 'a')
        f.write(csv_row)
        f.close()
        # myemail.email_files(output_filename)


