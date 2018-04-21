from keras.layers import LSTM, CuDNNLSTM, Dense
from keras.optimizers import SGD
from keras.optimizers import Nadam
from keras.layers.wrappers import Bidirectional, TimeDistributed

ran = [
    {
        'confId': 'lookback-newset',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 1,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 2,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 3,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 4,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 5,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 6,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 7,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 9,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 10,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 12,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 13,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 14,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 15,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 20,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 30,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 50,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 100,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
        ]
    },
    {
        'confId': 'optimizers-newset',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'sgd',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'rmsprop',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adagrad',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adadelta',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'adamax',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 100,
                'batch_size': 300
            }
        ]
    },
    {
        'confId': 'sgd-learning-rate-newset',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=1, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.00001, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.000001, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.0000001, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=1.5, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=2, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=5, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=10, momentum=0.0, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },

        ]
    },
    {
        'confId': 'sgd-learning-rate-momentum-newset',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=1, momentum=0.5, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.1, momentum=0.5, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.001, momentum=0.5, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': SGD(lr=0.0001, momentum=0.5, decay=0.0, nesterov=False),
                'epochs': 100,
                'batch_size': 300
            },
        ]
    },
    {
        'confId': 'nadam-learning-rate-newset',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': Nadam(lr=1),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': Nadam(lr=0.1),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': Nadam(lr=0.01),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': Nadam(lr=0.001),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': Nadam(lr=0.0001),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': Nadam(lr=0.00001),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': Nadam(lr=0.000001),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': Nadam(lr=1.5),
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': Nadam(lr=2),
                'epochs': 100,
                'batch_size': 300
            },
        ]
    },
    {
        'confId': 'epoch-newset',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 1,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 2,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 3,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 4,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 5,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 6,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 7,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 8,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 9,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 10,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 20,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 30,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 100,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 200,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 500,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 1000,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 2000,
                'batch_size': 300
            },
        ]
    },
    {
        'confId': 'layer-2-newset',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(1)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(2)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(3)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(4)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(5)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(6)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(7)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(8)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(9)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(10)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(20)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(30)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(40)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(50)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(100)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(150)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
        ]
    },
    {
        'confId': 'layer-2-50-newset',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(1)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(2)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(3)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(4)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(5)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(6)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(7)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(8)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(9)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(10)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(20)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(30)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(40)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(50)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(100)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 8,
                'modelLayers': [
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(150)
                ],
                'loss': 'mae',
                'optimizer': 'adam',
                'epochs': 50,
                'batch_size': 300
            },
        ]
    },
    {
        'confId': 'loss-functions',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mse',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'mape',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    LSTM(10),
                ],
                'loss': 'msle',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(10),
                ],
                'loss': 'squared_hinge',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(10),
                ],
                'loss': 'hinge',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(10),
                ],
                'loss': 'binary_crossentropy',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(10),
                ],
                'loss': 'kld',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(10),
                ],
                'loss': 'poisson',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(10),
                ],
                'loss': 'cosine_proximity',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
        ]
    },
    {
        'confId': 'layer-wrappers',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    Bidirectional(CuDNNLSTM(100, return_sequences=True)),
                    Bidirectional(CuDNNLSTM(10)),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    Bidirectional(CuDNNLSTM(50, return_sequences=True)),
                    Bidirectional(CuDNNLSTM(10)),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    Bidirectional(CuDNNLSTM(100, return_sequences=True)),
                    Bidirectional(CuDNNLSTM(50)),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    Bidirectional(CuDNNLSTM(200, return_sequences=True)),
                    Bidirectional(CuDNNLSTM(100)),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    Bidirectional(CuDNNLSTM(200, return_sequences=True)),
                    CuDNNLSTM(100),
                ],
                'loss': 'mae',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
        ]
    },
    {
        'confId': 'layer-3-newset',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(10)
                ],
                'loss': 'mse',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(20)
                ],
                'loss': 'mse',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(30)
                ],
                'loss': 'mse',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(40)
                ],
                'loss': 'mse',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    CuDNNLSTM(100, return_sequences=True),
                    CuDNNLSTM(50, return_sequences=True),
                    CuDNNLSTM(50)
                ],
                'loss': 'mse',
                'optimizer': 'nadam',
                'epochs': 50,
                'batch_size': 300
            },

        ]
    },
    {
        'confId': 'finals',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    Bidirectional(CuDNNLSTM(100, return_sequences=True)),
                    Bidirectional(CuDNNLSTM(50))
                ],
                'loss': 'mse',
                'optimizer': Nadam(lr=0.01),
                'epochs': 2000,
                'batch_size': 300
            },
        ]
    }

]

confs = [
    {
        'confId': 'finals',
        'config': [
            {
                'trainSize': 0.7,
                'lookback': 11,
                'modelLayers': [
                    Bidirectional(CuDNNLSTM(100, return_sequences=True)),
                    Bidirectional(CuDNNLSTM(50))
                ],
                'loss': 'mse',
                'optimizer': Nadam(lr=0.01),
                'epochs': 200,
                'batch_size': 100
            },
            # {
            #     'trainSize': 0.7,
            #     'lookback': 11,
            #     'modelLayers': [
            #         CuDNNLSTM(100, return_sequences=True),
            #         CuDNNLSTM(50),
            #     ],
            #     'loss': 'mse',
            #     'optimizer': Nadam(lr=0.01),
            #     'epochs': 100,
            #     'batch_size': 50
            # },
            #
            # {
            #     'trainSize': 0.7,
            #     'lookback': 11,
            #     'modelLayers': [
            #         Bidirectional(CuDNNLSTM(100, return_sequences=True)),
            #         Bidirectional(CuDNNLSTM(50))
            #     ],
            #     'loss': 'mse',
            #     'optimizer': Nadam(lr=0.01),
            #     'epochs': 2000,
            #     'batch_size': 50
            # },
            # {
            #     'trainSize': 0.7,
            #     'lookback': 11,
            #     'modelLayers': [
            #         CuDNNLSTM(100, return_sequences=True),
            #         CuDNNLSTM(50)
            #     ],
            #     'loss': 'mse',
            #     'optimizer': Nadam(lr=0.01),
            #     'epochs': 2000,
            #     'batch_size': 50
            # },

        ]
    }

]
