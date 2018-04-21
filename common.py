from numpy import delete
from pandas import read_csv, DataFrame, concat


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


def clean_data(inputFile):
    dataset = read_csv(inputFile)
    dataset.drop('Ticker', axis=1, inplace=True)
    dataset.drop('Conditions', axis=1, inplace=True)
    dataset.columns = ['EventType', 'Exchange', 'Price', 'Quantity', 'Timestamp', 'Wash']
    dataset.to_csv(ibmFullPre, index=False)
    print(dataset.head(5))


# Common vars
datasetsDir = 'datasets'

ibmbase = datasetsDir + '/IBM - Base.csv'
ibmFull = datasetsDir + '/IBM.csv'
ibmFullPre = datasetsDir + '/IBM_Full_pre.csv'
imbShortPre = datasetsDir + '/IBM_short_pre.csv'

#Testing sets

# The following 2 sets only have wash block sizes of 2
ibm50000 = datasetsDir + '/IBM - 50000.csv'
ibm80000 = datasetsDir + '/IBM - 80000.csv'

ibmMix30000Clean = datasetsDir + '/IBM_Mix_30000.csv'
ibmMix30000CleanNoise = datasetsDir + '/IBM_Mix_30000_30000.csv'
ibm10000 = datasetsDir + '/IBM_10000_180000.csv'
ibm100 = datasetsDir + '/IBM_100.csv'

spybase = datasetsDir + '/SPY.csv'
spy500 = datasetsDir + '/SPY_500_30000.csv'
spy100 = datasetsDir + '/SPY_100.csv'