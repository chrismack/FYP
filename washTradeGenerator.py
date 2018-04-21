import numpy as np
from pandas import read_csv
from pandas import DataFrame
import random
import common

output = common.ibmMix30000CleanNoise

#Random Trades Count
randomTradesCount = 60000

rnd_min_vol = 40
rnd_max_vol = 1400


# Number of washtrades to generate
washBlockCount = 30000
minimumSeperation = 5

minBids = 1
maxBids = 7

minSell = 1
maxSell = 9

maxVolVar = 4
minVol = 1 + maxVolVar
maxVol = 12 - maxVolVar

exchanges = ['ARCA', 'BATS', 'BATS Y', 'CSE', 'EDGA', 'EDGX', 'FINRA', 'NASDAQ', 'NASDAQ BX', 'NASDAQ PSX', 'NYSE']
eventTypes = ['QUOTE BID', 'QUOTE ASK']

# -1: bid and ask, 0: bid first, 1: ask first
order = -1

def random_range(n, total):
    randoms = [0] * n
    add_count = 0

    while add_count < total:
        rnd_index = random.randint(0, n - 1)
        randoms[rnd_index] += 1
        add_count += 1

    return randoms


newSet = DataFrame()
oldData = DataFrame()

def insertTrade(dataset, washTrade, row):
    cols = ['Timestamp', 'EventType', 'Price', 'Quantity', 'Exchange', 'Wash', 'Conditions', 'Ticker']

    global newSet

    df1 = dataset.iloc[0:row]
    newSet = newSet.append(df1)

    dataset = dataset.iloc[row: dataset.shape[0]]

    # print(newSet.shape[0])

    # print(df1.shape[0])
    # print(df2.shape[0])
    # print("Total: ", df1.shape[0] + df2.shape[0])

    dfs = []

    tradeLength = len(washTrade['EventType'])
    for i in range(tradeLength):
        df = DataFrame([[
            washTrade[cols[0]][i],
            washTrade[cols[1]][i],
            washTrade[cols[2]][i],
            washTrade[cols[3]][i],
            washTrade[cols[4]][i],
            washTrade[cols[5]][i],
            '1',
            'IBM'
        ]], columns=cols)
        newSet = newSet.append(df)
        # df1 = df1.concat(df)

    # df1 = df1.append(df2)
    # df1 = df1.append(df2)

    return dataset

# def insertTrade(dataset, washTrade, row):
#     cols = ['Timestamp', 'EventType', 'Price', 'Quantity', 'Exchange', 'Wash', 'Conditions', 'Ticker']
#
#     df1 = dataset.iloc[0:row]
#     df2 = dataset.iloc[row: dataset.shape[0]]
#
#     print(df1.shape[0])
#     print(df2.shape[0])
#     print("Total: ", df1.shape[0] + df2.shape[0])
#
#     dfs = []
#
#     tradeLength = len(washTrade['EventType'])
#     for i in range(tradeLength):
#         df = DataFrame([[
#             washTrade[cols[0]][i],
#             washTrade[cols[1]][i],
#             washTrade[cols[2]][i],
#             washTrade[cols[3]][i],
#             washTrade[cols[4]][i],
#             washTrade[cols[5]][i],
#             1,
#             'IBM'
#         ]], columns=cols)
#         df1 = df1.append(df)
#         # df1 = df1.concat(df)
#
#     # df1 = df1.append(df2)
#     df1 = df1.append(df2)
#
#     return df1


def generateWashTrades(rows, dataset):
    # Columns Timestamp, EventType, Price Quantity, Exchange
    global newSet
    previous = 0
    tradeSets = []


    eventType = -1

    for i in range(0, len(rows)):
        print(i)
        if (order < 0):
            eventType = random.randint(0, 1)
        else:
            eventType = order

        row = dataset.iloc[rows[i] - previous]
        previousRow = dataset.iloc[rows[i] - previous - 1]
        averagePrice = (row.Price + previousRow.Price) / 2

        sellCount = random.randint(minSell, maxSell)
        bidCount = random.randint(minBids, maxBids)

        counts = [bidCount, sellCount]

        sellVolume = random.randint(minVol, maxVol) * 100
        buyVolume = random.randint(sellVolume - maxVolVar, sellVolume + maxVolVar)

        rndSellVols = random_range(sellCount, sellVolume)
        rndBuyVols = random_range(bidCount, buyVolume)
        rnd_vols = [rndBuyVols, rndSellVols]

        volumes = [buyVolume, sellVolume]

        exchange = exchanges[random.randint(0, len(exchanges) - 1)]


        sellData = {'Timestamp': [], 'EventType': [], 'Price': [], 'Quantity': [], 'Exchange': [], 'Wash': []}
        buyData = {'Timestamp': [], 'EventType': [], 'Price': [], 'Quantity': [], 'Exchange': [], 'Wash': []}

        firstEvent = eventType
        firstCount = sellCount if eventType == 1 else bidCount

        for j in range(0, sellCount + bidCount):
            if (j == firstCount):
                eventType = 0 if eventType == 1 else 1
            sellData['Timestamp'].append(row.Timestamp)
            sellData['EventType'].append(eventTypes[eventType])
            sellData['Price'].append(float(format(averagePrice, '.2f')))

            if firstEvent != eventType:
                otherEvent = 0 if eventType == 1 else 1
                offset = len(rnd_vols[otherEvent])
                vol = rnd_vols[eventType][j - offset]
            else:
                vol = rnd_vols[eventType][j]

            sellData['Quantity'].append(int(vol))
            sellData['Exchange'].append(exchange)
            sellData['Wash'].append(1)

        # print(sellData)
        # print(buyData)
        # merged = {'Timestamp': buyData['Timestamp'] + sellData['Timestamp'],
        #           'EventType': buyData['EventType'] + sellData['EventType'],
        #           'Price': buyData['Price'] + sellData['Price'],
        #           'Quantity': buyData['Quantity'] + sellData['Quantity'],
        #           'Exchange': buyData['Exchange'] + sellData['Exchange']
        #           }
        tradeSets.append(sellData)
        dataset = insertTrade(dataset, sellData, rows[i] - previous)
        previous = rows[i]
    newSet = newSet.append(dataset.iloc[:])
    dataset = newSet
    dataset.to_csv(output, index=False)


# Rows to place a fake trade between
def findInsertRows(dataset):
    rows = []
    for i in range(0, washBlockCount):
        unique = False
        while not unique:
            unique = False
            row = random.randint(1, dataset.shape[0] - 1)
            if (row not in rows):

                meetsSeperation = False
                for j in range(row - minimumSeperation, row + minimumSeperation):
                    if (j not in rows):
                        meetsSeperation = True

                if (meetsSeperation):
                    rows.append(row)
                    unique = True
            else:
                unique = False

    return sorted(rows)

def generateRandomTrade(rows, dataset):
    global newSet
    previous = 0
    for i in range(0, len(rows)):
        print(i)
        row = dataset.iloc[rows[i] - previous]
        previousRow = dataset.iloc[rows[i] - previous - 1]
        averagePrice = (row.Price + previousRow.Price) / 2
        event = random.choice(eventTypes)
        volume = random.randint(rnd_min_vol, rnd_max_vol)
        exchange = random.choice(exchanges)

        tradeData = {'Timestamp': [], 'EventType': [], 'Price': [], 'Quantity': [], 'Exchange': [], 'Wash': []}
        tradeData['Timestamp'].append(row.Timestamp)
        tradeData['EventType'].append(event)
        tradeData['Price'].append(float(format(averagePrice, '.2f')))
        tradeData['Quantity'].append(int(volume))
        tradeData['Exchange'].append(exchange)
        tradeData['Wash'].append(0)

        dataset = insertTrade(dataset, tradeData, rows[i] - previous)
        previous = rows[i]
    newSet = newSet.append([dataset])
    dataset = newSet
    dataset.to_csv(output, index=False)


def findRandomRows(dataset):
    return sorted([random.randint(0, dataset.shape[0] - 1) for _ in range(0, randomTradesCount)])



# dataset = read_csv(common.ibmbase)
dataset = read_csv(output)
# generateRandomTrade(findRandomRows(dataset), dataset)
generateWashTrades(findInsertRows(dataset), dataset)
