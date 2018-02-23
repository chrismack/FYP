from pandas import read_csv
from pandas import DataFrame
import random
import common

# Number of washtrades to generate
washBlockCount = 10
minimumSeperation = 1000

minBids = 1
maxBids = 1

minSell = 1
maxSell = 1


maxVolVar = 50
minVol = 100 + maxVolVar
maxVol = 1000 - maxVolVar

exchanges = ['ARCA', 'BATS', 'BATS Y', 'CSE', 'EDGA', 'EDGX', 'FINRA', 'NASDAQ', 'NASDAQ BX', 'NASDAQ PSX', 'NYSE']


# -1: bid and ask, 0: bid first, 1: ask first
order = -1


def insertWashTrade(washTrade):
    return None


def generateWashTrades(rows, dataset):
    # Columns Timestamp, EventType, Price Quantity, Exchange

    eventTypes = ['QUOTE BID', 'QUOTE ASK']
    eventType = -1
    if(order < 0):
        eventType = random.randint(0,1)
    else:
        eventType = order


    for i in range(0, len(rows)):
        row = dataset.iloc[rows[i]]
        previousRow = dataset.iloc[rows[i] - 1]
        averagePrice = (row.Price + previousRow.Price) / 2

        sellCount = random.randint(minSell, maxSell)
        bidCount = random.randint(minBids, maxBids)

        sellVolume = random.randint(minVol, maxVol)
        buyVolume = random.randint(sellVolume - maxVolVar, sellVolume + maxVolVar)

        exchange = exchanges[random.randint(0, len(exchanges) - 1)]

        sellData = {'Timestamp': [], 'EventType': [], 'Price': [], 'Quantity': [], 'Exchange': []}
        buyData = {'Timestamp': [], 'EventType': [], 'Price': [], 'Quantity': [], 'Exchange': []}
        for j in range(0, sellCount + bidCount):
            if(j < sellCount):
                # create sells
                sellData['Timestamp'].append(row.Timestamp)
                sellData['EventType'].append(eventTypes[1])
                sellData['Price'].append(float(format(averagePrice, '.2f')))
                sellData['Quantity'].append(sellVolume / sellCount)
                sellData['Exchange'].append(exchange)
            else:
                # create buys
                buyData['Timestamp'].append(row.Timestamp)
                buyData['EventType'].append(eventTypes[0])
                buyData['Price'].append(float(format(averagePrice, '.2f')))
                buyData['Quantity'].append(buyVolume / bidCount)
                buyData['Exchange'].append(exchange)

        print(sellData)


    #line = DataFrame({"onset": 30.0, "length": 1.3}, index=[3])
    #df2 = concat([df.ix[:2], line, df.ix[3:]]).reset_index(drop=True)

    return None


# Rows to place a fake trade between
def findInsertRows(dataset):
    rows = []
    for i in range(0, washBlockCount):
        unique = False
        while not unique:
            unique = False
            row = random.randint(1, dataset.shape[0] - 1)
            if(row not in rows):

                meetsSeperation = False
                for j in range(row - minimumSeperation, row + minimumSeperation):
                    if(j not in rows):
                        meetsSeperation = True

                if(meetsSeperation):
                    rows.append(row)
                    unique = True
            else:
                unique = False

    return rows

dataset = read_csv(common.ibmFullPre)

generateWashTrades(findInsertRows(dataset), dataset)





