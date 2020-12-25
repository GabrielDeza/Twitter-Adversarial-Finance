import json
import pandas as pd
import os
import csv
import requests


def to_json(ticker_symbol, start, end, interval):
    """
    Constructs URL to get data from yahoo finance to retrieve and save to a JSON file.
    :param ticker_symbol: Ticker Symbol to look for (STR)
    :param start: Start date in YYYY-MM-DD (STR)
    :param end: End date in YYYY-MM-DD (STR)
    :param interval: interval to source data from (STR). ex: 1m,2m,5m,1h,1d
    :return: JSON File path, file name, end date in YYYY-MM-DD
    """
    #Getting URL
    domain = "https://query1.finance.yahoo.com/v8/finance/chart/"
    symbol = ticker_symbol + "?symbol=" + ticker_symbol
    Unix_since = (pd.to_datetime([start]) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    Unix_end = (pd.to_datetime([end]) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    time = "&period1=" + str(Unix_since[0]) + "&period2=" + str(Unix_end[0]) + "&interval=" + interval
    end_bit = "&includePrePost=true&events=div%7Csplit%7Cearn&lang=en-US&region=US&crumb=iNgynQL1Qys&corsDomain=finance.yahoo.com"
    url = domain+symbol+time+end_bit
    #Getting JSON data from URL and saving it to a local file
    filename = ticker_symbol + "_" +interval + ".json"
    file_path = '/Users/gabriel/PycharmProjects/Finance/Data/Stock_Prices/'+ filename
    r = requests.get(url, stream=True)
    if r.ok:
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        abs_file_path = file_path
        return abs_file_path, ticker_symbol + "_" +interval
    else:
        print("woops seems it didn't work")
        return None, None


def json_to_csv(file_path, file_name):
    """
    Given json file of stock data scraped from yahoo finance, converts it into a useful CSV.
    :param file_path: path to the JSON file of interest (STR)
    :param file_name: name of JSON of interest which will share the same name as csv
    :return:
    """
    if file_path == None:
        print("getting JSON failed so can't save to csv")
        return None
    data  = json.load(open(file_path, 'r'))
    data = data["chart"]["result"]

    data = data[0]  # Now data is a Dict with 3 keys: Meta, Timestamp, Indicators
    Indicators = data["indicators"]["quote"][0]
    high = Indicators["high"]
    close = Indicators["close"]
    low = Indicators["low"]
    open_price = Indicators["open"]
    volume = Indicators["volume"]
    timestamp = data["timestamp"]
    for i in range(0, len(timestamp)):
        timestamp[i] = pd.Timestamp(timestamp[i], unit='s', tz='America/New_York').to_pydatetime()

    with open(os.getcwd() + "/Data/Stock_Prices/" + file_name + ".csv", "w", encoding = "utf-8") as csvfile:
        fieldnames = ["Time", "Open", "Close", "Low", "High", "Volume"]
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames, lineterminator = "\n")
        writer.writeheader()
        for i in range(0,len(timestamp)):
            writer.writerow({"Time": timestamp[i],
                         "Open": open_price[i],
                         "Close": close[i],
                         "Low": low[i],
                         "High": high[i],
                         "Volume": volume[i]})
path, x = to_json('FB','2017-01-01','2020-01-01','1d')
json_to_csv(path,x)