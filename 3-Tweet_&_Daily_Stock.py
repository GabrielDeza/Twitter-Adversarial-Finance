from datetime import time, datetime, timedelta
import pandas as pd
from tqdm import tqdm
def Preparing_Tweets(tweet_path, stock_path):
    end = time.fromisoformat('09:30:00')
    start = time.fromisoformat('16:00:00')
    open_price = []
    Rounded_Date = []
    close_price = []
    cnt = 0
    stock_df = pd.read_csv(stock_path)
    stock_df['Time'] = [d[0:19] for d in stock_df['Time']]
    stock_df['Time'] = pd.to_datetime(stock_df['Time'])
    stock_df['Time'] = stock_df['Time'].dt.to_pydatetime()
    stock_df.set_index('Time', inplace=True)
    #Date, Retweet, Replies, Likes, Text
    master_df = pd.DataFrame(columns=['Date','Rounded_Date','Retweet','Replies','Likes', 'Text', 'open_price', 'close_price'])
    df = pd.read_csv(tweet_path)
    for i, row in tqdm(df.iterrows()):
                cnt += 1
                pred_date = None
                close_date = None
                timestamp = datetime.strptime(row["Date"][0:18], '%Y-%m-%d %H:%M:%S')
                day = timestamp.weekday()
                hour = timestamp.time()
                #Predicting monday morning if:
                # tweet is on weekend
                # tweet is on friday after 4:00 PM
                #tweet is on monday before 9:30 AM
                if (day == 4) and (hour > start):
                    pred_date = timestamp + timedelta(days = 3)
                    close_date = timestamp
                if day == 5:
                    pred_date = timestamp + timedelta(days = 2)
                    close_date = timestamp - timedelta(days = 1)
                if day == 6:
                    pred_date = timestamp + timedelta(days = 1)
                    close_date = timestamp - timedelta(days=1)
                # Happens after market but before midnight
                if ((day in [0,1,2,3]) and (hour > start)):
                    pred_date = timestamp + timedelta(days = 1)
                    close_date = timestamp
                # Happens after midnight but before market reopens
                if ((day in [1,2,3,4]) and (hour < end)):
                    pred_date = timestamp
                    close_date = timestamp - timedelta(days=1)
                if pred_date != None:
                    pred_time = datetime.combine(pred_date.date(), end)
                    close_time = datetime.combine(close_date.date(), end)
                    master_df = master_df.append(row)
                    open_price.append(stock_df.iloc[stock_df.index.get_loc(pred_time, method='nearest')]['Open'])
                    close_price.append(stock_df.iloc[stock_df.index.get_loc(close_time, method='nearest')]['Close'])
                    Rounded_Date.append(pred_time)
    master_df['open_price'] = open_price
    master_df['close_price'] = close_price
    master_df['Rounded_Date'] = Rounded_Date
    final_file_path = '/Users/gabriel/PycharmProjects/Finance/Data/3-Tweet_and_price/FB_1d.csv'
    master_df.to_csv(final_file_path, index=False)
    print("Went from ", cnt, " to ", master_df.shape[0], "tweets")

tweet_path = '/Users/gabriel/PycharmProjects/Finance/Data/1-Raw_Tweets/Facebook_Combined.csv'
stock_path = '/Users/gabriel/PycharmProjects/Finance/Data/2-Stock_Prices/FB_1d.csv'
Preparing_Tweets(tweet_path, stock_path)