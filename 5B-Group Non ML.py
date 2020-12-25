import pandas as pd
import datetime as datetime


def read_data(path):
  df = pd.read_csv(path)
  print("Reading CSV file...")
  df = df.drop('Date', 1)
  df['Date'] = [datetime.datetime.strptime(d[0:18], "%Y-%m-%d %H:%M:%S") for d in df["Rounded_Date"]]
  df['Day'] = [datetime.datetime.date(d) for d in df["Date"]]
  df['Time'] = [datetime.datetime.time(d) for d in df["Date"]]
  df = df.set_index(['Day','Time'])
  df = df.drop('Date', 1)
  return df

def Cull_Quiet_Intervals(df, threshold):
    size = []
    x= 0
    print("Culling intervals below threshold....")
    new_df = df
    cnt = 0
    removed_cnt = 0
    for date, small_df in new_df.groupby(level='Day'):
      x += 1
      if small_df.shape[0] < threshold:
        x -= 1
        cnt +=1
        removed_cnt += small_df.shape[0]
        new_df = new_df.drop(date, level=0)
      size.append(small_df.shape[0])
    print("got rid of ", cnt, "intervals resulting into ", removed_cnt, "tweets removed, so you have ", x, " intervals left")
    print('Avg: ', sum(size)/len(size), ' min: ', min(size), ' max: ', max(size))
    return new_df
def avg(x):
  return sum(x)/len(x)


def Final_DF(file_path, cull_threshold):
    df = read_data(file_path)
    df = Cull_Quiet_Intervals(df, cull_threshold)  # Removes intervals below this threshold
    columns = ['Time', 'fear','anger', 'anticipation','trust', 'suprise',
               'positive', 'negative', 'sadness','disgust','joy',
               'Volume_of_tweets','Retweet','Replies','Likes',
               'Close', 'Open']
    final_df = pd.DataFrame(columns=columns)
    i = 0
    for interval, new_df in df.groupby(level=0):
        fear = 0
        anger = 0
        anticipation =0
        trust = 0
        suprise =0
        pos = 0
        neg = 0
        sadness = 0
        disgust = 0
        joy = 0
        daily_volume = 0
        likes= 0
        retweets= 0
        replies= 0
        i = i + 1
        for index, row in new_df.iterrows():
            daily_volume += 1
            pos += row['positive']
            neg += row['negative']
            likes += row['Likes']
            retweets += row['Retweet']
            replies += row['Replies']
            fear += row['fear']
            anger += row['anger']
            anticipation += row['anticipation']
            trust += row['trust']
            suprise += row['surprise']
            sadness += row['sadness']
            disgust += row['disgust']
            joy += row['joy']
        rows = {'Time': interval,
            'positive': pos / daily_volume,
            'negative': neg / daily_volume,
            'fear': fear / daily_volume,
            'anger': anger / daily_volume,
            'anticipation': anticipation / daily_volume,
            'trust': trust / daily_volume,
            'suprise': suprise / daily_volume,
            'sadness': sadness / daily_volume,
            'disgust': disgust / daily_volume,
            'joy': joy / daily_volume,
            'Volume_of_tweets': daily_volume,
            'Retweet': retweets/ daily_volume,
            'Replies': replies /daily_volume,
            'Likes': likes/daily_volume,
            'Close': avg(new_df['close_price']),
            'Open': avg(new_df['open_price'])}
        final_df = final_df.append(rows, ignore_index=True)
    final_df.to_csv('/Users/gabriel/PycharmProjects/Finance/Data/5B- grouped final results no ML/FB_1d_final_results_no_ML.csv')


path = '/Users/gabriel/PycharmProjects/Finance/Data/4B-Ungrouped non-ML sentiment/FB_1d_no_ML.csv'
Final_DF(path, 0)

