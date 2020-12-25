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
    columns = ['Time', 'General_score','Pos_perc', 'neg_perc', 'Neutral_perc',
               'Pos_score', 'neg_score', 'Neutral_score',
               'Volume_of_tweets','Retweet','Replies','Likes',
               'Close', 'Open']

    final_df = pd.DataFrame(columns=columns)
    i = 0
    for interval, new_df in df.groupby(level=0):
        gen_score = 0
        pos_cnt = 0
        neg_cnt = 0
        neut_cnt = 0
        pos = 0
        neg = 0
        neut = 0
        daily_volume = 0
        likes= 0
        retweets= 0
        replies= 0
        i = i + 1
        for index, row in new_df.iterrows():
            daily_volume += 1
            pos += row['Positive']
            neg += row['Negative']
            neut += row['Neutral']
            likes += row['Likes']
            retweets += row['Retweet']
            replies += row['Replies']
            gen_score += row['score']
            if row['Sentiment'] == 'neutral':
                neut_cnt += 1
            if row['Sentiment'] == 'positive':
                pos_cnt += 1
            if row['Sentiment'] == 'negative':
                neg_cnt += 1
        row = {'Time': interval,
            'General_score': gen_score / daily_volume,
            'Pos_perc': pos_cnt /daily_volume,
            'neg_perc': neg_cnt / daily_volume,
            'Neutral_perc': neut_cnt / daily_volume,
            'Pos_score': pos / daily_volume,
            'neg_score': neg / daily_volume,
            'Neutral_score': neut / daily_volume,
            'Volume_of_tweets': daily_volume,
            'Retweet': retweets/ daily_volume,
            'replies': replies /daily_volume,
            'Likes': likes/daily_volume,
            'Close': avg(new_df['close_price']),
            'Open': avg(new_df['open_price'])}
        final_df = final_df.append(row, ignore_index=True)
    final_df.to_csv('/Users/gabriel/PycharmProjects/Finance/Data/7-Grouped final results/Facebook_Final.csv')


path = '/Users/gabriel/PycharmProjects/Finance/Data/6-Ungrouped_final_results/Facebook_1d.csv'
Final_DF(path, 0)