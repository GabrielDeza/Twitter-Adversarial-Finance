import pandas as pd

path2 = '/Users/gabriel/PycharmProjects/Finance/Data/3-Tweet_and_price/FB_1d.csv'  #23K tweets without sentiment
path1 = '/Users/gabriel/PycharmProjects/Finance/Data/5-Finbert-sentiment-txt/FB_1d.csv' #23K Tweets with sentiment
df_sentiment = pd.read_csv(path1)
df_normal = pd.read_csv(path2)

pos_score = []
neg_score = []
neut_score = []
sentiment = []
score = []
print(df_sentiment.shape[0])
for i in range(0,df_sentiment.shape[0]):
    if i %20000 == 0:
        print('done ',i, ' so far')
    logit = df_sentiment['logit'][i].split()
    pos_int = float(logit[0][1:])
    neg_int = float(logit[1])
    neut_int = float(logit[2][:-1])
    pos_score.append(pos_int)
    neg_score.append(neg_int)
    neut_score.append(neut_int)
    sentiment.append(df_sentiment['prediction'][i])
    score.append(df_sentiment['sentiment_score'][i])
df_normal['Positive'] = pos_score
df_normal['Negative'] = neg_score
df_normal['Neutral'] = neut_score
df_normal['Sentiment'] = sentiment
df_normal['score'] = score
df_normal.to_csv('/Users/gabriel/PycharmProjects/Finance/Data/6-Ungrouped_final_results/Facebook_1d.csv')








