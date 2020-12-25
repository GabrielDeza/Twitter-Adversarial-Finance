import pandas as pd
import re

def Convert_to_txt(file_path):
    df = pd.read_csv(file_path)
    file = open('/Users/gabriel/PycharmProjects/Finance/Data/4-Finbert-txt/FB_1d.txt', "a")
    for index, row in df.iterrows():
        tweet = row['Text']
        tweet = re.sub('\.', '', tweet)
        tweet = re.sub('\?', '', tweet)
        tweet = re.sub('!', '', tweet)
        tweet = re.sub('#', '', tweet)
        tweet = re.sub('\@', '', tweet)
        tweet = tweet + '     .'
        row['Text'] = tweet
        file.write(tweet + '\n')
    file.close()


Convert_to_txt('/Users/gabriel/PycharmProjects/Finance/Data/3-Tweet_and_price/FB_1d.csv')

'''
python predict.py --text_path Tweets_Finbert.txt --output_dir output/ --model_path models_sentiment/Final_Dir
'''