from nrclex import NRCLex
import nltk
nltk.download('punkt')
import pandas as pd

def get_emotions(df):
    emotions = {'fear': [], 'anger': [], 'anticipation': [], 'trust': [], 'surprise': [], 'positive': [],
              'negative': [], 'sadness': [], 'disgust': [], 'joy': []}
    for index, row in df.iterrows():
        tweet = row['Text']
        text_object = NRCLex(tweet)
        length_sentence = len(tweet.split())
        absolute_numbers = text_object.raw_emotion_scores #just a dictionary, similiar to one above
        for emot in emotions:
            try:
                val = absolute_numbers[emot]/length_sentence
            except:
                val = 0
            emotions[emot].append(val)
    for e in emotions:
        df[e] = emotions[e]
    df.to_csv('/Users/gabriel/PycharmProjects/Finance/Data/4B-Ungrouped non-ML sentiment/FB_1d_no_ML.csv')
    return True




df = pd.read_csv('/Users/gabriel/PycharmProjects/Finance/Data/3-Tweet_and_price/FB_1d.csv')
df = df.iloc[:, :-1]
get_emotions(df)