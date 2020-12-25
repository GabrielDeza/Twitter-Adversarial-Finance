from finbert.finbert import predict
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
import argparse
import nltk
nltk.download('punkt')
from pathlib import Path
import datetime
import os
import random
import string
'''
python predict.py --text_path Tweets_Finbert.txt --output_dir output/ --model_path models_sentiment/Final_Dir
parser = argparse.ArgumentParser(description='Sentiment analyzer')
parser.add_argument('-a', action="store_true", default=False)
parser.add_argument('--text_path', type=str, help='Path to the text file.')
parser.add_argument('--output_dir', type=str, help='Where to write the results')
parser.add_argument('--model_path', type=str, help='Path to classifier model')
args = parser.parse_args()
'''


text_path = '/Users/gabriel/PycharmProjects/Finance/Data/4-Finbert-txt/FB_1d.txt'
model_path = '/Users/gabriel/PycharmProjects/Finance/models/sentiment'
output_dir = '/Users/gabriel/PycharmProjects/Finance/Data/5-Finbert-sentiment-txt/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
import logging

with open(text_path,'r') as f:
    text = f.read()
model = BertForSequenceClassification.from_pretrained(model_path,num_labels=3,cache_dir=None)
random_filename = ''.join(random.choice(string.ascii_letters) for i in range(10))
output = random_filename + '.csv'
logger = logging.getLogger('my-logger')
logger.propagate = False
predict(text,model,write_to_csv=True,path=os.path.join(output_dir,output))