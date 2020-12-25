import snscrape.modules
import csv
import re
from tqdm import tqdm

def part1(query, since, until, file_name,min_likes)
    text_input = 'facebook OR fb OR Mark Zuckerberg since:2016-01-01 until:2018-04-03 lang:en'
    file_name = '/Data/1-Raw_Tweets/Facebook_2ndhalf.csv'
    with open(file_name,'w',encoding='utf-8') as csvfile:
        fieldnames = ["Date","Retweet","Replies","Likes","Text"]
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames, lineterminator= "\n")
        writer.writeheader()
        for i, tweet in enumerate(tqdm(snscrape.modules.twitter.TwitterSearchScraper(text_input).get_items())):
            if tweet.lang == 'en' and tweet.likeCount > 4:
                text = tweet.content
                date = tweet.date
                likes = tweet.likeCount
                replies = tweet.replyCount
                retweet = tweet.retweetCount
                text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
                writer.writerow({"Date":date,
                                 "Retweet":retweet,
                                 "Replies":replies,
                                 "Likes":likes,
                                 "Text":text})

