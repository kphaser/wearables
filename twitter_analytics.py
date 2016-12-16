import os
import json
import cPickle as pickle
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import re

### STREAM IN TWITTER DATA
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time

ckey = 'AV2gZrH0nfzRjufrEIIPgZuav'
csecret = '7DU5lJsVu9v1xHR9J4iYviOawlXwpioGSalV5P5itl3wERY17e'
atoken = '4328455754-rxyG4CnD50crNALSRlyn2vOzQxbaJnnUsZbPw3I'
asecret = 'CMjvZgSZunrgIwYLsiPlq7mldw8bcpRJ2SZYcfQPq1rzu'

class listener(StreamListener):

    def on_data(self, data):
        try:
            print(data)
            saveFile = open('twitdb.txt', 'a')
            saveFile.write(data)
            saveFile.write('\n')
            saveFile.close()
            return True
        except BaseException as e:
            print('failed on_data,', str(e))
            time.sleep(5)


    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track=["[cC]harge[Hh][rR]", "ChargeHR", 'chargehr', 'chargeHR', 'fitbit', 'jawbone'])


# READ IN COLLECTED TWITTER DATA
tweets_data = []
tweets_file = open('twitdb.txt', 'r')
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue
        
### PRELIMINARY TWITTER ANALYSIS
# just to check we have the number of tweets we expect
print len(tweets_data) 

# see what the keys are in tweets
tweets_data[0].keys()

# create dataframe and important columns into tweet dataframe to analyze
tweets = pd.DataFrame()
tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)

# plot top 5 languages of tweets
tweets_by_lang = tweets['lang'].value_counts()
fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Languages', fontsize=15)
ax.set_ylabel('Number of Tweets', fontsize=15)
ax.set_title('Top 5 Languages', fontsize=15, fontweight='bold')
tweets_by_lang.plot(ax=ax, kind='bar', color='red')

# plot top 5 countries
tweets_by_country = tweets['country'].value_counts()
fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Countries', fontsize=15)
ax.set_ylabel('Number of Tweets', fontsize=15)
ax.set_title('Top 5 Countries', fontsize=15, fontweight='bold')
tweets_by_country.plot(ax=ax, kind='bar', color='blue')

### USING REGULAR EXPRESSION TO SEARCH FOR ISSUES IN TWEETS
str_style_issues = [r"[Dd]idn't.*[sS]tyl",r"[Nn]eed.*style",r"[Ww]ant.*stylish",r"[wW]ish.*style",r"[nN]ot.*stylish",r"[Uu]gly",r"[wW]ant.*color"]
str_comfort_issues = [r'[wW]ish.*comfort',r'[nN]ot.*[Cc]omfort',r'[oO]uch',r'[uU]ncomfortable',r'[bB]ulky',r"[wW]asn't.*comfortable",r"[tT]ight"]
str_cost_issues = [r'[eE]xpens*',r'[mM]oney',r'[cC]ost too',r'[wW]allet']
str_setup_issues = [r'[wWish].*[s]etup',r'[pP]ortal',r'[sS]etup.*difficult',r'[sS]etup.*hard',r'[cC]ouldnt.*[s]etup',r'[uU]se',r'[sS]etup',r'[nN]ot.*[eE]asy',r'[uU]nintuitive',r'[iI]nstruction*']
str_feature_issues = [r'[hH]ard.*portal',r"[cC]ouldn't.*portal",r'[nN]ot Interested',r'[mM]eh',r'[oO]kay']

# turn text key into series
text=tweets['text']

# count each issue and append to the issues data frame
issues_columns = ['Issue','Count']
issues_df = pd.DataFrame(columns = issues_columns)

def count_issues(str_issue_word_list, str_issue_title):
    issue_count = 0
    for item in str_issue_word_list:
        issue_count += text.str.findall(item).str.len().sum()
    return pd.DataFrame([[str_issue_title,issue_count]],columns=issues_columns)

issues_df = issues_df.append(count_issues(str_style_issues,'Style'))
issues_df = issues_df.append(count_issues(str_comfort_issues,'Comfort'))
issues_df = issues_df.append(count_issues(str_setup_issues,'Setup'))
issues_df = issues_df.append(count_issues(str_cost_issues,'Cost'))
issues_df = issues_df.append(count_issues(str_feature_issues,'Features'))
issues_df = issues_df.sort(columns = 'Count',ascending=False)

# bar chart to show biggest issues in lowest satisfactions
issues_df.plot(kind='bar',legend=None,rot=17)
plt.xticks(np.arange(5),(issues_df['Issue'].tolist()))
plt.title("Top 5 Dissatisfaction Issues From Twitter", fontweight='bold')
plt.xlabel("Issue", fontweight='bold')
plt.ylabel("Dissatisfied Count", fontweight='bold')
plt.show()

### DETERMINE TERM FREQUENCIES AND USING STOPWORDS
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
import string
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'us', 'via']
text = tweets['text']

tokens = []
for txt in text.values:
    tokens.extend([t.lower().strip(":,.") for t in txt.split()])

filtered_tokens = [w for w in tokens if not w in stop]
freq_dist = nltk.FreqDist(filtered_tokens)
freq_dist.keys()[:20]
freq_dist.plot(20, title='Term Frequencies')


### SENTIMENT ANALYSIS
# source code from:http://www.webstreaming.com.ar/articles/using-sentiment-analysis-and-python-to-evaluate-image-in-twitter-for-the-most-important-argentinian-candidates/
import tweepy
from tweepy import OAuthHandler
import urllib2
import json
from unidecode import unidecode

#set up keyword and limit of tweets collected
Keyword = "Fitbit"
LANGUAGE = 'es'
LIMIT = 2500 
URL_SENTIMENT140 = "http://www.sentiment140.com/api/bulkClassifyJson"


def parse_response(json_response):
    negative_tweets, positive_tweets = 0, 0
    for j in json_response["data"]:
        if int(j["polarity"]) == 0:
            negative_tweets += 1
        elif int(j["polarity"]) == 4:
            positive_tweets += 1
    return negative_tweets, positive_tweets

def main():
    auth = OAuthHandler(o.consumer_key, o.consumer_secret)
    auth.set_access_token(o.access_token_key, o.access_token_secret)
    api = tweepy.API(auth)
    tweets = []

    for tweet in tweepy.Cursor( api.search,
                                q=Keyword,
                                result_type='recent',
                                include_entities=True,
                                lang=LANGUAGE).items(LIMIT):
        aux = { "text" : unidecode(tweet.text.replace('"','')), "language": LANGUAGE, "query" : Keyword, "id" : tweet.id }
        tweets.append(aux)

    result = { "data" : tweets }

    req = urllib2.Request(URL_SENTIMENT140)
    req.add_header('Content-Type', 'application/json')
    response = urllib2.urlopen(req, str(result))
    json_response = json.loads(response.read())
    negative_tweets, positive_tweets = parse_response(json_response)

    print "Positive Tweets: " + str(positive_tweets)
    print "Negative Tweets: " + str(negative_tweets)

if __name__ == '__main__':
    main()
    

# Plot positive and negative tweets
import matplotlib.pyplot as plt  

labels = 'Pos', 'Neg'
sizes = [65,35]
colors = ['yellowgreen', 'gold']


plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)