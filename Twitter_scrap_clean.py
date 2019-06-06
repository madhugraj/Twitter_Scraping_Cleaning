import tweepy
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string


####input your credentials here
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####Holidays and Trips


# Open/Create a file to append data
csvFile = open('...\Trip.csv', 'w')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#Holidays OR #holiday OR #Trips OR #Vaccation ",count=10000,
                           lang="en",
                           since="2015-05-01").items():
    print (tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    
 def load_data():
    data = pd.read_csv('....\Trip.csv')
    return data   

tweet_df = load_data()
tweet_df.head()

tweet_df.dropna()
tweet_df.head(10)

tweet_df.columns=['timestamp','text']
tweet_df.head(5)

print('Dataset size:',tweet_df.shape)
print('Columns are:',tweet_df.columns)

tweet_df.info()
df  = pd.DataFrame(tweet_df[['text']])
string.punctuation


def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df['Tweet_punct'] = df['text'].apply(lambda x: remove_punct(x))
df.head(10)

def tokenization(text):
    text = re.split('\W+', text)
    return text

df['Tweet_tokenized'] = df['Tweet_punct'].apply(lambda x: tokenization(x.lower()))
df.head()

stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
df['Tweet_nonstop'] = df['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))
df.head(10)

ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

df['Tweet_stemmed'] = df['Tweet_nonstop'].apply(lambda x: stemming(x))
df.head()

wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

df['Tweet_lemmatized'] = df['Tweet_nonstop'].apply(lambda x: lemmatizer(x))
df.head()

def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text
    
    
countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(df['text'])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))
print(countVectorizer.get_feature_names())

count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
count_vect_df.head()

df.to_csv(r'D:\Jupyter\Twitter Data\Trip_Clear.csv')
df
