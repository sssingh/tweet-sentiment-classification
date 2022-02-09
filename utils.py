import re
from collections import Counter
from itertools import chain
import numpy as np
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

# We need to save certain pieces of data used in training
# for latter when model will be used for prediction.
# Ideally these needs to be serialized after training
meta_data = {}

def init():
    '''
    Initialise and load required data
    '''
    # Download NLTK tokenizer model and stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    # Get list of stop words and punctuations and store in meta data as well
    meta_data['stop_words'] = stopwords.words('english')
    meta_data['punctuations'] = list(punctuation)


def clean_and_tokenize_tweet(tweet):
    '''
    Returns a clean tokenized representation of a tweet
    1) Remove twitter handles, hash-tags and hyper-links
    2) Convert every tweet to lowercase
    3) Remove punctuation
    4) Remove stop words
    5) Apply stemming
    '''         
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks    
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    # Remove stop-words and punctuation
    stop_words = meta_data['stop_words']
    punctuations = meta_data['punctuations']
    tweet_tokens = [token for token in tweet_tokens 
                    if token not in stop_words and
                       token not in punctuations]
    # Apply stemming
    stemmer = PorterStemmer()
    tweet_tokens = [stemmer.stem(token) for token in tweet_tokens] 
    
    return tweet_tokens


def build_freq_dict(tweets): 
    '''
    Builds a dictionary of unique words and number of its occurances
    in the list of tweets provided
    '''
    # Flatten the list of tweet token lists
    flat_tweets = list(chain(*tweets))
    # Get the frequency of tokens
    freq = Counter(flat_tweets)
    
    return freq


def extract_features(tweet, pos_freq_dict, neg_freq_dict):
    '''
    Encodes a single list of tweet tokens into three features
    (bias (1), sum-of-positive-freq, sum-of-negative-freq)
    '''
    pos_freq = 0
    neg_freq = 0
    for token in tweet:
        pos_freq += pos_freq_dict.get(token, 0)
        neg_freq += neg_freq_dict.get(token, 0)
        
    return [1, pos_freq, neg_freq]


def build_feature_matrix(tweets, label, pos_freq_dict, neg_freq_dict):
    '''
    Prepares the feature matrix and corresponding label vector from the list
    of tweets
    '''
    n_tweets = len(tweets)
    X = np.zeros((n_tweets, 3))
    y = np.zeros((n_tweets, 1))
    for i, tweet in enumerate(tweets):
        X[i, :] = extract_features(tweet, pos_freq_dict, neg_freq_dict)
        
        y[i] = label
    return X, y


def get_sentiment(tweet, model, encode_tweet=True):
    '''
    Computes the probability of passed raw tweet's sentiment
    being positive or negative.
    '''
    # First get meta-data used in training
    pos_freq_dict = meta_data['pos_freq_dict']
    neg_freq_dict = meta_data['neg_freq_dict']
    
    # Clean and encode tweet
    tweet = clean_and_tokenize_tweet(tweet)
    X = None
    if encode_tweet:
        tweet = extract_features(tweet, pos_freq_dict, neg_freq_dict)
        X = np.zeros((1, 3))
        X[0, :] = tweet
    else:
        X = np.expand_dims(np.array(tweet), axis=0)
    # Use passed model to get prediction
    label = model.predict(X)
    sentiment = 'POSITIVE' if label.squeeze() == 1.0 else 'NEGATIVE'
    
    return sentiment
