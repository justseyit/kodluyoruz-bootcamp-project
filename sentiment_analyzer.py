from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":

    positive_tweets = twitter_samples.strings('positive_tweets.json')
    #print(positive_tweets[:20])
    for i in positive_tweets[:20]:
        print(i)
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')
    #print("stop words",stop_words)

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    #print("positive tokens",positive_tweet_tokens[:2])
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
    #print("negative tokens",negative_tweet_tokens[:2])

    all_pos_words = get_all_words(positive_tweet_tokens)

    #print("all_pos_words",all_pos_words)
    freq_dist_pos = FreqDist(all_pos_words)
    #print(freq_dist_pos.most_common(10),"burada")

    positive_tokens_for_model = get_tweets_for_model(positive_tweet_tokens)
    negative_tokens_for_model = get_tweets_for_model(negative_tweet_tokens)
    #print("positive tokens model",positive_tokens_for_model[1])
    #print("negative tokens model",negative_tokens_for_model[:2])
    
    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]
    #print(positive_dataset[0:2],"burada 2")

    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]
    #print(negative_dataset[0:2],"burada negative")

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    #print("train data",train_data[:2])
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    #print(classifier.show_most_informative_features(10))

    custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    #print(custom_tokens,dict([token, True] for token in custom_tokens))

    print(custom_tweet,"---", classifier.classify(dict([token, True] for token in custom_tokens)))

    custom_tweet = 'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies'
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    print(custom_tweet,"---", classifier.classify(dict([token, True] for token in custom_tokens)))
