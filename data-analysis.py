import numpy as np
import pandas as pd
import string
import re
import nltk
from regex import A
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_colwidth', 70)
data = pd.read_excel(r'udemy-courses.xlsx')

# Removing Punctuation


def remove_punctuation(txt):
    txt_nopunt = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunt


data['title_clean'] = data['Course Title'].apply(
    lambda x: remove_punctuation(x))
print(data.head())

# Tokenization


def tokenize(txt):
    txt_tokens = re.split('\W+', txt)
    return txt_tokens


data['title_tokens'] = data['title_clean'].apply(lambda x: tokenize(x.lower()))

# Stop words Removal
# nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
print(data.head())


def remove_stopwords(txt_tokenize):
    txt_without_stopwords = [
        word for word in txt_tokenize if word not in stopwords]
    return txt_without_stopwords


data['title_without_stopwords'] = data['title_tokens'].apply(
    lambda x: remove_stopwords(x))
print(data.head())

# Stemming

ps = nltk.PorterStemmer()


def stemming(txt_without_stopwords):
    txt_stem = [ps.stem(word) for word in txt_without_stopwords]
    return txt_stem


data['title_stem'] = data['title_without_stopwords'].apply(
    lambda x: stemming(x))
print(data.head())

# Lemmatization

# nltk.download('omw-1.4')
wn = nltk.WordNetLemmatizer()


def lemmatization(txt_without_stopwords):
    txt_lemmatization = [wn.lemmatize(word) for word in txt_without_stopwords]
    return txt_lemmatization


data['title_lemmatization'] = data['title_without_stopwords'].apply(
    lambda x: lemmatization(x))
print(data.head())

# Without Stopwords


def join_withoutStopwords(word_list):
    sentence = " ".join([c for c in word_list])
    return sentence


data['sentence_without_stopwords'] = data['title_without_stopwords'].apply(
    lambda x: join_withoutStopwords(x))

# TF-IDF (create Word Cloud)

vectorizer = TfidfVectorizer()

tdm = vectorizer.fit_transform(data['sentence_without_stopwords'])

print(vectorizer.vocabulary_.items())

tfidf_weights = [(word, tdm.getcol(idx).sum())
                 for word, idx in vectorizer.vocabulary_.items()]

w = WordCloud(width=1500, height=1200, mode='RGBA',
              background_color='white', max_words=2000).fit_words(dict(tfidf_weights))

plt.figure(figsize=(20, 15))
plt.imshow(w)
plt.axis('off')
plt.savefig('udemy_wordcloud.png')

# N-Grams


def N_Grams(text, n):
    tokens = re.split("\\s+", text)
    ngrams = []

    for i in range(len(tokens)-n+1):
        temp = [tokens[j] for j in range(i, i+n)]
        ngrams.append(" ".join(temp))
    return ngrams


data['title_ngrams'] = data['Course Title'].apply(
    lambda x: N_Grams(x, 2))
print(data.head())

# VSM based Similarity


def count_same_words(sentence, word):
    count = 0
    for i in sentence:
        if (i.lower() == word):
            count = count + 1
    return count


def similar_words(word_list1, word_list2):
    s_words = []
    for c in word_list1:
        s_words.append(count_same_words(word_list2, c))
    return s_words


query = "Python for Beginners Course"

remove_punc_query = remove_punctuation(query)

tokenize_query = tokenize(remove_punc_query)

query_sim = similar_words(tokenize_query, tokenize_query)

data['similar_with_query'] = data['title_tokens'].apply(
    lambda x: similar_words(tokenize_query, x))


def cos_sim(a, b):
    dot_product = np.dot(a, b, out=None)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if dot_product == 0:
        return 0
    else:
        return dot_product/(norm_a * norm_b)


def vsm(a, b):
    sentence1 = np.array(a)
    sentence2 = np.array(b)
    return cos_sim(sentence1, sentence2)


data['title_cosine'] = data['similar_with_query'].apply(
    lambda x: vsm(x, query_sim))

print(data.head())
