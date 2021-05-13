from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, preprocessing
from tqdm import tqdm
import multiprocessing
import pandas as pd
import numpy as np
import nltk
import re
import gensim
import warnings
warnings.filterwarnings('ignore')

'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):

    # clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    # Tokenize (convert from string to list)
    lst_text = nltk.word_tokenize(text)
    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    # Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    # back to string from list
    text = " ".join(lst_text)

    return text


train_data = pd.read_csv('data/train_full.csv', error_bad_lines=False)
train_tweetContent = train_data[['tweet']]
train_labels = train_data['region']

dev_data = pd.read_csv('data/dev_full.csv', error_bad_lines=False)
dev_tweetContent = dev_data[['tweet']]
dev_labels = dev_data['region']

test_data = pd.read_csv('data/test_full.csv', error_bad_lines=False)
test_tweetContent = test_data[['tweet']]

assert(len(train_tweetContent) == len(train_labels))
assert(len(dev_tweetContent) == len(dev_labels))

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
lst_stopwords = nltk.corpus.stopwords.words("english")

train_tweetContent = train_tweetContent["tweet"].apply(lambda x:
                                                       utils_preprocess_text(x, flg_stemm=False, flg_lemm=False,
                                                                             lst_stopwords=lst_stopwords))

dev_tweetContent = dev_tweetContent["tweet"].apply(lambda x:
                                                   utils_preprocess_text(x, flg_stemm=False, flg_lemm=False,
                                                                         lst_stopwords=lst_stopwords))

test_tweetContent = test_tweetContent["tweet"].apply(lambda x:
                                                     utils_preprocess_text(x, flg_stemm=False, flg_lemm=False,
                                                                           lst_stopwords=lst_stopwords))

# Train Word2Vec model using all the tweets we have
# Tokenize all the tweets by splitting
train_tokens = []
for tweet in train_tweetContent:
    train_tokens.append(tweet.split())

dev_tokens = []
for tweet in dev_tweetContent:
    dev_tokens.append(tweet.split())

test_tokens = []
for tweet in test_tweetContent:
    test_tokens.append(tweet.split())

all_tokens = np.concatenate((train_tokens, dev_tokens, test_tokens))

cores = multiprocessing.cpu_count()

w2v_model = gensim.models.word2vec.Word2Vec(
    all_tokens, vector_size=300, window=8, min_count=1, sg=1, epochs=30, workers=cores)
w2v_words = w2v_model.wv.index_to_key

classifiers = [
    KNeighborsClassifier(n_neighbors=49),
    LogisticRegression(n_jobs=-1),
    LinearSVC(C=0.12, random_state=42),
    MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto',
                  beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-06,
                  hidden_layer_sizes=(100,), learning_rate='adaptive',
                  learning_rate_init=0.001, max_iter=10000, momentum=0.9,
                  n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                  random_state=42, shuffle=True, solver='adam', tol=0.0001,
                  validation_fraction=0.1, verbose=False, warm_start=False),
    DecisionTreeClassifier(random_state=0),
    ExtraTreesClassifier(n_jobs=-1),
    GaussianNB()
]

# average Word2Vec
print("average word2vec")
# compute average word2vec for each tweet
train_vectors = []  # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(train_tokens):  # for each tweet
    sent_vec = np.zeros(300)  # as word vectors are of zero length 300,
    cnt_words = 0  # num of words with a valid vector in the sentence/review
    for word in sent:  # for each word in a tweet
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    train_vectors.append(sent_vec)

dev_vectors = []  # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(dev_tokens):  # for each tweet
    sent_vec = np.zeros(300)  # as word vectors are of zero length 300,
    cnt_words = 0  # num of words with a valid vector in the sentence/review
    for word in sent:  # for each word in a tweet
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    dev_vectors.append(sent_vec)

test_vectors = []  # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(test_tokens):  # for each tweet
    sent_vec = np.zeros(300)  # as word vectors are of zero length 300,
    cnt_words = 0  # num of words with a valid vector in the sentence/review
    for word in sent:  # for each word in a tweet
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    test_vectors.append(sent_vec)

for classifier in classifiers:
    clf = classifier.fit(train_vectors, train_labels)
    predicted_labels = clf.predict(dev_vectors)
    f1 = metrics.f1_score(
        dev_labels, predicted_labels, average='micro')
    print(f'{type(clf).__name__} F1： {f1:.4f}')

print("l1 normalized sum word2vec")
# compute l1-norm-sum word2vec for each tweet
train_vectors = []  # the sum-w2v for each sentence/review is stored in this list
for sent in tqdm(train_tokens):  # for each tweet
    sent_vec = np.zeros(300)  # as word vectors are of zero length 300,
    for word in sent:  # for each word in a tweet
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
    train_vectors.append(sent_vec)
# L1 normalize
train_vectors = preprocessing.normalize(train_vectors, norm='l1')

dev_vectors = []  # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(dev_tokens):  # for each tweet
    sent_vec = np.zeros(300)  # as word vectors are of zero length 300,
    cnt_words = 0  # num of words with a valid vector in the sentence/review
    for word in sent:  # for each word in a tweet
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    dev_vectors.append(sent_vec)
dev_vectors = preprocessing.normalize(dev_vectors, norm='l1')

test_vectors = []  # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(test_tokens):  # for each tweet
    sent_vec = np.zeros(300)  # as word vectors are of zero length 300,
    cnt_words = 0  # num of words with a valid vector in the sentence/review
    for word in sent:  # for each word in a tweet
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    test_vectors.append(sent_vec)
dev_vectors = preprocessing.normalize(dev_vectors, norm='l1')

for classifier in classifiers:
    clf = classifier.fit(train_vectors, train_labels)
    predicted_labels = clf.predict(dev_vectors)
    f1 = metrics.f1_score(
        dev_labels, predicted_labels, average='micro')
    print(f'{type(clf).__name__} F1： {f1:.4f}')


print("l2 normalized sum word2vec")
# compute l1-norm-sum word2vec for each tweet
train_vectors = []  # the sum-w2v for each sentence/review is stored in this list
for sent in tqdm(train_tokens):  # for each tweet
    sent_vec = np.zeros(300)  # as word vectors are of zero length 300,
    for word in sent:  # for each word in a tweet
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
    train_vectors.append(sent_vec)
# L1 normalize
train_vectors = preprocessing.normalize(train_vectors, norm='l2')

dev_vectors = []  # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(dev_tokens):  # for each tweet
    sent_vec = np.zeros(300)  # as word vectors are of zero length 300,
    cnt_words = 0  # num of words with a valid vector in the sentence/review
    for word in sent:  # for each word in a tweet
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    dev_vectors.append(sent_vec)
dev_vectors = preprocessing.normalize(dev_vectors, norm='l2')

test_vectors = []  # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(test_tokens):  # for each tweet
    sent_vec = np.zeros(300)  # as word vectors are of zero length 300,
    cnt_words = 0  # num of words with a valid vector in the sentence/review
    for word in sent:  # for each word in a tweet
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    test_vectors.append(sent_vec)
dev_vectors = preprocessing.normalize(dev_vectors, norm='l2')

for classifier in classifiers:
    clf = classifier.fit(train_vectors, train_labels)
    predicted_labels = clf.predict(dev_vectors)
    f1 = metrics.f1_score(
        dev_labels, predicted_labels, average='micro')
    print(f'{type(clf).__name__} F1： {f1:.4f}')
