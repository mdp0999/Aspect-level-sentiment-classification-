#%matplotlib inline #only in ipython console
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas as pd
import sklearn
import _pickle as cPickle
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve
from itertools import takewhile



text = open("C:/Python36/Aspect_Classification/spect_annoated_file.txt", "r", encoding='utf8').read()
reviews = pd.read_csv('C:/Python36/Aspect_Classification/aspect_words_1k_restaurant_reviews.txt', names = ['words'],encoding='utf8')

sentences = []
split = iter(text.splitlines())
while True:
    sentence = list(takewhile(bool, split))
    if not sentence:
        break
    types = set(el.split()[1] for el in sentence)
    words = [el.split(' ', 1)[0] for el in sentence]
    
    
    sentences.append(sentence)

a_train = sentences[:750]
a_test = sentences[750:1000]


b = []
train = []
for a in a_train:
    for b in a:
        c = b.split('\t')
        train.append(c)
        

i = []
test = []
for i in a_test:
    for w in i:
        w = w.split('\t')
        test.append(w)
    


train_data = pd.DataFrame(train, columns = ['words','Aspects'])
test_data = pd.DataFrame(test, columns = ['words','Aspects'])

print(train_data.head())
print(test_data.head())

train_words = train_data['words']
train_Aspects = train_data['Aspects']

test_words = test_data['words']
test_Aspects = test_data['Aspects']

CountVectorizer = CountVectorizer(analyzer= u'word')

pipeline = Pipeline([
    ('bow', CountVectorizer),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
    
scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         train_words,  # training data
                         train_Aspects,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )

print (scores)
print(scores.mean(), scores.std())

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    return plt


%time plot_learning_curve(pipeline, "accuracy vs. training set size", train_words, train_Aspects, cv=5)



params = {
    'tfidf__use_idf': (True, False),
    #'bow__analyzer': (u'word'),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(train_Aspects, n_folds=5),  # what type of cross validation to use
)

%time nb_detector = grid.fit(train_words, train_Aspects)
print(nb_detector.grid_scores_)

print( nb_detector.predict_proba(["food"]))
print( nb_detector.predict_proba(["place"]))

print( nb_detector.predict(["food"]))
print( nb_detector.predict(["place"]))

predictions = nb_detector.predict(test_words)
print(confusion_matrix(test_Aspects, predictions))
print(classification_report(test_Aspects, predictions))

# store the spam detector to disk after training
with open('aspect_detector.pkl', 'wb') as fout:
    cPickle.dump(nb_detector, fout)

# ...and load it back, whenever needed, possibly on a different machine
nb_detector_reloaded = cPickle.load(open('aspect_detector.pkl'))

