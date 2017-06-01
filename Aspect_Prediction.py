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


train_data.groupby('Aspects').describe()#train data:With pandas, we can also view aggregate statistics easily
train_data['length'] = train_data['words'].map(lambda text: len(text))
train_data.length.plot(bins=20, kind='hist')

train_data.length.describe()
#difference in message length between asp and nasp
train_data.hist(column='length', by='Aspects', bins=50)

#Data preprocessing: already words are splited and processed so don't need this step

#Data to vectors
count_vectors = CountVectorizer(analyzer= u'word').fit(train_words)

print(len(count_vectors.vocabulary_))

#checking the words with position numbers
print( count_vectors.get_feature_names()[902]) #made
print( count_vectors.get_feature_names()[1001])#needs

#The bag-of-words counts for the entire SMS corpus are a large, sparse matrix:
train_cv = count_vectors.transform(train_words)
test_cv = count_vectors.transform(test_words)
print( 'sparse matrix shape:', train_cv.shape )
print( 'number of non-zeros:', train_cv.nnz )
print( 'sparsity: %.2f%%' % (100.0 * train_cv.nnz / (train_cv.shape[0] * train_cv.shape[1])))

#And finally, after the counting, the term weighting and normalization can be done with TF-IDF, using scikit-learn's TfidfTransformer:
tfidf_transformer = TfidfTransformer().fit(train_cv)
tfidf_transformer1 = tfidf_transformer.transform(test_cv)
print(tfidf_transformer1)
#the IDF (inverse document frequency) of the word "tip"
print(tfidf_transformer.idf_[count_vectors.vocabulary_['tip']])

#To transform the entire bag-of-words corpus into TF-IDF corpus at once:
train_tfidf = tfidf_transformer.transform(train_cv)
test_tfidf1 = tfidf_transformer.transform(test_cv)
print(train_tfidf.shape)
    
##Training a model, detecting aspect
#%time aspect_detector = MultinomialNB().fit(train_tfidf, train_Aspects) #only ipython console
aspect_detector = MultinomialNB().fit(train_tfidf, train_Aspects)  

aspect_detector1 = BernoulliNB().fit(train_tfidf, train_Aspects)  
 
#train data predictions
all_predictions2 = aspect_detector.predict(train_tfidf)
print( all_predictions2 )  
    
all_predictions3 = aspect_detector1.predict(train_tfidf)
print( all_predictions3 )   

#test data predictions
all_predictions = aspect_detector.predict(test_tfidf1)
print( all_predictions )  
    
all_predictions1 = aspect_detector1.predict(test_tfidf1)
print( all_predictions1 )

#MultinomialNB accuracy
print( 'accuracy', accuracy_score(train_Aspects, all_predictions2))
print( 'confusion matrix\n', confusion_matrix(train_Aspects, all_predictions2))
print( '(row=expected, col=predicted)')
#BernoulliNB accuracy
print( 'accuracy', accuracy_score(train_Aspects, all_predictions3))
print( 'confusion matrix\n', confusion_matrix(train_Aspects, all_predictions3))
print( '(row=expected, col=predicted)')

#here I am choosing Multinomial Naive Bayes because MultinomialNB(98%) is greater than BernoulliNB(95%) accuracy.

plt.matshow(confusion_matrix(train_Aspects, all_predictions2), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected Aspect')
plt.xlabel('predicted Aspect')                                                             

print(classification_report(train_Aspects, all_predictions2))
#for test data
print( 'accuracy', accuracy_score(test_Aspects, all_predictions))
print( 'confusion matrix\n', confusion_matrix(test_Aspects, all_predictions))
print( '(row=expected, col=predicted)')

print(classification_report(test_Aspects, all_predictions))

######################
reviews_cv = count_vectors.transform(reviews['words'])
reviews_tfidf = tfidf_transformer.transform(reviews_cv)
##predictions
all_predictions4 = aspect_detector.predict(reviews_tfidf)
print( all_predictions4 )
####save results in text file
res = pd.DataFrame(all_predictions4,columns = ['Aspects'] )
out = pd.concat([reviews['words'],res], axis = 1)
out = np.array(out)
np.savetxt('C:/Python36/Aspect_Classification/output.txt', out, delimiter='\t', fmt = '%s%s')

###################################



