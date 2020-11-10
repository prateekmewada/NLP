#!/usr/bin/env python
# coding: utf-8

# Importing the nltk library.

# In[221]:


import nltk
#nltk.download()


# Importing the list of stopwords from nltk.corpus

# In[223]:


from nltk.corpus import stopwords

stopwords.words("English")[:300:10]


# In[329]:


rawData = open('SMSSpamCollection.tsv').read()


# ## Preprocessing the data with basic approach.
# Replacing all the tabs with a new line and then splitting based on the new line.

# In[225]:


parsedData = rawData.replace('\t', '\n').split('\n')
parsedData[:5]


# In[226]:


labelData = parsedData[0::2]
textData = parsedData[1::2]
print(labelData[:5])
print(textData[:5])


# In[227]:


import pandas as pd

fullCorpus = pd.DataFrame({'label':labelData[:-1], 'body_list':textData})

fullCorpus.head()


# In[228]:


fullCorpus = pd.read_csv('SMSSpamCollection.tsv', sep = '\t', header = None)

fullCorpus.head()


# In[229]:


fullCorpus.columns = ['label', 'body_text']


# In[230]:


fullCorpus.head()


# In[231]:


print("The dataset has {} rows and {} columns".format(len(fullCorpus), len(fullCorpus.columns)))


# In[232]:


print("There are {} spam records and {} ham records.".format(len(fullCorpus[fullCorpus['label'] == 'spam']),
                                                                           len(fullCorpus[fullCorpus['label'] == 'ham'])))


# In[233]:


print("There are {} and {} null values for each columns".format(fullCorpus['label'].isnull().sum(), fullCorpus['body_text'].isnull().sum()))


# Trying regular expressions.

# In[234]:


import re

re_test = 'This is a made up string to test 2 different regex methods'
re_test_messy = 'This      is a made up     string to test 2    different regex methods'
re_test_messy1 = 'This-is-a-made/up.string*to>>>>test----2""""""different~regex-methods'


# In[235]:


re.split('\s', re_test)


# In[236]:


re.split('\s+', re_test_messy)


# In[237]:


re.split('\W+', re_test_messy1)


# In[238]:


re.findall('\w+', re_test)


# In[239]:


re.findall('\w+', re_test_messy)


# In[240]:


re.findall('\w+', re_test_messy1)


# In[241]:


pep8_test = 'I try to follow PEP8 guidelines'
pep7_test = 'I try to follow PEP7 guidelines'
peep8_test = 'I try to follow PEEP8 guidelines'


# In[242]:


re.findall('[A-Z]+[0-9]+', peep8_test)


# In[243]:


re.sub('[A-Z]+[0-9]+', 'PEP8 Python Style', pep7_test)


# In[244]:


data = fullCorpus.copy()

data.head()


# In[245]:


pd.set_option('display.max_colwidth', 100)


# In[246]:


data.head()


# ## Preprocessing with functions.

# In[247]:


import string

string.punctuation


# In[248]:


def remove_punct(text):
    text_nopunct = ''.join([char for char in text if char not in string.punctuation])
    return text_nopunct


# In[249]:


data['body_text_cleaned'] = data['body_text'].apply(lambda x: remove_punct(x))

data.head()


# In[250]:


def tokenizer(text):
    tokens = re.split('\W+', text.lower())
    return tokens

data['body_text_tokenized'] = data['body_text_cleaned'].apply(lambda x: tokenizer(x))

data.head()


# In[251]:


stopword = nltk.corpus.stopwords.words('english')


# In[252]:


def remove_stop(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x: remove_stop(x))

data.head()


# ## Simple word trimmer

# In[253]:


ps = nltk.PorterStemmer()


# In[254]:


print(ps.stem('grow'))
print(ps.stem('growing'))
print(ps.stem('grew'))
print(ps.stem('grown'))


# In[255]:


def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

data['body_text_stemmed'] = data['body_text_nostop'].apply(lambda x: stemming(x))

data.head()


# ## A better word trimmer.

# In[256]:


wn = nltk.WordNetLemmatizer()


# In[257]:


print(wn.lemmatize('foot'))
print(wn.lemmatize('feet'))
print(wn.lemmatize('grown'))


# In[258]:


def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

data['body_text_lemmatized'] = data['body_text_nostop'].apply(lambda x: lemmatizing(x))
data.head()


# ## Cleaning the Data by combining splitting and trimming the words together in a function.

# In[259]:


def clean_text(text):
    text = ''.join([char.lower() for char in text if char not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [wn.lemmatize(word) for word in tokens if word not in stopword]
    return text


# ## Trying Count Vectorizer

# In[260]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer = clean_text)
X_counts = count_vect.fit_transform(data['body_text'])
print(X_counts.shape)
print(count_vect.get_feature_names())


# In[261]:


data_sample = data[0:20]

count_vect_sample = CountVectorizer(analyzer = clean_text)
X_counts_sample = count_vect_sample.fit_transform(data_sample['body_text'])
print(X_counts_sample.shape)
print(count_vect_sample.get_feature_names())


# In[262]:


X_counts_sample


# In[263]:


X_counts_df = pd.DataFrame(X_counts_sample.toarray())


# In[264]:


X_counts_df.columns = count_vect_sample.get_feature_names()

X_counts_df.head()


# In[265]:


def clean_text1(text):
    text = ''.join([char.lower() for char in text if char not in string.punctuation])
    tokens = re.split('\W+', text)
    text = ' '.join([wn.lemmatize(word) for word in tokens if word not in stopword])
    return text

data['cleaned_text1'] = data['body_text'].apply(lambda x: clean_text1(x))


# ## Trying n-gram vectorizer

# In[266]:


ngram_vect = CountVectorizer(ngram_range=(2,2))
X_ng_counts = ngram_vect.fit_transform(data['cleaned_text1'])
print(X_ng_counts.shape)
print(ngram_vect.get_feature_names())


# In[267]:


data_ng_sample = data[0:20]

ngram_vect_sample = CountVectorizer(ngram_range=(2,2))
X_ng_counts_sample = ngram_vect_sample.fit_transform(data_ng_sample['cleaned_text1'])
print(X_ng_counts_sample.shape)
print(ngram_vect_sample.get_feature_names())


# In[268]:


X_ng_sample_df = pd.DataFrame(X_ng_counts_sample.toarray())

X_ng_sample_df


# In[269]:


X_ng_sample_df.columns = ngram_vect_sample.get_feature_names()
X_ng_sample_df


# ## Trying Tfidf Vectorizer

# In[270]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[271]:


tfidf_vect = TfidfVectorizer(analyzer= clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names())


# In[272]:


data_tfidf_sample = data[0:20]


# In[273]:


tfidf_vect_sample = TfidfVectorizer(analyzer= clean_text)
X_tfidf_sample = tfidf_vect_sample.fit_transform(data_tfidf_sample['body_text'])
print(X_tfidf_sample.shape)
print(tfidf_vect_sample.get_feature_names())


# In[274]:


X_tfidf_sampledf = pd.DataFrame(X_tfidf_sample.toarray())


# In[275]:


X_tfidf_sampledf.columns = tfidf_vect_sample.get_feature_names()

X_tfidf_sampledf


# ## To check the length of the message body.

# In[276]:


data['body_length'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))


# In[277]:


data.head()


# ## Defining a function to check if the message body contains more punctuations.

# In[278]:


import string

def percent(text):
    count = sum([1 for word in text if word in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['punct_percent'] = data['body_text'].apply(lambda x: percent(x))

data.head()


# ## Plotting the graphs on the basis of message length to check spam messages

# In[279]:


from matplotlib import pyplot
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[280]:


bins1 = np.linspace(0,200,40)

pyplot.hist(data[data['label'] == 'spam']['body_length'], bins1, alpha = 0.5, density = True, label = 'spam')
pyplot.hist(data[data['label'] == 'ham']['body_length'], bins1, alpha = 0.5, density = True, label = 'ham')
pyplot.legend(loc = 'upper left')
pyplot.show()


# ## Plotting the graphs on the basis of punctuations to check spam messages

# In[281]:


bins2 = np.linspace(0,50,40)

pyplot.hist(data[data['label'] == 'spam']['punct_percent'], bins2, alpha = 0.5, density = True, label = 'spam')
pyplot.hist(data[data['label'] == 'ham']['punct_percent'], bins2, alpha = 0.5, density= True, label = 'ham')
pyplot.legend(loc = 'upper right')
pyplot.show()


# In[282]:


pyplot.hist(data['body_length'], bins1)
pyplot.title("Body length Distribution")
pyplot.show()


# In[283]:


pyplot.hist(data['punct_percent'], bins2)
pyplot.title("Punctuation percent Distribution")
pyplot.show()


# In[285]:


X_features = pd.concat([data['body_length'], data['punct_percent'], pd.DataFrame(X_tfidf.toarray())], axis=1)
X_features.head()


# In[286]:


from sklearn.ensemble import RandomForestClassifier

print(dir(RandomForestClassifier))


# In[290]:


print(RandomForestClassifier)


# In[291]:


from sklearn.model_selection import KFold, cross_val_score


# In[293]:


rf = RandomForestClassifier(n_jobs=1)
k_fold = KFold(n_splits=5)
cross_val_score(rf, X_features, data['label'], cv = k_fold, scoring = 'accuracy', n_jobs = 1)


# In[299]:


from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


# In[295]:


X_train, X_test, Y_train, Y_test = train_test_split(X_features, data['label'], test_size=0.2)


# In[296]:


rf1 = RandomForestClassifier(n_estimators=50, max_depth= 20, n_jobs= -1)
rf_model = rf1.fit(X_train, Y_train)


# In[298]:


sorted(zip(rf_model.feature_importances_, X_train.columns), reverse = True)[0:10]


# In[301]:


Y_pred = rf_model.predict(X_test)
precision, recall, fscore, support = score(Y_test, Y_pred, pos_label = 'spam', average = 'binary')


# In[305]:


print("Precision: {} / Recall: {} / Accuracy: {}".format(round(precision, 3),
                                                         round(recall,3), 
                                                         round((Y_test==Y_pred).sum()/len(Y_pred),3)))


# In[310]:


def train_RF(n_est, depth):
    rf = RandomForestClassifier(n_estimators = n_est, max_depth = depth, n_jobs = -1)
    rf_model = rf.fit(X_train, Y_train)
    Y_pred = rf_model.predict(X_test)
    precision, recall, fscore, support = score(Y_test, Y_pred, pos_label = 'spam', average = 'binary')
    print('Est: {} / Depth: {} / Precision: {} / Recall: {} / Accuracy: {}'.format(n_est, depth, round(precision, 3), round(recall,3),
                                                                                  round((Y_pred==Y_test).sum()/len(Y_pred), 3)))


# In[311]:


for n_est in [10,50,100]:
    for depth in [10,20,30, None]:
        train_RF(n_est, depth)


# In[312]:


# TF-IDF
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
X_tfidf_feat = pd.concat([data['body_length'], data['punct_percent'], pd.DataFrame(X_tfidf.toarray())], axis=1)

# CountVectorizer
count_vect = CountVectorizer(analyzer=clean_text)
X_count = count_vect.fit_transform(data['body_text'])
X_count_feat = pd.concat([data['body_length'], data['punct_percent'], pd.DataFrame(X_count.toarray())], axis=1)


# In[313]:


from sklearn.model_selection import GridSearchCV


# In[316]:


param = {'n_estimators' : [10, 150, 300],
        'max_depth' : [30, 60, 90, None]}
rf = RandomForestClassifier()

gs = GridSearchCV(rf, param, cv = 5, n_jobs = -1)
gs_fit = gs.fit(X_tfidf_feat, data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]


# In[318]:


gs = GridSearchCV(rf, param, cv = 5, n_jobs = -1)
gs_fit = gs.fit(X_count_feat, data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]


# In[319]:


from sklearn.ensemble import GradientBoostingClassifier


# In[323]:


def train_GF(n_est, depth, lr):
    gb = GradientBoostingClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr)
    gb_model = gb.fit(X_train, Y_train)
    Y_pred = gb_model.predict(X_test)
    prediction, recall, fscore, support = score(Y_test, Y_pred, pos_label = 'spam', average = 'binary')
    print('Est: {} / Depth: {} / Learning Rate: {} / --- Prediction: {} / Recall: {} / Accuracy: {}'.format(
            n_est, depth, lr, round(prediction, 3), round(recall, 3), round((Y_pred==Y_test).sum()/ len(Y_pred), 3)))


# In[324]:


for n_est in [50, 100, 150]:
    for depth in [3, 7, 11]:
        for lr in [0.01, 0.1]:
            train_GF(n_est, depth, lr)


# In[ ]:




