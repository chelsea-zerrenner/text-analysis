# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 23:21:25 2018

@author: chzerr
"""

# Chelsea Zerrenner
# Publicis Data Scientist Evaluation


# The following script was created according to the explicit instructions 
# provided.  Note, however that it came to my attention that certain steps 
# were not taken or were done out of the sequence as outlined by data science
# "best practices."  As such, I've provided additional attempts taking this
# into account.

###############################################################################
# Attempt 1: Follows instructions explicitly                                  #
###############################################################################
import os
os.chdir("C:\\Users\\chzerr\\Documents\\Programming\\Python Scripts")

# Create dataset from Stack Exchange Posts
from pymongo import MongoClient
import pandas as pd
import numpy as np
import nltk
import io


# Connect to mongo client
client = MongoClient()
db = client.stack


# Convert mongo collections to pandas dataframe 
astronomy = pd.DataFrame(list(db.astronomy.find()))
aviation = pd.DataFrame(list(db.aviation.find()))
beer = pd.DataFrame(list(db.beer.find()))
outdoors = pd.DataFrame(list(db.outdoors.find()))
pets = pd.DataFrame(list(db.pets.find()))


# Close connection before moving on
client.close()


# Combine collections to create one dataset of all documents named "posts"
posts = pd.concat([astronomy, aviation, beer, outdoors, pets])


# Create function to apply labels to documents
def f(x):
    if x['id'].startswith('astronomy'): return 'astronomy'
    elif x['id'].startswith('aviation'): return 'aviation'
    elif x['id'].startswith('beer'): return 'beer'
    elif x['id'].startswith('outdoors'): return 'outdoors'
    else: return 'pets'

posts['label'] = posts.apply(f, axis = 1)


# Reduce dataset to include only the fields: title, body & label
X = posts['title'] + posts['body']
y = posts['label']

# Test to make sure dataset was reduced correctly
X.head()



# Data Preprocessing
# Remove punctuation
import string
def remove_punc(text):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in text if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    return nopunc
       
X = X.apply(remove_punc)


# Find 1000 most common words in 'X' dataset
nltk.download('punkt')
nltk.download('wordnet')
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer

# Define lemma tokenizer from nltk package to be used in sklearn vectorizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

# Apply vectorizer to obtain word counts after stemming & produce term document
# matrix
vect = CountVectorizer(tokenizer = LemmaTokenizer(), 
                       stop_words = 'english',  
                       lowercase = True).fit(X)
tdm = vect.transform(X)

# Obtain 1000 most common words & output list
sum_words = tdm.sum(axis = 0)
words_freq = [(word, sum_words[0, idx]) for word, 
              idx in vect.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], 
                    reverse = True)
common_words = words_freq[:1000]
common_words
np.savetxt('.\\1000_most_common.csv', 
           common_words, 
           delimiter = ',', 
           fmt = '%s')

# Term document matrix
tdm.toarray()


# Perform Dimension Reduction
from sklearn.decomposition import TruncatedSVD

# Evaluate extracting all possible components
svd = TruncatedSVD(n_components = tdm.shape[1] - 1, 
                   random_state = 1234)
tsvd = svd.fit_transform(tdm)
svd_var_ratios = svd.explained_variance_ratio_

# Define function to select number of components based on variability retention
def select_n_components(var_ratio, goal_var):
    # Set initial variance explained so far
    total_variance = 0.0
    
    # Set initial number of features
    n_components = 0
    
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        
        # Add the explained variance to the total
        total_variance += explained_variance
        
        # Add one to the number of components
        n_components += 1
        
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
            
    # Return the number of components
    return n_components

select_n_components(svd_var_ratios, 0.95)

# Final selection of components based on 95% variability
svd_final = TruncatedSVD(n_components = select_n_components(svd_var_ratios, 0.95), 
                         random_state = 1234)
tdm_final = svd_final.fit_transform(tdm)



# Build Classifier
from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split


# Set aside 500 random discussions from entire dataset to serve as hold-out
X_train, X_test, y_train, y_test = train_test_split(tdm_final, 
                                                    y, 
                                                    test_size = 500, 
                                                    random_state = 0) 


# Define model & performance measure
mdl = RandomForestClassifier(n_estimators = 20)
accuracy = make_scorer(accuracy_score)


# Random search for parameters
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 100),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

rand_search = RandomizedSearchCV(mdl, 
                                 param_distributions = param_dist, 
                                 n_iter = 20, 
                                 random_state = 1234, 
                                 scoring = accuracy, 
                                 refit = True, 
                                 cv = 10)
mdl_1 = rand_search.fit(X_train, 
                        y_train)
att1_cv_accuracy = mdl_1.best_score_
att1_cv_accuracy


# Return best estimator & parameters
print(mdl_1.best_estimator_)
print(mdl_1.best_params_)


# Evaluate on hold-out test sample
att1_y_pred = mdl_1.predict(X_test)
att1_test_accuracy = accuracy_score(y_test, 
                                    att1_y_pred)
att1_test_accuracy


# Classification Report
from sklearn.metrics import classification_report, confusion_matrix

target_names = ['astronomy', 'aviation', 'beer', 'outdoors', 'pets']
att1_class_report = classification_report(y_test, 
                                          att1_y_pred, 
                                          target_names = target_names)
print(att1_class_report)

# Save report
with io.open('./att1_class_report.txt', 'w', encoding = 'utf-8') as class_report1: 
    class_report1.write(att1_class_report)


# Confusion Matrix
att1_conf_matrix = confusion_matrix(y_test, 
                                    att1_y_pred, 
                                    labels = target_names)
print(att1_conf_matrix)

# Save matrix
np.savetxt('.\\att1_conf_matrix.csv', 
           att1_conf_matrix, 
           delimiter = ',', 
           fmt = '%s')


# Store evaluation metrics
from sklearn.metrics import precision_recall_fscore_support

att1_train_accuracy = accuracy_score(y_train, 
                                     mdl_1.predict(X_train))
att1_L = list(precision_recall_fscore_support(y_test, 
                                              att1_y_pred, 
                                              average = 'weighted'))
metrics = pd.DataFrame([[1, 
                         att1_train_accuracy, 
                         att1_cv_accuracy, 
                         att1_test_accuracy] + att1_L[:3]], 
                            columns = ['attempt', 
                                       'train accuracy', 
                                       'mean cv accuracy', 
                                       'test accuracy', 
                                       'test precision', 
                                       'test recall', 
                                       'test f1-score'])




###############################################################################
# Attempt 2: Fit Classifier on Term Freq Inverse Document Freq Matrix         #
###############################################################################
# Perform traning on tf-idf matrix in place of term document matrix citing only
# term frequency in attempt to improve classifier's performance

# Data Preprocessing
# Convert tdm to tf-idf matrix
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(tdm)
tfidf = tfidf_transformer.transform(tdm)
tfidf


# Perform Dimension Reduction
tfidf_svd = TruncatedSVD(n_components = tfidf.shape[1] - 1, 
                         random_state = 1234)
tfidf_tsvd = tfidf_svd.fit_transform(tfidf)
tfidf_svd_var_ratios = tfidf_svd.explained_variance_ratio_

select_n_components(tfidf_svd_var_ratios, 0.95)

tfidf_svd_final = TruncatedSVD(n_components = select_n_components(tfidf_svd_var_ratios, 
                                                                  0.95), 
                               random_state = 1234)
tfidf_final = tfidf_svd_final.fit_transform(tfidf)



# Build Classifier 
# Split tf-idf into train & test sets (500 random documents for test)
X_train, X_test, y_train, y_test = train_test_split(tfidf_final, 
                                                    y, 
                                                    test_size = 500, 
                                                    random_state = 0) 


# Fit classifier using same random search parameters & modeling algorithm
# See mdl, accuracy, param_dist & rand_search above
mdl_2 = rand_search.fit(X_train, 
                        y_train)
att2_cv_accuracy = mdl_2.best_score_
att2_cv_accuracy


# Return best estimator & parameters
print(mdl_2.best_estimator_)
print(mdl_2.best_params_)


# Evaluate on hold-out test sample
att2_y_pred = mdl_2.predict(X_test)
att2_test_accuracy = accuracy_score(y_test, 
                                    att2_y_pred)
att2_test_accuracy


# Classification Report
target_names = ['astronomy', 'aviation', 'beer', 'outdoors', 'pets']
att2_class_report = classification_report(y_test, 
                                          att2_y_pred, 
                                          target_names = target_names)
print(att2_class_report)

# Save report
with io.open('./att2_class_report.txt', 'w', encoding = 'utf-8') as class_report2: 
    class_report2.write(att2_class_report)
        
        
# Confusion Matrix
att2_conf_matrix = confusion_matrix(y_test, 
                                    att2_y_pred, 
                                    labels = target_names)
print(att2_conf_matrix)

# Save matrix
np.savetxt('.\\att2_conf_matrix.csv', 
           att2_conf_matrix, 
           delimiter = ',', 
           fmt = '%s')


# Store evaluation metrics
att2_train_accuracy = accuracy_score(y_train, 
                                     mdl_2.predict(X_train))
att2_L = list(precision_recall_fscore_support(y_test, 
                                              att2_y_pred, 
                                              average = 'weighted'))
metrics = metrics.append(pd.DataFrame([[2, 
                                        att2_train_accuracy, 
                                        att2_cv_accuracy, 
                                        att2_test_accuracy] + att2_L[:3]], 
                            columns = ['attempt', 
                                       'train accuracy', 
                                       'mean cv accuracy', 
                                       'test accuracy', 
                                       'test precision', 
                                       'test recall', 
                                       'test f1-score']), 
                         ignore_index = True)




# Final 2 attempts will mimic attempts 1 & 2, but with the train-test split   
# occuring prior to dimension reduction.  This is to ensure that no information
# from the hold-out test set is reflected in the training data as this could
# bias our result, thus providing an inaccurate evaluation of our classifier.

###############################################################################
# Attempt 3: Train-Test split prior to Dimension Reduction                    #
#            Build Classifier on Term Freq Document Matrix                    #
###############################################################################

# Data Preprocessing
# Train-test split prior to dimension reduction
X_train, X_test, y_train, y_test = train_test_split(tdm, 
                                                    y, 
                                                    test_size = 500, 
                                                    random_state = 0) 


# Perform Dimension Reduction
svd = TruncatedSVD(n_components = X_train.shape[1] - 1, 
                   random_state = 1234)
tsvd = svd.fit_transform(X_train)
svd_var_ratios = svd.explained_variance_ratio_

select_n_components(svd_var_ratios, 0.95)

svd_final = TruncatedSVD(n_components = select_n_components(svd_var_ratios, 
                                                            0.95), 
                         random_state = 1234)
X_train_final = svd_final.fit_transform(X_train)
X_test_final = svd_final.transform(X_test)

print('X_train = ' + str(X_train_final.shape))
print('X_test = ' + str(X_test_final.shape))



# Build Classifier
# Fit classifier using same random search parameters & modeling algorithm
# See mdl, accuracy, param_dist & rand_search above
mdl_3 = rand_search.fit(X_train_final, 
                        y_train)
att3_cv_accuracy = mdl_3.best_score_
att3_cv_accuracy 


# Return best estimator & parameters
print(mdl_3.best_estimator_)
print(mdl_3.best_params_)


# Evaluate on hold-out test sample
att3_y_pred = mdl_3.predict(X_test_final)
att3_test_accuracy = accuracy_score(y_test, 
                                    att3_y_pred)
att3_test_accuracy


# Classification Report
target_names = ['astronomy', 'aviation', 'beer', 'outdoors', 'pets']
att3_class_report = classification_report(y_test, 
                                          att3_y_pred, 
                                          target_names = target_names)
print(att3_class_report)

# Save report
with io.open('./att3_class_report.txt', 'w', encoding = 'utf-8') as class_report3: 
    class_report3.write(att3_class_report)
        
        
# Confusion Matrix
att3_conf_matrix = confusion_matrix(y_test, 
                                    att3_y_pred, 
                                    labels = target_names)
print(att3_conf_matrix)

# Save matrix
np.savetxt('.\\att3_conf_matrix.csv', 
           att3_conf_matrix, 
           delimiter = ',', 
           fmt = '%s')


# Store evaluation metrics
att3_train_accuracy = accuracy_score(y_train, 
                                     mdl_3.predict(X_train_final))
att3_L = list(precision_recall_fscore_support(y_test, 
                                              att3_y_pred, 
                                              average = 'weighted'))
metrics = metrics.append(pd.DataFrame([[3, 
                                        att3_train_accuracy, 
                                        att3_cv_accuracy, 
                                        att3_test_accuracy] + att3_L[:3]], 
                            columns = ['attempt', 
                                       'train accuracy', 
                                       'mean cv accuracy', 
                                       'test accuracy', 
                                       'test precision', 
                                       'test recall', 
                                       'test f1-score']), 
                         ignore_index = True)
    

    

###############################################################################
# Attempt 4: Train-Test split prior to Dimension Reduction                    #
#            Build Classifier on TF-IDF Matrix                                #
###############################################################################

# Data Preprocessing
# Split tf-idf into train & test sets using same methodology above
X_train, X_test, y_train, y_test = train_test_split(tfidf, 
                                                    y, 
                                                    test_size = 500, 
                                                    random_state = 0)


# Perform Dimension Reduction
tfidf_svd = TruncatedSVD(n_components = X_train.shape[1] - 1, 
                         random_state = 1234)
tfidf_tsvd = tfidf_svd.fit_transform(X_train)
tfidf_svd_var_ratios = tfidf_svd.explained_variance_ratio_

select_n_components(tfidf_svd_var_ratios, 0.95)

tfidf_svd_final = TruncatedSVD(n_components = select_n_components(tfidf_svd_var_ratios, 
                                                                  0.95), 
                               random_state = 1234)
X_train_final = tfidf_svd_final.fit_transform(X_train)
X_test_final = tfidf_svd_final.transform(X_test)

print('X_train = ' + str(X_train_final.shape))
print('X_test = ' + str(X_test_final.shape))



# Build Classifier
# Fit classifier using same random search parameters & modeling algorithm
# See mdl, accuracy, param_dist & rand_search above
mdl_4 = rand_search.fit(X_train_final, 
                        y_train)
att4_cv_accuracy = mdl_4.best_score_
att4_cv_accuracy


# Return best estimator & parameters
print(mdl_4.best_estimator_)
print(mdl_4.best_params_)


# Evaluate on hold-out test sample
att4_y_pred = mdl_4.predict(X_test_final)
att4_test_accuracy = accuracy_score(y_test, 
                                    att4_y_pred)
att4_test_accuracy


# Classification Report
target_names = ['astronomy', 'aviation', 'beer', 'outdoors', 'pets']
att4_class_report = classification_report(y_test, 
                                          att4_y_pred, 
                                          target_names = target_names)
print(att4_class_report)

# Save report
with io.open('./att4_class_report.txt', 'w', encoding = 'utf-8') as class_report4: 
    class_report4.write(att4_class_report)


# Confusion Matrix
att4_conf_matrix = confusion_matrix(y_test, 
                                    att4_y_pred, 
                                    labels = target_names)
print(att4_conf_matrix)

# Save matrix
np.savetxt('.\\att4_conf_matrix.csv', 
           att4_conf_matrix, 
           delimiter = ',', 
           fmt = '%s')


# Store evaluation metrics
att4_train_accuracy = accuracy_score(y_train, 
                                     mdl_4.predict(X_train_final))
att4_L = list(precision_recall_fscore_support(y_test, 
                                              att4_y_pred, 
                                              average = 'weighted'))
metrics = metrics.append(pd.DataFrame([[4, 
                                        att4_train_accuracy, 
                                        att4_cv_accuracy, 
                                        att4_test_accuracy] + att4_L[:3]], 
                            columns = ['attempt', 
                                       'train accuracy', 
                                       'mean cv accuracy', 
                                       'test accuracy', 
                                       'test precision', 
                                       'test recall', 
                                       'test f1-score']), 
                         ignore_index = True)
    
# Save metrics
pd.DataFrame.to_csv(metrics, './model_comparison.csv', sep = ',')