#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import warnings
warnings.filterwarnings('ignore')
import tester
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary' , 'deferral_payments', 'total_payments',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Exploring dataset
# People in the dataset
people = len(data_dict)
print "There are " + str(people) + " people in the dataset."

# POIs in the dataset
pois = 0
for person in data_dict:
    if data_dict[person]["poi"]:
        pois += 1

print "There are  " + str(pois) + " POI's in the dataset."

# Total Poi
fpoi = open("poi_names.txt", "r")
rfile = fpoi.readlines()
poi = len(rfile[2:])
print "There were " + str(poi) + " poi's total."

### Task 2: Remove outliers
features = ["bonus", "salary"]
data = featureFormat(data_dict, features)

# Plotting
print data.max()
for point in data:
    bonus = point[0]
    salary = point[1]
    plt.scatter( bonus, salary )

plt.xlabel("bonus")
plt.ylabel("salary")
plt.show()

## Check what is this outlier
from pprint import pprint
bonus_outliers = []
for key in data_dict:
    val = data_dict[key]['bonus']
    if val == 'NaN':
        continue
    bonus_outliers.append((key,int(val)))

pprint(sorted(bonus_outliers,key=lambda x:x[1],reverse=True)[:2])

salary_outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    salary_outliers.append((key,int(val)))

pprint(sorted(salary_outliers,key=lambda x:x[1],reverse=True)[:2])

# Print the number of employees
print "Numbers of employees ", len(data_dict)

# Removing data points as it will cause outliers in features
outliers = ['TOTAL', 'LOCKHART EUGENE E', 'TRAVEL AGENCY', 'LAVORATO JOHN J', 'SKILLING JEFFREY K']
for outlier in outliers:
    data_dict.pop(outlier, 0)

# After removing the outliers we can confirm that reduced the size
print "Numbers of employees ", len(data_dict)

# Removing all 'loan_advances' as they are missing values
for name in data_dict:
    data_dict[name].pop('loan_advances',0)

# After removing the outliers
for point in data_dict:
    bonus = point[0]
    salary = point[1]
    plt.scatter( bonus, salary )

plt.xlabel("bonus")
plt.ylabel("salary")
plt.show()

### Task 3: Create new feature(s)
# Use scikit-learn's SelectKBest feature selection:
from sklearn.feature_selection import SelectKBest
# Number of features
k=10
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
k_best = SelectKBest(k=k)
k_best.fit(features, labels)
scores = k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
k_best_features = dict(sorted_pairs[:k])
print "{0} best features: {1}\n".format(k, k_best_features.keys())

# Print the KBest features where returns keys = features and values = scores
print k_best_features

# Feature for fraction bonus
for value in data_dict:
    if data_dict[value]["total_payments"] == 0:
        data_dict[value]["fraction_bonus"] = 0.0
    elif data_dict[value]["bonus"] == "NaN" or data_dict[value]["total_payments"] == "NaN":
        data_dict[value]["fraction_bonus"] = "NaN"
    else:
        data_dict[value]["fraction_bonus"] = float(data_dict[value]["bonus"]) / float(data_dict[value]["total_payments"])

# Feature for fraction salary
for value in data_dict:
    if data_dict[value]["total_payments"] == 0:
        data_dict[value]["fraction_salary"] = 0.0
    elif data_dict[value]["salary"] == "NaN" or data_dict[value]["total_payments"] == "NaN":
        data_dict[value]["fraction_salary"] = "NaN"
    else:
        data_dict[value]["fraction_salary"] = float(data_dict[value]["salary"]) / float(data_dict[value]["total_payments"])

# Feature for fraction total_stock_value
for value in data_dict:
    if data_dict[value]["total_payments"] == 0:
        data_dict[value]["fraction_stock"] = 0.0
    elif data_dict[value]["total_stock_value"] == "NaN" or data_dict[value]["total_payments"] == "NaN":
        data_dict[value]["fraction_stock"] = "NaN"
    else:
        data_dict[value]["fraction_stock"] = float(data_dict[value]["total_stock_value"]) / float(data_dict[value]["total_payments"])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Gaussian
from sklearn.naive_bayes import GaussianNB
clfGB = GaussianNB()
fit = clfGB.fit(features_train, labels_train)
predGB = clfGB.predict(features_test)
#tester.test_classifier(clfGB, data_dict, features_list)
gb_score = clfGB.score(features_test, labels_test)
gb_precision = precision_score(labels_test, predGB)
gb_recall = recall_score(labels_test, predGB)
print ('GaussianNB accuracy:', gb_score)
print ('GaussianNB precision:', gb_precision)
print ('GaussianNB recall:', gb_recall, '\n')

# SVC
from sklearn.svm import SVC
clfSV = SVC(kernel='linear',max_iter=1000)
fit = clfSV.fit(features_train, labels_train)
predSV = clfSV.predict(features_test)
#print tester.test_classifier(clfSV, data_dict, features_list)
sv_score = clfSV.score(features_test, labels_test)
sv_precision = precision_score(labels_test, predSV)
sv_recall = recall_score(labels_test, predSV)
print ('SVC accuracy:', sv_score)
print ('SVC precision:', sv_precision)
print ('SVC recall:', sv_recall, '\n')

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clfDT = DecisionTreeClassifier()
fit = clfDT.fit(features_train, labels_train)
predDT = clfDT.predict(features_test, labels_test)
#tester.test_classifier(clfDT, data_dict, features_list)
dt_score = clfSV.score(features_test, labels_test)
dt_precision = precision_score(labels_test, predDT)
dt_recall = recall_score(labels_test, predDT)
print ('DecisionTree accuracy:', dt_score)
print ('DecisionTree precision:', dt_precision)
print ('DecisionTree recall:', dt_recall, '\n')
print "\n"

# Printing the report
#target_names = ['Not PoI', 'PoI']
#print "Report GaussianNB: ", classification_report(labels_test, predGB, target_names=target_names)
#print "Report SVC: ", classification_report(labels_test, predSV, target_names=target_names)
#print "Report DecisionTree: ", classification_report(labels_test, predDT, target_names=target_names)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

# Tuning with GridSearchCV
score_list = []

scaler = MinMaxScaler()
select = SelectKBest()
gnb = GaussianNB()

steps = [('scaler', scaler),
         ('feature_select', select),
         ('classifier', gnb)]

param_grid = {'feature_select__k': range(1,17)}
sss = StratifiedShuffleSplit(100, test_size=0.3, random_state = 0)
pipe = Pipeline(steps)
gs = GridSearchCV(pipe, param_grid, cv=sss, scoring='f1')

gs.fit(features_train, labels_train)
clf = gs.best_estimator_

print "Testing the tuning the algorithm"
print test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
