#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.
    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
    """

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
clf = svm.SVC(C=10000.0,kernel='rbf',gamma='auto')
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
t0 = time()
clf.fit(features_train, labels_train)
print "training time training:", round(time()-t0, 3), "s"
pred = clf.predict(features_test)
print(pred)
print "No. of predicted to be in the 'Chris'(1): %r" % sum(pred)
#answer = pred[10]
#print("answer")
#print(answer)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print(acc)

#########################################################
