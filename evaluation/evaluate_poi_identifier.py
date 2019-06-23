#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

clf = DecisionTreeClassifier()
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "[Q1] How many POIs are predicted for the test set for your POI identifier?"
print "[A1]", sum(pred)
print "[Q2] How many people total are in your test set?"
print "[A2]", len(pred)
print "[Q3] If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?"
print "[A3]", pred.tolist().count(0) / float(len(pred))
print "[Q4] Do you get any true positives? (In this case, we define a true positive as a case where both the actual label and the predicted label are 1)"
true_positives = 0
for i in range(len(pred)):
    if (pred[i] == labels_test[i]) and labels_test[i] == 1:
        true_positives += 1
print "[A3]", true_positives
print "Precision score:", precision_score(pred, labels_test)
print "Recall score:", recall_score(pred, labels_test)


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

# What's the precision of this classifier?
print precision_score(true_labels, predictions)

# What's the recall of this classifier?
print recall_score(true_labels, predictions)



