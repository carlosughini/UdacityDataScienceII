#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
#print(len(enron_data))
#print(enron_data["SKILLING JEFFREY K"]["total_payments"])
#print(enron_data["FASTOW ANDREW S"]["total_payments"])
#print(enron_data["LAY KENNETH L"]["total_payments"])
## How many POI have "Nan" for their total payments? What percentage of POI's as a whole is this?
pois = 0
for _ in enron_data:
    if enron_data[_]["poi"]==1 and enron_data[_]["total_payments"]=="NaN":
        pois += 1
print(pois)

## Check how many have "NaN" for their salary
"""
nan_salaries = 0
for _ in enron_data:
    if enron_data[_]["total_payments"]=="NaN":
        nan_salaries += 1
print("Total:")
print(nan_salaries)
"""
## Check for email address and quantified salary
"""
emails = 0
salaries = 0
for _ in enron_data:
    if enron_data[_]["salary"]!="NaN":
        salaries += 1
    if enron_data[_]["email_address"]!="NaN":
        emails += 1
print(salaries)
print(emails)
"""
## Check for POI (Person of Interest)
"""
pois = 0
for _ in enron_data:
    if enron_data[_]["poi"]==1:
        pois += 1
    else:
        print("nada")
print("Final:")
print(pois)
"""

