#set working directory
import os
os.chdir('C:/Users/unkno/Desktop/MS Data Science/Class 9 - ANA680/Week 2/midterm')

#import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle as pkl

#import csv file
data = pd.read_csv('StudentsPerformance.csv')

#print NaN values in data
print('Number of NaN values in data: ', data.isnull().sum())

#print the first 5 rows of the data
print(X.head())

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'parental level of education', 'lunch', 'test preparation course', 'race/ethnicity']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

#print encoded values for each encoded column (for UI purposes)
for column, le in label_encoders.items():
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f'Mapping for {column}: {mapping}')

#separate features and target 
X = data.drop('race/ethnicity', axis=1)
y = data['race/ethnicity']

#flatten y into a 1D array
y = np.ravel(y)

#split data into training and testing sets (testing = 25% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)

#create a KNN model
model = KNeighborsClassifier(n_neighbors=3)

#fit the model
model.fit(X_train, y_train)

#predict the target values
y_pred = model.predict(X_test)

#predict the accuracy of the model
print('Accuracy: ', accuracy(y_test, y_pred))

#confusion matrix
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))

#print all feature keys and values
print('Feature keys and values: ')

for key, value in zip(X.keys(), model.feature_importances_):
    print(key, value)

# model has a 27.2% accuracy


# Save the model
pkl.dump(model, open('midterm_model.pkl', 'wb'))