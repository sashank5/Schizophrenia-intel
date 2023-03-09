import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

survey_data=pd.read_csv('depression.csv')
survey_data.drop(['Timestamp','Username','Name'],axis=1,inplace=True)
missing_data1=survey_data.isnull().sum()

features_needed =['Age', 'Gender', 'physical_health','mental_health','workpressure','emotionalpressure','mental_ability_workdone','mental_relationship','felt_low','diet_change','last_happy','felt_good','positive','history_disorder','therapist','medication','sleep_quality','smoke','drink']

encoder=LabelEncoder()
for i in survey_data:
    survey_data[i] = encoder.fit_transform(survey_data[i])

features = survey_data[features_needed]
target = survey_data['mental_disorder']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
clf.fit(X_train, y_train)

prob = clf.predict_proba(X_train)[0][1]
print("Probability of disease occurrence:", prob)
