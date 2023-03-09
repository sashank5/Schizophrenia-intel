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
model = AdaBoostClassifier(estimator=clf, n_estimators=500)
model.fit(X_train, y_train)

import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

pkl.dump(model,open('model.pkl','wb'))

