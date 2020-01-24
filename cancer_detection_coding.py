# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np



dataset=pd.read_csv('modified1_data.csv')
dataset=dataset.drop("id",axis=1)
data={'M':1,'B':0}
dataset.diagnosis=[data[i] for i in dataset.diagnosis.astype(str)]

X = dataset.iloc[:,1:31]
y = dataset.iloc[:,0]

#estimator = GaussianNB()
#estimator=RandomForestClassifier(n_estimators=100)
estimator=LogisticRegression()
#estimator=svm.LinearSVC()
from sklearn.model_selection import learning_curve

def draw_learning_curves(X, y, estimator, num_trainings):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    import matplotlib.pyplot as plt

    plt.grid()

    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.plot(train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(test_scores_mean, 'o-', color="y",
             label="Cross-validation score")


    plt.legend(loc="best")

    plt.show()
    
draw_learning_curves(X,y,estimator,5)

train,test=train_test_split(dataset,test_size=0.33,random_state=0)
train_feature=train.ix[:,0:31]
train_label=train.ix[:,0]
test_feature=test.ix[:,0:31]
test_label=test.ix[:,0]

#from sklearn.metrics import confusion_matrix
import seaborn as sns
#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


estimator.fit(train_feature,train_label)
prediction=estimator.predict(test_feature)
cm = confusion_matrix(test_label, prediction)
sns.heatmap(cm, annot=True)
print(classification_report(test_label, prediction))

print("The OutPut Prediction Are Given Below:  \n\n")

prediction1=[]
for j in prediction:
    if j==1:
        prediction1.append("M")
    else:
        prediction1.append("B")
        
print(prediction1)

#from sklearn.metrics import accuracy_score
#accuracy_score(test_label, prediction)
Accuracy_Score = accuracy_score(test_label, prediction)
Precision_Score = precision_score(test_label, prediction,  average="macro")
Recall_Score = recall_score(test_label, prediction,  average="macro")
F1_Score = f1_score(test_label, prediction,  average="macro")

print('Average Accuracy: %0.2f +/- (%0.1f) %%' % (Accuracy_Score.mean()*100, Accuracy_Score.std()*100))
print('Average Precision: %0.2f +/- (%0.1f) %%' % (Precision_Score.mean()*100, Precision_Score.std()*100))
print('Average Recall: %0.2f +/- (%0.1f) %%' % (Recall_Score.mean()*100, Recall_Score.std()*100))
print('Average F1-Score: %0.2f +/- (%0.1f) %%' % (F1_Score.mean()*100, F1_Score.std()*100))





