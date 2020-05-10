#import important libraries
import pandas as pd
import numpy as np

#import LinearRegression
from sklearn.linear_model import LogisticRegression

#import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

#load the dataset
df = pd.read_csv("D:\HeartDiseaseornot\Logistic Regression\heartdiseaseEDA.csv")

#spliting into X(label) and y(target)
X = df.drop("target",axis = 1)  #leaving the target column
y = df["target"]

#split into training and validation 
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size = 0.2) #80% data for training


#model preparation
#parameters are decided based on hyperparameter tunning using RandomizedSearchCV
clf = LogisticRegression(penalty='l2',C=1.373824,solver='liblinear')

#fitting the model
clf.fit(X_train,y_train)
 
#lets see the accuracy
accuracy = clf.score(X_valid,y_valid)

#lets see the accuracy based on crossvalidation with 5 holds
accuracyCV = cross_val_score(clf,X,y,cv=5)
accuracyCV = np.mean(accuracyCV)

#now for confusion matrix
ypredicts = clf.predict(X_valid)
y_df = pd.DataFrame(
    {
        "actual":y_valid,
        "predicted":ypredicts
    }
)
confusion_matrix = pd.crosstab(y_df["actual"],y_df["predicted"])

classification_report_result = classification_report(y_valid,ypredicts)

result ={
    'model_accuracy' : str(round(accuracy*100,3)) +'%',
    'model_CV_accuracy': str(round(accuracyCV*100,3)) +'%',
    'confusion_matrix' : confusion_matrix,
    'classification_report': classification_report_result
}

print("---------------Overall result--------------")
for key,value in result.items():
    print(f"{key}:\n{value}")
    print("-----------------------------------")




