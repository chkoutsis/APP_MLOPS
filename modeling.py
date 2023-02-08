import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

path = os.getcwd()
data = pd.read_csv(path + '\\aaaa.csv')

y = data['UserID']
X = data.drop(['UserID'], axis=1)

# # Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3) # 70% training and 30% test


#Create a svm Classifier
svc = SVC()
svc_classifier = OneVsRestClassifier(svc)
svc_classifier.fit(X_train, y_train)
y_pred = svc_classifier.predict(X_test)
joblib.dump(svc_classifier, 'svm_model.sav')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



#Create a Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
joblib.dump(rf_classifier, 'rf_model.sav')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Create a XGBoost Classifier
xgb_classifier = OneVsRestClassifier(XGBClassifier())
xgb_classifier.fit(X, y)
y_pred = xgb_classifier.predict(X_test)
joblib.dump(xgb_classifier, 'xgb_model.sav')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

