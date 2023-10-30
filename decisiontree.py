# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import OrdinalEncoder


col_names = ['type','education','status','service','relation','race','gender','location','capital']
# load dataset
test_data = pd.read_csv("adultTestCorrected.test", header=None, names=col_names)
# print(test_data.head())

enc = OrdinalEncoder()
#encoded_data = pd.DataFrame(enc.fit_transform(test_data))
#print(encoded_data.head())
enc.fit(test_data[["type","education","status","service","relation","race","gender","location","capital"]])
test_data[["type","education","status","service","relation","race","gender","location","capital"]] = enc.fit_transform(
    test_data[["type","education","status","service","relation","race","gender","location","capital"]])

# print(test_data)

feature_cols = ['type','status','relation','race','gender','education','service','location'] 
X = test_data[feature_cols]
y = test_data.capital

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
print("Precision: ",metrics.precision_score(y_test,y_pred,average="weighted"))
print("F1 Score: ",metrics.f1_score(y_test,y_pred,average="weighted"))