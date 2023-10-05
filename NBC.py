# Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
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
x = test_data[feature_cols]
y = test_data.capital

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=125
)

model = GaussianNB()
model.fit(x_train, y_train)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

y_pred = model.predict(x_test)

accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)
