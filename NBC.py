# Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

data = pd.read_csv('hotData.csv')
print(data.isna().sum())
x = data.drop(columns=['Workclass_new', 'Education_new', 'Marital-status_new', 'Occupation_new', 'Relationship_new', 'Race_new', 'Sex_new','Native-country_new', 'Salary_new'])
y = data['Salary_new']

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
