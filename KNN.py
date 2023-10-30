from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from DataConstruction import construct_cat_value


data = construct_cat_value()

x_all = data.drop(['Salary'], axis=1)
x_all = x_all.tail(20)
y_all = data['Salary']
y_all = y_all.tail(20)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.5, random_state=42)

k_value = 10
classifier = KNeighborsClassifier(n_neighbors=k_value)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print("K-Value " + str(k_value) + ": ")
print(classification_report(y_test, y_pred, zero_division=0))
