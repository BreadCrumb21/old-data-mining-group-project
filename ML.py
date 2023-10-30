import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DataConstruction import construct_cat_value

data = construct_cat_value()
x_all = data.drop(['Salary'], axis=1)
y_all = data['Salary']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)

# Create a Multi-Layer Perceptron (MLP) Neural Network Classifier
mlp = MLPClassifier(max_iter=1000, verbose=True, activation='tanh', alpha=.00005, hidden_layer_sizes=(150, 150), learning_rate='invscaling', learning_rate_init=.0005)

# Train the classifier on the training data
mlp.fit(x_train, y_train)

# Make predictions on the test data
y_pred = mlp.predict(x_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Neural Network Classifier Accuracy: {accuracy:.3f}")
