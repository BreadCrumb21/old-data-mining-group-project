import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('./correctedDataFiles/adultDataCorrected.csv')
df['Workclass'] = df['Workclass'].astype('category')
df['Education'] = df['Education'].astype('category')
df['Marital-status'] = df['Marital-status'].astype('category')
df['Occupation'] = df['Occupation'].astype('category')
df['Relationship'] = df['Relationship'].astype('category')
df['Race'] = df['Race'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Native-country'] = df['Native-country'].astype('category')
df['Salary'] = df['Salary'].astype('category')

df['Workclass'] = df['Workclass'].cat.codes
df['Education'] = df['Education'].cat.codes
df['Marital-status'] = df['Marital-status'].cat.codes
df['Occupation'] = df['Occupation'].cat.codes
df['Relationship'] = df['Relationship'].cat.codes
df['Race'] = df['Race'].cat.codes
df['Sex'] = df['Sex'].cat.codes
df['Native-country'] = df['Native-country'].cat.codes
df['Salary'] = df['Salary'].cat.codes

data = pd.DataFrame(df[['Workclass', 'Education', 'Marital-status',
                        'Occupation', 'Relationship', 'Race', 'Sex', 'Native-country', 'Salary']])
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
