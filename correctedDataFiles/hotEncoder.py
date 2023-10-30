import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy import savetxt

# Import the data required from a CSV file named 'adultDataCorrected.csv' into a Pandas DataFrame
data = pd.read_csv('adultDataCorrected.csv')

# Convert categorical columns to category data type and create new numerical columns
# These new numerical columns will contain the category codes of the original categorical data
data['Workclass'] = data['Workclass'].astype('category')
data['Education'] = data['Education'].astype('category')
data['Marital-status'] = data['Marital-status'].astype('category')
data['Occupation'] = data['Occupation'].astype('category')
data['Relationship'] = data['Relationship'].astype('category')
data['Race'] = data['Race'].astype('category')
data['Sex'] = data['Sex'].astype('category')
data['Native-country'] = data['Native-country'].astype('category')
data['Salary'] = data['Salary'].astype('category')

data['Workclass_new'] = data['Workclass'].cat.codes
data['Education_new'] = data['Education'].cat.codes
data['Marital-status_new'] = data['Marital-status'].cat.codes
data['Occupation_new'] = data['Occupation'].cat.codes
data['Relationship_new'] = data['Relationship'].cat.codes
data['Race_new'] = data['Race'].cat.codes
data['Sex_new'] = data['Sex'].cat.codes
data['Native-country_new'] = data['Native-country'].cat.codes
data['Salary_new'] = data['Salary'].cat.codes

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()

# Perform one-hot encoding on selected columns and convert the result to a DataFrame
enc_data = pd.DataFrame(encoder.fit_transform(data[['Workclass_new', 'Education_new',
                                                    'Marital-status_new', 'Occupation_new',
                                                    'Relationship_new', 'Race_new', 'Sex_new',
                                                    'Native-country_new', 'Salary_new']]).toarray())

# Join the original data DataFrame with the encoded DataFrame to create a new DataFrame
New_df = data.join(enc_data)

# # Print the new DataFrame
# print(New_df)

# # Extract only the numerical part of New_df to a CSV file
# numerical_data = New_df.select_dtypes(include=[np.number])

# # Get the column labels (column names) of the numerical_data DataFrame
# column_labels = numerical_data.columns.tolist()

# # Convert New_df to a NumPy array
# compatibleArr = numerical_data.values

# # Save the NumPy array to a CSV file named 'hotData.csv'
# savetxt('hotData.csv', compatibleArr, delimiter=',')

# # Print the column labels
# print("Column Labels:", column_labels)
