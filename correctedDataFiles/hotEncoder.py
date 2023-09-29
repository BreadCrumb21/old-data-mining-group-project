# import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy import savetxt

# import the data required
data = pd.read_csv('adultDataCorrected.csv')


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

encoder = OneHotEncoder()

enc_data = pd.DataFrame(encoder.fit_transform(data[['Workclass_new','Education_new','Marital-status_new','Occupation_new','Relationship_new','Race_new','Sex_new','Native-country_new','Salary_new']]).toarray())

New_df = data.join(enc_data)
print(New_df)

# # Extract only the numerical part of New_df
# numerical_data = New_df.select_dtypes(include=[np.number])

# # Get the column labels (column names) of the numerical_data DataFrame
# column_labels = numerical_data.columns.tolist()

# # Convert New_df to a NumPy array
# compatibleArr = numerical_data.values

# # Save the NumPy array to a CSV file
# savetxt('hotData.csv', compatibleArr, delimiter=',')

# Print the column labels
# print("Column Labels:", column_labels)
