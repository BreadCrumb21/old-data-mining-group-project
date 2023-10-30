import pandas as pd


def construct_cat_value():
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

    return pd.DataFrame(df[['Workclass', 'Education', 'Marital-status',
                            'Occupation', 'Relationship', 'Race', 'Sex', 'Native-country', 'Salary']])
