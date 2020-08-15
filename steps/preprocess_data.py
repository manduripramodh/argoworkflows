import pickle
import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pprint import pprint

np.random.seed(123456)

root_path = '/training'

if __name__ == '__main__':
    pprint("Entered Preprocessing step")
    # Importing the data
    df = pd.read_csv("german_credit_data.csv", index_col=0)

    df_good = df.loc[df["Risk"] == 'good']['Age'].values.tolist()
    df_bad = df.loc[df["Risk"] == 'bad']['Age'].values.tolist()
    df_age = df['Age'].values.tolist()

    # Let's look the Credit Amount column
    interval = (18, 25, 35, 60, 120)

    cats = ['Student', 'Young', 'Adult', 'Senior']
    df["Age Bucket"] = pd.cut(df.Age, interval, labels=cats)

    df_good = df[df["Risk"] == 'good']
    df_bad = df[df["Risk"] == 'bad']

    # Purpose
    df['Purpose'].replace(
        ['radio/TV', 'education', 'furniture/equipment', 'car', 'business', 'domestic appliances', 'repairs',
         'vacation/others'], [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)

    # Sex
    df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

    # Housing
    df['Housing'].replace(['own', 'free', 'rent'], [0, 1, 2], inplace=True)

    # Saving accounts
    df['Saving accounts'] = df['Saving accounts'].fillna('empty')
    df['Saving accounts'].replace(['empty', 'little', 'moderate', 'quite rich', 'rich', ], [0, 1, 2, 3, 4],
                                  inplace=True)

    # Checking accounts
    df['Checking account'] = df['Checking account'].fillna('empty')
    df['Checking account'].replace(['empty', 'little', 'moderate', 'rich', ], [0, 1, 2, 3], inplace=True)

    # Age Bucket
    df['Age Bucket'].replace(['Student', 'Young', 'Adult', 'Senior'], [0, 1, 2, 3], inplace=True)

    # Risk
    df['Risk'].replace(['good', 'bad'], [0, 1], inplace=True)

    del df["Age"]

    df['Credit amount'] = np.log(df['Credit amount'])

    # Creating the X and y variables
    X = df.drop('Risk', 1)
    y = df["Risk"]

    # Spliting X and y into train and test version
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    os.makedirs('{}/output/'.format(root_path), exist_ok=True)
    # X_train.to_pickle('{}/output/train.pickle'.format(root_path))
    # y_train.to_pickle('{}/output/ytrain.pickle'.format(root_path))
    os.makedirs('{}/output/'.format(root_path), exist_ok=True)
    with open('{}/output/train.pickle'.format(root_path), 'wb') as f:
        pickle.dump([X_train, y_train], f)
    # X_train.to_pickle('train.pickle')
    pprint("Finished Preprocessing step")
