# Hackathon

# Data Preprosseing 
#importing the libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing the dataset
dataset_test = pd.read_csv('Test_dataset.csv')
X_test = dataset_test.iloc[:, :].values
	


dataset_test.head()

# Taing care of missing values


from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):
    def __init__(self):

        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 

        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X_test, y=None):

        self.fill = pd.Series([X_test[c].value_counts().index[0]

            if X_test[c].dtype == np.dtype('O') else X_test[c].mean() for c in X_test],

            index=X_test.columns)

        return self

    def transform(self, X_test, y=None):

        return X_test.fillna(self.fill)



X_test = pd.DataFrame(X_test)
X_testImp = DataFrameImputer().fit_transform(X_test)
print('before...')
print(X_test)
print('after...')
print(X_testImp)

# One hot encoding the categorical values 

from sklearn.base import TransformerMixin
class DataFrameEncoder(TransformerMixin):

    def __init__(self):
        """Encode the data.

        Columns of data type object are appended in the list. After 
        appending Each Column of type object are taken dummies and 
        successively removed and two Dataframes are concated again.

        """
    def fit(self, X_testImp, y=None):
        self.object_col = []
        for col in X_testImp.columns:
            if(X_testImp[col].dtype == np.dtype('O')):
                self.object_col.append(col)
        return self

    def transform(self, X_testImp, y=None):
        dummy_df = pd.get_dummies(X_testImp[self.object_col],drop_first=True)
        X_testImp = X_testImp.drop(X_testImp[self.object_col],axis=1)
        X_testImp = pd.concat([dummy_df,X_testImp],axis=1)
        return X_testImp
X_testImpEnc = DataFrameEncoder().fit_transform(X_testImp)










