""" Some of my considerations include :
    1) Removed the 'Name' column as it makes a big issue to One Hot Encode it 
    2) Removed the 'Designation' column as the 'Gender' column takes care regarding the same
    3) The 'Region' column has also been removed because the cities in 'Training set' are diffrent 
       when compared to the 'Testing set'.
    4) The Training set has been further divided into traing and testing to figure out the best model

"""
# Data Preprosseing 
#importing the libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing the dataset
dataset_train = pd.read_csv('Train_dataset.csv')
X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, 24].values
dataset_train.head()


# Taing care of missing values

from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def __init__(self):

        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 

        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X_train, y=None):

        self.fill = pd.Series([X_train[c].value_counts().index[0]

            if X_train[c].dtype == np.dtype('O') else X_train[c].mean() for c in X_train],

            index=X_train.columns)

        return self

    def transform(self, X_train, y=None):

        return X_train.fillna(self.fill)



X_train = pd.DataFrame(X_train)
X_trainImp = DataFrameImputer().fit_transform(X_train)
print('before...')
print(X_train)
print('after...')
print(X_trainImp)

# One hot encoding the categorical values 

from sklearn.base import TransformerMixin
class DataFrameEncoder(TransformerMixin):

    def __init__(self):
        """Encode the data.

        Columns of data type object are appended in the list. After 
        appending Each Column of type object are taken dummies and 
        successively removed and two Dataframes are concated again.

        """
    def fit(self, X_trainImp, y=None):
        self.object_col = []
        for col in X_trainImp.columns:
            if(X_trainImp[col].dtype == np.dtype('O')):
                self.object_col.append(col)
        return self

    def transform(self, X_trainImp, y=None):
        dummy_df = pd.get_dummies(X_trainImp[self.object_col],drop_first=True)
        X_trainImp = X_trainImp.drop(X_trainImp[self.object_col],axis=1)
        X_trainImp = pd.concat([dummy_df,X_trainImp],axis=1)
        return X_trainImp
X_trainImpEnc = DataFrameEncoder().fit_transform(X_trainImp)


#Splitting the dataset into training and testing 
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X_trainImpEnc, y_train ,test_size = 0.2 , random_state = 0)






