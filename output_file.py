""" Please restart the kernel & clear all the variables , then run the entire code from 
    covidTrain.py and covidTest.py
    Then proceed with running the entire code in this section """
# This will a little time :)
# Random Forset selected

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the probability of a person getting infected by covid-19

y_pred = pd.DataFrame(regressor.predict(X_testImpEnc))
""" Do Open 'y_pred' variable from the 'Variable Explorer Tab' , Thank You """
