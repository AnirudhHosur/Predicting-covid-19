""" Please run each model indvitually as the same variables have been defined all over """# Importing Math Libraries 

import operator
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


# FITTING SIMPLE LINEAR REGRESSION TO THE TRAINING SET

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediciting the test set results
y_pred = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse)
print(r2)


# APPLYING DECISION TREE ON THE TRAINING DATASET

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = pd.DataFrame(regressor.predict(X_test))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse)
print(r2)

""" This will a little time :) """
# FITTING RANDOM FOREST TO THE TRAINING SET

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting on test set 
y_pred = pd.DataFrame(regressor.predict(X_test))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse)
print(r2)


# FITTING POLYNOMIAL REGRESSION THE THE TRAINING SET 

from sklearn.preprocessing import PolynomialFeatures 
poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(X_train) 
poly.fit(X_poly, y_train) 
poly_reg = LinearRegression() 
poly_reg.fit(X_poly, y_train) 

# Predicting a new result with Polynomial Regression 
y_pred = poly_reg.predict(poly.fit_transform(X_test)) 
y_pred = pd.DataFrame(y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse)
print(r2)






