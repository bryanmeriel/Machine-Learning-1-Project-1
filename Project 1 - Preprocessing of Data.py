import pandas as pd
import sklearn as sk
import numpy as np
import math
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import dmba
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

Housing_df = pd.read_csv("HousingDataSet.csv")

#Checking Data Types
Housing_df.info()  #Date, bathrooms and floors have incorrect data types

#Checked for missing values using isnull()
print(Housing_df.isnull().sum())

#Changed 'date' from object to datetime
Housing_df['date']=pd.to_datetime(Housing_df['date'].astype(str), format='%Y/%m/%d')

#Changed format to M-D-Y
Housing_df['date'] = Housing_df['date'].dt.strftime('%m-%d-%Y')
#THIS PART CHANGES THE FORMAT, BUT ALSO CHANGES DATATYPE BACK TO OBJECT? UNSURE WHY
Housing_df.info()

#Checking range of values in both bathroom and floor
Housing_df['bathrooms'].unique()
Housing_df['floors'].unique()

#Changing data type to Integers for the columns bathrooms and floors
Housing_df[['bathrooms', 'floors']] = Housing_df[['bathrooms', 'floors']].astype(int)

#Changes are reflected, will also now drop unnecessary columns:'yr_renovated','sqft_living15' & 'sqft_lot15' columns
Housing_df.head()

#dropping 'yr_renovated','sqft_living15','sqft_lot15'
Housing_df = Housing_df.drop(['yr_renovated','sqft_living15','sqft_lot15'],axis=1)

#reviewing descriptive statistics to check for outliers or other inconsistencies
Housing_df.describe()

#Keeping rows with less than 15 bedrooms, (33 bedrooms seems unrealistic)
Housing_df = Housing_df.loc[Housing_df["bedrooms"] < 15]

#All values in sqft_basement that have basement (>=1) = 1
Housing_df['sqft_basement'].mask(Housing_df['sqft_basement'] >=1 ,1, inplace=True)

#max bedrooms is now 11 and sqft_basement reflects changes
Housing_df.describe()

#lastly, we need to convert 'waterfront','view' and 'sqft_basement' into categorical data types
Housing_df[['waterfront','view','sqft_basement']] = Housing_df[['waterfront','view','sqft_basement']].astype('category')

Housing_df.dtypes #all data types are now correct, EXCEPT date (should be datetime)

#Changed column name of 'sqft_basement' to just 'basement'
Housing_df.rename(columns={"sqft_basement":"basement"})

print(Housing_df) #all changes are reflected


#Grabbing all of the Numeric values
Housing_df_numeric = Housing_df.drop(['id','date','price','zipcode','lat','long'], axis = 1)

#Normalization
scaler = preprocessing.MinMaxScaler()
#normalizing the data
scaler = MinMaxScaler()
names = Housing_df_numeric.columns
d = scaler.fit_transform(Housing_df_numeric)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()

#Estimating feature importance
X = scaled_df
y = Housing_df['price']
model = LinearRegression()
model.fit(X, y)
importance = model.coef_

for i, v in enumerate(importance):
    print(f"Feature {i}, Score: {v}")
    
#Predictors used for Linear Regression
predictors = ['waterfront','bedrooms','view','grade','floors']
x = pd.get_dummies(Housing_df_numeric[predictors], drop_first=True)
y = Housing_df['price']
x.head()

#Test/Train/Split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.6, shuffle=True)

#Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

#To get the number of training,testing + total
y_predicted_training = model.predict(X_train)
y_predicted_test = model.predict(X_test)
print ("train size={}, test_size={}, total_size={}".format(
    X_train.shape[0], X_test.shape[0], Housing_df_numeric.shape[0]))

#Prediction
y_predict=model.predict(X_test)
print(y_predict)
print(y_test)

#Prediction results with residuals
model_pred = model.predict(X_test)
result = pd.DataFrame({'Predicted': model_pred, 'Actual': y_test,
'Residual': y_test - model_pred})
print(result.head(20))

#Summary + Prediction rate for the variables used
regressionSummary (y_test, model_pred)

print('Prediction rate: %.3f' % r2_score(y_test,y_predict))

#Training and Predicting KNN Regression
#K=5
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(X_train, y_train)

#Make predictions
y_pred = regressor.predict(X_test)

#Calculate error
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')

#Training and Predicting KNN Regression
#K=10
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=10)
regressor.fit(X_train, y_train)

#Make predictions
y_pred = regressor.predict(X_test)


#Calculate error
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')



# Train a classifier for different values of k - It didnt run
results = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    results.append({
        'k': k,
        'accuracy': accuracy_score(y_pred, knn.predict(X_test))
    })
    
# Convert results to a pandas data frame
results = pd.DataFrame(results)
print(results)


#trying to find the parameters for regression tree

param_grid = {
    'max_depth': [5, 10, 15, 20, 25],
    'min_impurity_decrease': [0, 0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.008],
    'min_samples_split': [14, 15, 16, 18, 20]
}

gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Improved parameters: ', gridSearch.best_params_)

param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'min_impurity_decrease': [0, 0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.008],
    'min_samples_split': [14, 15, 16, 18, 20]
}

gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Improved parameters: ', gridSearch.best_params_)

regTree = gridSearch.best_estimator_

#printing results of testing and training

print("Regression Tree Training Data")
dmba.regressionSummary(train_y, regTree.predict(train_X))

print("\nRegression Tree Test Data")
dmba.regressionSummary(valid_y, regTree.predict(valid_X))
