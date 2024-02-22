import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Downloading the data
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
df = pd.read_csv(filepath, header=0)

#print(df.head().to_string())

# Exploring Simple Linear Regression
lm = LinearRegression()

X = df[["highway-mpg"]]
Y = df["price"]

lm.fit(X, Y)

Yhat = lm.predict(X)
#print(Yhat[0:5])

#print(lm.intercept_)
#print(lm.coef_)

X1 = df[["engine-size"]]
Y1 = df["price"]

lm.fit(X1, Y1)
Yhat1 = lm.predict(X1)
#print(Yhat1[0:5])
#print(lm1.coef_)
#print(lm1.intercept_)

# Exploring Multiple Linear Regression

Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
lm.fit(Z, df["price"])
#print(lm.intercept_)
#print(lm.coef_)

lm2 = LinearRegression()
lm2.fit(df[["normalized-losses", "highway-mpg"]],df["price"])
#print(lm2.coef_)

# Model evaluation

# Visualization of correlation using regression plots (Simple Linear Regression)

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)


plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
#plt.show()

#print(df[["peak-rpm","highway-mpg","price"]].corr())

# Visualization of residual (Simple Linear Regression)
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-mpg'], y=df['price'])
#plt.show()

Y_hat = lm.predict(Z)

plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

#plt.show()
plt.close()

# Exploring Polynomial Regression Model

# Plotting polynomial data
def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

# Setting variables for the model
x = df['highway-mpg']
y = df['price']

#  Fitting and displaying the 3rd order polynomial using the functions of polyfit and poly1d
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
#print(p)

# Plotting
PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)
plt.show()

#  Fitting and displaying the 11th order polynomial using the functions of polyfit and poly1d
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
#print(p1)

# Plotting
PlotPolly(p1, x, y, 'highway-mpg')
np.polyfit(x, y, 11)
#plt.show()

# Exploring Multivariate Polynomial Model

pr=PolynomialFeatures(degree=2)
pr

# Transformation to Multivariate Polynomial
Z_pr=pr.fit_transform(Z)

# Checking number of samples and features
#print(Z.shape)
# Checking number of samples and features after transformation
#print(Z_pr.shape)

# Creating Pipeline for polynomial regression model
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

# Pipeline contructor
pipe=Pipeline(Input)
#print(pipe)

# Converting Z data to float (for StandardScaler input)
Z = Z.astype(float)
pipe.fit(Z,y)

# Performing prediction
ypipe=pipe.predict(Z)
print(ypipe[0:4])

# Creating Pipeline for linear regression model
Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
print(ypipe[0:10])

# Measuring accurateness of model (In-Sample Evaluation)

# Simple Linear Regression

lm.fit(X, Y)
# Finding the R^2 (coefficient of determination)
print('The R-square is: ', lm.score(X, Y))

# Prediction
Yhat1=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat1[0:4])

# Comparing predicted results with the real data
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

# Multiple Linear Regression

lm.fit(Z, df['price'])

# Finding the R^2 (coefficient of determination)
print('The R-square is: ', lm.score(Z, df['price']))

# Prediction
Y_predict_multifit = lm.predict(Z)

# Comparing predicted results with the real data
print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], Y_predict_multifit))

# Polynomial fit

# Finding the R^2 (coefficient of determination)
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

# Comparing predicted results with the real data
mean_squared_error(df['price'], p(x))

# Producing Prediction for Decision-making
new_input=np.arange(1, 100, 1).reshape(-1, 1)

#Fitting the model
lm.fit(X, Y)

#Making a prediction:
yhat=lm.predict(new_input)
yhat[0:5]

plt.plot(new_input, yhat)
plt.show()

#Result: the MLR model is the best model
"""
Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price.

    R-squared: 0.49659118843391759
    MSE: 3.16 x10^7

Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.

    R-squared: 0.80896354913783497
    MSE: 1.2 x10^7

Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.

    R-squared: 0.6741946663906514
    MSE: 2.05 x 10^7
"""