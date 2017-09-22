from sklearn.linear_model import Ridge
from numpy import *
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
from sklearn import cross_validation

mat = spio.loadmat('olympics.mat') # loadded data from olympics.mat
m100 = mat['male100'] #extacted data of male100 in variable m100
m200 = mat['male200']
m400 = mat['male400']
fm100 = mat['female100']
fm200 = mat['female200']
fm400 = mat['female400']
X = m100[:,0] #first column of data set
Y = m100[:,1] #second column of data set
def slope_intercept(X_val,Y_val):
    x = np.array(X_val)
    y = np.array(Y_val)
    m = ((np.mean(x)*np.mean(y))-np.mean(x*y))/((np.mean(x)*np.mean(x))-np.mean(x*x))
    m = round(m,2)
    b = (np.mean(y)-np.mean(x)*m)
    b = round(b,2)
    return m,b
# predict function will predict the value

def predict_value(years,m,b):
    return (m*years + b)
m,b=slope_intercept(X,Y)
print('value of M is',m)
print('Value of B is',b)
print("predcition of year 2012 is",predict_value(2012,m,b))
print("predcition of year 2016 is",predict_value(2016,m,b))
reg_line = [(m*x)+b for x in X] # will print the linear regression line
plt.plot(X,reg_line,color="red")
X1 = fm400[:,0]
Y1 = fm400[:,1]
m1,b1 = slope_intercept(X1,Y1)
reg_line1 = [(m1*x)+b1 for x in X1]
plt.plot(X1,reg_line1,color="black")

plt.scatter(X,Y)
plt.ylabel('Time (In seconds)')
plt.xlabel('Year')
plt.title("Male100 Graph")
plt.show()

