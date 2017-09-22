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

def slope_intercept(X_val,Y_val):
    x = np.array(X_val)
    y = np.array(Y_val)
    m = ((np.mean(x)*np.mean(y))-np.mean(x*y))/((np.mean(x)*np.mean(x))-np.mean(x*x))
    m = round(m,2)
    b = (np.mean(y)-np.mean(x)*m)
    b = round(b,2)
    return m,b

X1 = fm400[:,0] #years
Y1 = fm400[:,1] # first colum of time

xy = fm400[:,np.newaxis,0]
xt = fm400[:,np.newaxis,1]
m,b = slope_intercept(X1,Y1)
print('M ',m )
print('B',b)
reg_line1 = [(m*x)+b for x in X1]

xy1=np.array(xy)
a = np.polyfit(X1,Y1,3)
b = np.polyfit(X1,Y1,5)
d3 = np.poly1d(a)
d5= np.poly1d(b)
z5= np.array(d5(X1))
z3 = np.array(d3(X1))

print('Degree 3 means squared error: ', mean_squared_error(xt,d3(xy)))
print('Degree 5 mean squared error: ',mean_squared_error(xt,d5(xy)))
plt.plot(X1,d3(X1),color='blue')
plt.plot(X1,d5(X1),color='red')
plt.scatter(X1,Y1)
plt.plot(X1,reg_line1)

#-------------
x8 = np.array(xy)
y8 = np.array(d3(xt))

lo = cross_validation.LeaveOneOut(len(y8))
reg = linear_model.LinearRegression()
s = cross_validation.cross_val_score(reg,x8,y8,scoring='neg_mean_squared_error',cv=lo)
print(s)
print("Order 3 Polynomial Accuracy",s.mean(),s.std()*2)

y85 = np.array(d5(xy))

lo1 = cross_validation.LeaveOneOut(len(y85))
reg1 = linear_model.LinearRegression()
s1 = cross_validation.cross_val_score(reg1,x8,y85,scoring='neg_mean_squared_error',cv=lo1)
print(s1)
print("Order 5 Polynomial Accuracy",s.mean(),s.std()*2)


#--------------

rid = Ridge(alpha=0.1)
rid.fit(xy,y85)
print('Coef is ',rid.coef_)
print('Intercept is',rid.intercept_ )

#------------

rcv = RidgeCV(alphas=[0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0])
rcv.fit(xy,xt)
print('Best Value of Alpha across the range',rcv.alpha_)
plt.show()