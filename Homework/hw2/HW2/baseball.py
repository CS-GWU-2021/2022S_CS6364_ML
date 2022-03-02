# -------------------------------------------------------------------------
# Author:   Tenphun0503
# Intro:    This part analyze the baseball player salaries problem
# -------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import temp

data_file = 'D:/2022 Spring/2022S_CS6364_ML/Homework/hw2/public/baseball-9192.csv'

data = pd.read_csv(data_file)
X = data.iloc[:,2:-1]
y = data.iloc[:,1]

#X = X[['Runs','Hits','Doubles','HomeRuns','RBI','Walks']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

coe = []
mse = []
coe_de = []
for i in range(0, 16):
    X_train_i = X_train.iloc[:,i].values.reshape(-1,1)
    X_test_i = X_test.iloc[:,i].values.reshape(-1,1)
    regr = LinearRegression()
    regr.fit(X_train_i, y_train.values.reshape(-1,1))
    y_pred = regr.predict(X_test_i)
    # The coefficients
    coe.append(regr.coef_)
    #print("Coefficients: \n", regr.coef_)
    # The mean squared error
    mse.append(mean_squared_error(y_test.values.reshape(-1,1), y_pred))
    #print("Mean squared error: %.2f" % mean_squared_error(y_test.values.reshape(-1,1), y_pred))
    # The coefficient of determination: 1 is perfect prediction
    coe_de.append(r2_score(y_test.values.reshape(-1,1), y_pred))
    #print("Coefficient of determination: %.2f" % r2_score(y_test.values.reshape(-1,1), y_pred))
    '''
    # Plot outputs
    plt.scatter(X_test_i, y_test, color="black")
    plt.plot(X_test_i, y_pred, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    '''
result = {
    'features' : X_train.columns,
    'coefficient' : coe,
    'mean squared error' : mse,
    'coefficient of determination' : coe_de
}
frame = pd.DataFrame(result)
frame = frame.sort_values(by=['mean squared error'])
print(frame)
