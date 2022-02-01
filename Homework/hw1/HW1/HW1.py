# --------------------------------------------------------------------------------
# Author:       Tenphun0503
# Intro:        Main code; Find best k for knn, and make prediction
# --------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import tools

# files direction
train_file = 'D:/2022 Spring/2022S_CS6364_ML/Homework/hw1/public/data_mnist.csv'
test_file = 'D:/2022 Spring/2022S_CS6364_ML/Homework/hw1/public/test_mnist.csv'
output_file = './output.csv'

def find_k( X, y, k_range, method = 1):
    """ Use method to find the best k value
    It also shows the error rate in the range

    Parameters:
    ----------
    X : all of the training data
    y : all of the training label
    k_range : range to find the best k value
    method : int type. The method used for finding the best k value.
            Accept 1 or 2
            1: train_test_split()
            2: cross_val_score()

    Returns: 
    -------
    k_best : the best k value that has lowest error rate.
    """
    k_error = []
    k_best = 0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        if method != 1 and method != 2:
            method = 1
        # method=1 use train_test_split() to find the best k value
        if method == 1:
            print("using train_test_split() to compute when k = " + str(k))
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42, stratify=y)
            knn.fit(X_train, y_train)
            scores = knn.score(X_test, y_test)
        # method=2 use cross_val_score() to find the best k value
        elif method == 2:
            print("using cross_val_score() to compute when k = " + str(k))
            scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()
        k_error.append(1-scores)
    for i in range(len(k_error)):
        if k_error[i] == min(k_error):
            k_best = i+1

    print("Best k value found is " + str(k_best))
    plt.plot(k_range, k_error)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Error')
    plt.show()

def prediction(data, k_value, X_fit, y_fit):
    """Funtion for prediction

    Parameters:
    -----------
    data : data need to be predicted
    k_value : an appropriate k value for knn
    X_fit : training dataset for fitting the knn
    y_fit : training label for fitting the knn
    """
    # new a k neighbors classifier with the best k value
    knn = KNeighborsClassifier(n_neighbors = k_value)
    knn.fit(X_fit,y_fit)
    # prediction
    results = knn.predict(data)
    # put result into a np array [imgIndex, label]
    length = len(results)
    result = np.empty([length,2], dtype=int)
    for i in range(len(results)):
        result[i][0] = i+1
        result[i][1] = results[i]
    # write result into ouput file
    np.savetxt(output_file,result, fmt='%d', delimiter=',', header='ImageId,Label')
    print('Prediction has been done')
    print('csv file is saved as ' + output_file)


# sample of show img
tools.show_img(train_file,415,True)
#for i in range(2,60):
#    tools.show_img(test_file,i,False)

"""
for testing:
It is recommended to reduce the amount of input data
It is also recommended to test training and prediction separately
"""


X, y = tools.load_train_data(train_file, 42001)
# fit a scaler and scale the values
scaler = StandardScaler().fit(X)
sc_X = scaler.transform(X)
# find the best k value
k_range = range(1, 20)
#k_value = find_k(sc_X, y, k_range, 2)

# The k value can be calculated from the above
# but for convenience, I use the corresponding value directly
k_value = 3
test_X = tools.load_test_data(test_file, 10001)
sc_test_X = scaler.transform(test_X)
prediction(sc_test_X, k_value, sc_X, y)
