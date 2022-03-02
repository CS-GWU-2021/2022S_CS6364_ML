# --------------------------------------------------------------------------------
# Author:       Tenphun0503
# Intro:        Main code; Logistic regression
# --------------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tools

# files direction
train_file = 'D:/2022 Spring/2022S_CS6364_ML/Homework/hw2/public/data_mnist.csv'
test_file = 'D:/2022 Spring/2022S_CS6364_ML/Homework/hw2/public/test_mnist.csv'
output_file = './output.csv'
sample_file = './sample.csv'

def write_result(output_file, results):
    """This function writes the array of result into a .csv file in a specific format
    Format : [imgIndex, label]
    """
    length = len(results)
    result = np.empty([length,2], dtype=int)
    for i in range(len(results)):
        result[i][0] = i+1
        result[i][1] = results[i]
    # write result into ouput_file
    np.savetxt(output_file, result, fmt='%d', delimiter=',', header='ImageId,Label')   
    print('csv file is saved as ' + output_file)

def inspect(output_file, sample_file):
    """This function compares two specific format .csv file and return the correct rate
    Format : [imgIndex, label]
    """
    output = np.loadtxt(output_file, dtype = int, delimiter=',', skiprows=1)
    output = output.flatten()
    sample = np.loadtxt(sample_file, dtype = int, delimiter=',', skiprows=1)
    sample = sample.flatten()
    diff = 0
    sum = len(output)
    for i in range(1, sum, 2):
        if output[i] != sample[i]:
            diff += 1
    error = float(diff)/float(sum)
    return 1 - error


X, y = tools.load_train_data(train_file, 42001)
print("The training data is loaded")
# fit a scaler and scale the values
scaler = StandardScaler().fit(X)
sc_X_train = scaler.transform(X)

'''
# split the dataset 
X_train, X_test, y_train, y_test = train_test_split(sc_X_train, y, test_size=0.20, random_state=0)
clf = LogisticRegression(random_state=0, solver='sag',multi_class='multinomial').fit(X_train,y_train)
# compute the accuracy 0.9208333
acc = clf.score(X_test,y_test)
print(acc)
'''

# write the answer and test the accuracy 0.9644
clf = LogisticRegression(random_state=0, solver='sag',multi_class='multinomial').fit(sc_X_train,y)
X_test2 = tools.load_test_data(test_file, 10001)
sc_X_test2 = scaler.transform(X_test2)
y_result = clf.predict(sc_X_test2)
write_result(output_file, y_result)
print(inspect(output_file, sample_file))

