from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import numpy as np

import tools, train

train_file = "D:\2022 Spring\2_6364 Machine Learning\Homework\hw1\public\data_mnist.csv"
test_file = "D:\2022 Spring\2_6364 Machine Learning\Homework\hw1\public\test_mnist.csv"
output_file = "D:\2022 Spring\2_6364 Machine Learning\Homework\hw1\public\output.csv"


#tools.show_img(train_file,55,True)
#tools.show_img(test_file,48,False)


X, y = tools.load_train_data(train_file, 40001)
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
    
#train.find_k(X,y)


test_data = tools.load_test_data(test_file, 101)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X,y)
results = knn.predict(test_data)
length = len(results)
result = np.empty([length,2], dtype=int)

for i in range(len(results)):
    result[i][0] = i+1
    result[i][1] = results[i]

np.savetxt(output_file,result, fmt='%d', delimiter=',', header='ImageId,Label')
