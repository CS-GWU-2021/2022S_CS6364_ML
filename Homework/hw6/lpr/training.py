from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2


def find_k(_file, k_range):
    """Find the best k value
    It also shows the error rate in the range

    Parameters:
    ----------
    X : all the training data
    y : all the training label
    k_range : range to find the best k value

    Returns:
    -------
    k_best : the best k value that has lowest error rate.
    """
    k_error = []
    k_best = 0
    df = pd.read_csv(_file)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        # use train_test_split() to find the best k value
        print("using train_test_split() to compute when k = " + str(k))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        knn.fit(X_train, y_train)
        scores = knn.score(X_test, y_test)
        k_error.append(1 - scores)
    for i in range(len(k_error)):
        if k_error[i] == min(k_error):
            k_best = i + 1
    print("Best k value found is " + str(k_best))
    plt.plot(k_range, k_error)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Error')
    plt.show()
    return k_best


def read_data(_file):
    img = cv2.imread(_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    col_name = []
    for i in range(784):
        col_name.append(f'pixel{i}')

    _data = []
    for i in range(28):
        for j in range(28):
            if img[i][j] == 255:
                _data.append(1)
            else:
                _data.append(img[i][j])
    _data = pd.DataFrame([_data], columns=col_name)
    return _data


def test(k, train_file, test_file):
    _df = pd.read_csv(train_file)
    _X = _df.iloc[:, 1:]
    _y = _df.iloc[:, 0]
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(_X, _y)
    for file in os.listdir(test_file):
        data = read_data(test_file + file)
        res = knn.predict(data)
        print(file[2:-4], res)


def mean_accuracy(_path, k):
    df = pd.read_csv(_path)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    knn_cn = KNeighborsClassifier(n_neighbors=k)
    print(f'mean of KNN for {_path[:2]}: {cross_val_score(knn_cn, X, y, cv=10).mean()}')


if __name__ == '__main__':
    train_file_en = 'en.csv'
    train_file_cn = 'cn.csv'
    test_file_cn = 'Test/cn/'
    test_file_en = 'Test/en/'

    # k_value = find_k(train_file_cn, range(1, 20, 2))
    # k_value = find_k(train_file_en, range(1, 20, 2))

    # test for cn
    test(1, train_file_cn, test_file_cn)
    # test for en
    test(5, train_file_en, test_file_en)

    # mean for cn
    # mean_accuracy(train_file_cn, 1)
    # mean for en
    # mean_accuracy(train_file_en, 6)




