import os
import cv2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


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


def decode(lpn, dic_cn, dic_en):
    lpn = lpn.split('_')
    cn = dic_cn[int(lpn[0])]
    en = ''
    for i in lpn[1:]:
        en += dic_en[int(i)]
    return cn + en


if __name__ == '__main__':
    file = 'Seg/'
    dic_en = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
              5: 'F', 6: 'G', 7: 'H', 8: 'J', 9: 'K',
              10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q',
              15: 'R', 16: 'S', 17: 'T', 18: 'U', 19: 'V',
              20: 'W', 21: 'X', 22: 'Y', 23: 'Z',
              24: '0', 25: '1', 26: '2', 27: '3', 28: '4',
              29: '5', 30: '6', 31: '7', 32: '8', 33: '9'}
    dic_cn = {0: '皖', 1: '沪', 2: '津', 3: '渝', 4: '翼', 5: '晋',
              6: '蒙', 7: '辽', 8: '吉', 9: '黑', 10: '苏',
              11: '浙', 12: '京', 13: '闽', 14: '赣', 15: '鲁',
              16: '豫', 17: '鄂', 18: '湘', 19: '粤', 20: '桂',
              21: '琼', 22: '川', 23: '贵', 24: '云', 25: '藏',
              26: '陕', 27: '甘', 28: '青', 29: '', 30: '新'}

    knn_cn = KNeighborsClassifier(n_neighbors=1)
    knn_en = KNeighborsClassifier(n_neighbors=6)
    df_cn = pd.read_csv('cn.csv')
    df_en = pd.read_csv('en.csv')
    knn_cn.fit(df_cn.iloc[:, 1:], df_cn.iloc[:, 0])
    knn_en.fit(df_en.iloc[:, 1:], df_en.iloc[:, 0])

    number = 0
    max = 1000
    miss = 0
    hit = 0

    for name in os.listdir(file):
        number += 1
        if number == max:
            break
        print(number, ':', max)
        res = ''

        data = []
        for word in os.listdir(file + name + '/'):
            data.append(read_data(file + name + '/' + word))

        res += dic_cn[knn_cn.predict(data[0])[0]]
        for i in data[1:]:
            res += dic_en[knn_en.predict(i)[0]]

        ans = decode(name.split('-')[4], dic_cn, dic_en)

        if res != ans:
            print(f'res:{res} : ans:{ans} : N\n')
            miss += 1
        else:
            print(f'res:{res} : ans:{ans} : Y\n')
            hit += 1

    print(f'Total number: {number}')
    print(f'Total hit: {hit}')
    print(f'Accuracy: {round(hit / number, 2)}')