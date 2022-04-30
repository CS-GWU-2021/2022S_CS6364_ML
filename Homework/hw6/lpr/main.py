import shutil
import cv2
import numpy as np
import os
import random
import math
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class MyImage:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        # area ratio of license plate are to the entire picture area
        self.area_ratio = None
        # horizontal tilt degree and vertical tilt degree
        self.tilt_degree = None
        # coordinates of the left-up and the right-bottom of bounding box
        self.b_box = []
        # four vertices of LP, start from right-bottom
        self.vertices = []
        # license plate number
        self.lpn = None
        # brightness
        self.bright = None
        # blurriness
        self.blur = None
        self.__name_decode()

        # read license plate area read with cv2
        bb = self.b_box
        self.img_data = cv2.imread(self.path + self.name)
        self.area = self.img_data[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]]

        # 7 characters
        self.words = None

    def __name_decode(self):
        """This function decodes the name of the img"""
        name = self.name[:-4].split('-')
        # area ratio
        self.area_ratio = name[0]
        # horizontal tilt degree and vertical tilt degree
        tilt_degree = name[1].split('_')
        self.tilt_degree = [int(tilt_degree[0]), int(tilt_degree[1])]
        # coordinates of bounding box
        bounding_box = name[2].split('_')
        for v in bounding_box:
            temp = v.split('&')
            self.b_box.append((int(temp[0]), int(temp[1])))
        # four vertices
        vertices = name[3].split('_')
        for v in vertices:
            temp = v.split('&')
            self.vertices.append((int(temp[0]), int(temp[1])))
        # license plate number
        self.lpn = name[4]
        # brightness
        self.bright = int(name[5])
        # blurriness
        self.blur = int(name[6])

    def print_info(self):
        """This function prints out the info of the img"""
        print(f'Img Name:\t\t{self.name}')
        print(f'Area ratio:\t\t{self.area_ratio}')
        print(f'Horizontal TD:\t{self.tilt_degree[0]}')
        print(f'Vertical TD:\t{self.tilt_degree[1]}')
        print(f'Bound box:\t\t{self.b_box}')
        print(f'Vertices:\t\t{self.vertices}')
        print(f'LPN\t\t\t\t{self.lpn}')
        print(f'Brightness\t\t{self.bright}')
        print(f'Blurriness\t\t{self.blur}')

    def show_img(self):
        """This function draws and shows the bounding box and the vertices of the img"""
        bb = self.b_box
        box_v = [bb[0], (bb[1][0], bb[0][1]), bb[1], (bb[0][0], bb[1][1])]
        cv2.line(self.img_data, box_v[0], box_v[1], (255, 0, 0), thickness=3)
        cv2.line(self.img_data, box_v[1], box_v[2], (255, 0, 0), thickness=3)
        cv2.line(self.img_data, box_v[2], box_v[3], (255, 0, 0), thickness=3)
        cv2.line(self.img_data, box_v[3], box_v[0], (255, 0, 0), thickness=3)
        for c in self.vertices:
            cv2.circle(self.img_data, c, 1, (0, 0, 255), thickness=3)
        cv2.imshow('img', self.img_data)

    def preprocess(self):
        temp = self.area
        # Gray
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        # Two-bit
        ret, temp = cv2.threshold(temp, 0, 255, cv2.THRESH_OTSU)
        # Tilting
        temp = self.tilting(temp)
        # resize
        temp = cv2.resize(temp, (315, 100))
        self.area = temp
        self.words = self.separate()

    def tilting(self, image):
        h, w = image.shape
        center = (h // 2, w // 2)
        wrap_mat = cv2.getRotationMatrix2D(center, self.tilt_degree[0] - 90, 1.0)
        image = cv2.warpAffine(image, wrap_mat, (w, h))
        # image = self.vertical_tilting(image)
        return image

    # TODO: still have some problems
    def vertical_tilting(self, image):
        h, w = image.shape
        angle = self.tilt_degree[1] - 90
        center = (h // 2, w // 2)
        dst = np.zeros((h, w))
        for y in range(center[0]):
            delta = int(np.floor(math.tan(angle) * (center[0] - y)))
            for x in range(w):
                if 0 < x + delta < w:
                    dst[y][x] = image[y][x + delta]

        for y in range(center[0], h):
            delta = int(np.floor(math.tan(angle) * (y - center[0])))
            for x in range(w):
                if w > x - delta > 0:
                    dst[y][x] = image[y][x - delta]
        return dst

    def separate(self):
        h, w = self.area.shape
        sum_rows = []
        sum_cols = []
        sum = 0
        for i in range(h):
            sum_row = 0
            for j in range(w):
                if self.area[i][j] == 255:
                    sum_row += 1
            sum_rows.append(sum_row)
            sum += sum_row

        avg_row = sum / h
        avg_row = avg_row * 0.8

        # find the max spectrum
        space = []
        max_dis = 0
        start, end = 0, 1
        a = 1
        while a < h - 2:
            a += 1
            if sum_rows[a] > avg_row:
                start = a
                for m in range(start + 1, h - 1):
                    if sum_rows[m] < avg_row:
                        end = m
                        break
                    else:
                        end = h - 2
                if start > end:
                    break
                d = end - start
                a = end
                max_dis = max(max_dis, d)
                space.append([d, start, end])
        cut_y0 = 0
        cut_y1 = 0
        for x in space:
            if x[0] == max_dis:
                cut_y0 = x[1]
                cut_y1 = x[2]
        self.area = self.area[cut_y0:cut_y1 + 5, :]
        h, w = self.area.shape

        sum = 0
        # vertical
        for i in range(w):
            sum_col = 0  # sum of 255 in vertical
            for j in range(h):
                if self.area[j][i] == 255:
                    sum_col += 1
            sum_cols.append(sum_col)
            sum += sum_col
        avg_col = sum / w

        # horizontal
        sum_row = []
        for i in range(h):
            sum_row = 0  # sum of 255 in horizon
            for j in range(w):
                if self.area[i][j] == 255:
                    sum_row += 1
            sum_rows.append(sum_row)

        avg_col = 0.2 * avg_col
        clip_imgs = []

        # 查找最大跨度
        n = 1
        while n < w - 2:
            n += 1
            if sum_cols[n] > avg_col:
                start = n
                for m in range(start + 1, w - 1):
                    if sum_cols[m] < avg_col:
                        end = m
                        break
                d = end - start
                if d > 10:
                    if d < 20:
                        end += 30
                    clip_img = self.area[:, start - 5:end + 5]
                    clip_imgs.append(clip_img)
                if start > end:
                    break
                n = end

        resize_img = []
        for i in clip_imgs:
            try:
                i = cv2.resize(i, (28, 28))
                resize_img.append(i)
            except:
                AssertionError

        if len(resize_img) == 7:
            # resize and save
            print(f'{self.lpn} is separated')
            self.save_words(resize_img)
            return resize_img
        else:
            print(f'{self.lpn} seg error')
        return False

    def save_words(self, _imgs):
        filepath = f'Res/{self.name}/'
        if os.path.exists(filepath):
            shutil.rmtree(filepath)
        os.mkdir(filepath)
        lpn = self.lpn.split('_')
        for i, res in enumerate(_imgs):
            # save
            cv2.imshow(str(i), res)
            cv2.imwrite(f'{filepath}{i}_{lpn[i]}.jpg', res)
        print(f'{self.lpn} is saved')


class Recognizer:
    def __init__(self, images, _lpn):
        self.imgs = images
        self.dic_ln = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
                       5: 'F', 6: 'G', 7: 'H', 8: 'J', 9: 'K',
                       10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q',
                       15: 'R', 16: 'S', 17: 'T', 18: 'U', 19: 'V',
                       20: 'W', 21: 'X', 22: 'Y', 23: 'Z',
                       24: '0', 25: '1', 26: '2', 27: '3', 28: '4',
                       29: '5', 30: '6', 31: '7', 32: '8', 33: '9'}
        self.dic_cn = {0: '皖', 1: '沪', 2: '津', 3: '渝', 4: '翼', 5: '晋',
                       6: '蒙', 7: '辽', 8: '吉', 9: '黑', 10: '苏',
                       11: '浙', 12: '京', 13: '闽', 14: '赣', 15: '鲁',
                       16: '豫', 17: '鄂', 18: '湘', 19: '粤', 20: '桂',
                       21: '琼', 22: '川', 23: '贵', 24: '云', 25: '藏',
                       26: '陕', 27: '甘', 28: '青', 29: '', 30: '新'}

        self.ans = self.decode(_lpn)
        self.res = ''

    def read_data(self, _img):
        ret, _img = cv2.threshold(_img, 0, 255, cv2.THRESH_OTSU)
        col_name = []
        for i in range(784):
            col_name.append(f'pixel{i}')
        _data = []
        for i in range(28):
            for j in range(28):
                if _img[i][j] == 255:
                    _data.append(1)
                else:
                    _data.append(_img[i][j])
        _data = pd.DataFrame([_data], columns=col_name)
        return _data

    def recognize(self, _knn_cn, _knn_en):
        print('Start Recognizing...')
        # Transform images into data
        cn = [self.read_data(self.imgs[0])]
        ens = []
        for _img in self.imgs[1:]:
            data = self.read_data(_img)
            ens.append(data)

        cn_res = _knn_cn.predict(cn[0])
        en_res = []
        for i in ens:
            en_res.append(_knn_en.predict(i))

        self.res += self.dic_cn[int(cn_res)]
        for i in en_res:
            self.res += self.dic_ln[int(i)]

    def decode(self, _lpn):
        _lpn = _lpn.split('_')
        cn = self.dic_cn[int(_lpn[0])]
        en = ''
        for i in _lpn[1:]:
            en += self.dic_ln[int(i)]
        return cn + en


def run_test(path):
    # traverse all images in the Src
    df = pd.read_csv('cn.csv')
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    knn_cn = KNeighborsClassifier(n_neighbors=1)
    knn_cn.fit(X, y)

    df = pd.read_csv('en.csv')
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    knn_en = KNeighborsClassifier(n_neighbors=5)
    knn_en.fit(X, y)

    seg_miss = 0
    rec_miss = 0
    rec_hit = 0
    number = 0
    sum = 5000

    for name in os.listdir(path):
        if number == sum:
            break
        number += 1
        print(f'Dealing with {number}/{len(os.listdir(path))}')
        img = MyImage(path, name)
        img.preprocess()
        if img.words:
            rc = Recognizer(img.words, img.lpn)
            rc.recognize(knn_cn, knn_en)
            if rc.res != rc.ans:
                print(f'res:{rc.res} : ans:{rc.ans} : N\n')
                rec_miss += 1
            else:
                print(f'res:{rc.res} : ans:{rc.ans} : Y\n')
                rec_hit += 1

        else:
            seg_miss += 1

    print(f'Total number: {number}')
    print(f'Total hit: {rec_hit}')
    print(f'Accuracy: {round(rec_hit / number, 2)}')
    print(f'Segmentation error: {seg_miss}\tRecognition error: {rec_miss}')


def random_run(_path):
    # random select one img in the Src
    random_one = random.randint(0, len(os.listdir(_path)))
    img_name = os.listdir(_path)[random_one]
    img = MyImage(_path, img_name)
    img.show_img()
    img.print_info()
    img.preprocess()
    cv2.waitKey(0)
    if img.words:
        df = pd.read_csv('cn.csv')
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        knn_cn = KNeighborsClassifier(n_neighbors=1)
        knn_cn.fit(X, y)

        df = pd.read_csv('en.csv')
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        knn_en = KNeighborsClassifier(n_neighbors=6)
        knn_en.fit(X, y)

        rc = Recognizer(img.words, img.lpn)
        print('The answer is ', rc.ans)
        rc.recognize(knn_cn, knn_en)
        print('Recognized ', rc.res)


if __name__ == '__main__':
    img_path = 'Src/'
    random_run(img_path)
    # run_test(img_path)
