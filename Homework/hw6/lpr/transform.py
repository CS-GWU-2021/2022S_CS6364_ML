import pandas as pd
import os
import cv2


# 28*28 write into csv
# resize to 28*28


def resize(_filepath):
    num = 0
    for _words in os.listdir(_filepath):
        print(num, ':', len(os.listdir(_filepath)))
        num += 1
        for _word in os.listdir(_filepath + _words):
            _file = f'{_filepath}{_words}/{_word}'
            _img = cv2.imread(_file)
            _img = cv2.resize(_img, (28, 28))
            cv2.imwrite(_file, _img)


def copy_by_name(_names, _src, _dis):
    for _name in os.listdir(_names):
        img = cv2.imread(_src + _name)
        cv2.imwrite(_dis + _name, img)


def transform_csv(_filepath):
    file_ch = 'cn_raw.csv'
    file_en_num = 'en_raw.csv'
    cn = []
    en_num = []
    col_name = ['label']
    for i in range(784):
        col_name.append(f'pixel{i}')

    num = 0
    for words in os.listdir(_filepath):
        print(num, ':', len(os.listdir(_filepath)))
        num += 1
        # if num == 5000:
        #     break
        for word in os.listdir(_filepath + words):
            file = f'{_filepath}{words}/{word}'
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            if img.shape != (28, 28):
                print(img.shape)
                continue
            try:
                data = [int(word[2:-4])]
            except:
                continue
            for i in range(28):
                for j in range(28):
                    if img[i][j] == 255:
                        data.append(1)
                    else:
                        data.append(img[i][j])
            if word[0] == '0':
                cn.append(data)
            else:
                en_num.append(data)

    df_ch = pd.DataFrame(cn, columns=col_name)
    df_ch.to_csv(file_ch, index=False)

    df_en_num = pd.DataFrame(en_num, columns=col_name)
    df_en_num.to_csv(file_en_num, index=False)


if __name__ == '__main__':
    filepath = 'Test/'
    resize(filepath)

    name = 'Train/'
    src = '../public/CCPD2019/ccpd_base/'
    dis = 'Src/'
    # copy_by_name(name, src, dis)
    # transform_csv(filepath)
