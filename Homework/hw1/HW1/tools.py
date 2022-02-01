# --------------------------------------------------------------------------------
# Author:       Tenphun0503
# Intro:        Some functions such as show images and load dataset from csv file
# --------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#show image of specific line of a csv file
def show_img(file : str, line : int, if_label : bool):
    """ shows image of specified line

    Parameters:
    -----------
    file : the csv file direction, either data_mnist.csv or test_mnist.csv
    line : specified line, up to 42001 for data file and 10001 for test file
    if_label : to filter labels, True for data file and False for test file
    """
    # load data from csv as 'data'
    data = np.loadtxt(file, dtype = int, delimiter=',', skiprows=1, max_rows=line)

    # get the data of requested line
    line = line - 2     # -1 for skiprows, -1 for data starts from 0 
    img_data = data[line]

    # split the label and data if 'if_label' is true
    if(if_label):
        label = img_data[0]
        img_data = img_data[1:]

    # split data into 28x28 data array
    img = np.empty([28, 28], dtype=int)
    for i in range(0, 28):
        for j in range(0, 28):
            img[i][j] = img_data[i * 28 + j]

    # darw the img
    plt.imshow(img,interpolation='nearest',cmap='bone',origin='lower')
    plt.colorbar()
    plt.xticks(())
    plt.yticks(())
    plt.show()

def load_train_data(file : str, line : int):
    """load first (line:int) lines of training dataset and sperate the datas and labels

    Parameters:
    ----------
    file : the csv file direction
    line : the number of lines to output to

    Return:
    ----------
    data : (linex784) size np array contains datas
    target : (line) size np array contains labels
    """
    # load data from csv as 'raw_data'
    raw_data = np.loadtxt(file, dtype = int, delimiter=',', skiprows=1, max_rows=line)
    # number of data = line - 1(skip row)
    length = line - 1

    # label
    target = np.empty(length, dtype=int)
    # features of 784 Dimensions
    data = np.empty([length,784], dtype=int)
    # split the label and features
    for i in range(length):
        target[i] = raw_data[i][0]
        data[i] = raw_data[i][1:]
    return data, target

def load_test_data(file : str, line : int):
    """load first (line:int) lines of test dataset

    Parameters:
    ----------
    file : the csv file direction
    line : the number of lines to output to

    Return:
    ----------
    data : (linex784) size np array contains datas
    """
    # load data from csv as 'data'
    data = np.loadtxt(file, dtype = int, delimiter=',', skiprows=1, max_rows=line)
    # number of data = line - 1(skip row)
    length = line - 1
    return data