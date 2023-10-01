import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def plotter(img_list, r, w, gray, wr, hr, fig_name = None):
    '''
    Plots images' list with its' caption and saves result image if you want.

    Parameters:
        img_list (list): The list of tuples of image and its' caption.
        r (int): The number of row(s).
        w (int): The number of colunm(s).
        gray (bool): The flag for plotting images in grayscale mode.
        wr (int): The width of one figure.
        hr (int): The height of one figure.
        fig_name (str): The name of the image of the plot. if not set this parameter the plot doesn't save.
    '''

    plt.rcParams['figure.figsize'] = (wr, hr)
    for i in range(len(img_list)):
        plt.subplot(r, w, i + 1)
        if gray:
            plt.imshow(img_list[i][0], cmap = 'gray')
        else:
            plt.imshow(img_list[i][0])
        plt.title(img_list[i][1])
        plt.xticks([])
        plt.yticks([])
    if fig_name is not None:
        plt.savefig(fig_name + '.png')
    plt.show()


def show_histogram(histogram, title, fig_name):
    '''
    Plots histogram with it's caption and saves result image.

    Parameters:
        histogram (numpy.ndarray): The numpy array of numbers in histogram.
        title (str): The title of the plot.
        fig_name (str): The name of the image of the plot.
    '''

    plt.figure()
    plt.bar(np.arange(256),histogram,color = 'c')
    plt.title(title)
    plt.savefig(fig_name + '.png')
    plt.show()


def compute_histogram(image):
    '''
    Computes histogram of the input image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The numpy array of numbers in histogram.
    '''

    histogram = np.zeros((256), np.int32)

    ####### your code ########

    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        histogram[image[i,j]]= histogram[image[i,j]]+1


    ##########################

    return histogram

img = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)
#print(img)
h = compute_histogram(img)

show_histogram(h, 'hist', 'q3a')