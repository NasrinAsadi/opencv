import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math


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
        if img_list[i][2] == 'img':
            if gray:
                plt.imshow(img_list[i][0], cmap = 'gray')
            else:
                plt.imshow(img_list[i][0])
            plt.xticks([])
            plt.yticks([])
        elif img_list[i][2] == 'hist':
            plt.bar(np.arange(len(img_list[i][0])), img_list[i][0], color = 'c')
        else:
            raise Exception("Only image or histogram. Use third parameter of tuples in img_list and set it to img or hist.")
        plt.title(img_list[i][1])
    if fig_name is not None:
        plt.savefig(fig_name + '.png')
    plt.show()


def guassian_func(std, x, y, ee, pii):
  s = ee**((-(x**2)+ y**2)/2* std**2)
  t = 1/(2*pii)*s
  return t

def gaussian_filter(size, std):
    '''
    Creates the Guassian kernel with given size and std.

    Parameters:
        size (int): The size of the kernel. It must be odd.
        std (float): The standard deviation of the kernel.

    Returns:
        numpy.ndarray: The Guassina kernel.
    '''
    ee= math.e
    pii=math.pi
    kernel = np.zeros((size,size), np.float32)
    sum=0
    ####### your code ########
    for i in range(size):
      for j in range(size):

        kernel[i,j]= guassian_func(std, i-((size-1)/2), j-((size-1)/2), ee,pii)
        sum = sum+ kernel[i,j]

    kernel = (1.0 / sum)*kernel

    ##########################

    return kernel

def opencv_filter(img, size, std):
    '''
    Applys the OpenCV's guassian blur function on input image.

    Parameters:
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The result image.
    '''

    out = None
    kernel = np.zeros((size,size), np.float32)
    ####### your code ########
    #dst = cv2.GaussianBlur(src,(10,10),cv2.BORDER_DEFAULT)
    out = cv2.GaussianBlur(img, (size,size), std)

    ##########################

    return out

