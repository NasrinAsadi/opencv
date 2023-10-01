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



def eq_hist(h):
    sum_img=0
    sum_img = h.sum()
    print(sum_img)
    ####### your code ########
    out_image_hist = np.zeros((256), np.int32)
    out_image_hist[0]= h[0]
    L = 256

    for hi,hv in enumerate(h[1:]):
      out_image_hist[hi]= out_image_hist[hi-1]+hv

    out_image_hist =  (L-1)/sum_img * out_image_hist
    return out_image_hist


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



def histogram_equalization(image):
    '''
    Equalizes the histogram of the input image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The result image that it's histogram be eqaulized.
    '''

    h = compute_histogram(image)
    out_image = image.copy()

    out_image_hist= eq_hist(h)

    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        out_image[i,j] = out_image_hist[out_image[i,j]]
    ##########################

    return out_image

img = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)
out = histogram_equalization(img)
image_list = []
image_list.append([img, 'input_image'])
image_list.append([out, 'histogram_equalized'])
plotter(image_list, 1, 2, True, 20, 10, 'q3b-1')
h2 = compute_histogram(out)
show_histogram(h2, 'result_hist', 'q3b-2')
