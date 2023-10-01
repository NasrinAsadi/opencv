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

def draw_circles(width, height, radius, color, thickness):
    '''
    Draws nested circles.

    Parameters:
        width (int): The width of result image.
        height (int): The height of result image.
        radius (int): The radius of circles.
        color (tuple): The color of circles in (b, g, r) mode.
        thickness (int): The thickness of circles.

    Returns:
        numpy.ndarray: The result image.
    '''

    image = np.zeros((height, width, 3), np.uint8) + 255 # creates blank image.

    a=width/2
    b=height/2
    r=radius
    for t in range(0,360,30):
      x=r*np.sin(t)+a;
      y=r*np.cos(t)+b;
     # print(x,y)
      center = (int(x),int(y))
      image= cv2.circle (image,center,radius,color, thickness)
      # Write your code here

    return image

width = 200
height = 100
radius = 20
color = (0, 0, 255) # blue color in (r, g, b) mode

image_list = []
thickness = 1
image_list.append([draw_circles(width, height, radius, color, thickness), 'circles_t1'])
thickness = 2
image_list.append([draw_circles(width, height, radius, color, thickness), 'circles_t2'])
plotter(image_list, 1, 2, False, 20, 10, 'q6a')


def draw_lines(width, height, color, thickness):
    '''
    Draws nested lines.

    Parameters:
        width (int): The width of result image.
        height (int): The height of result image.
        color (tuple): The color of lines in (b, g, r) mode.
        thickness (int): The thickness of lines.

    Returns:
        numpy.ndarray: The result image.
    '''

    image = np.zeros((height, width, 3), np.uint8) + 255 # creates blank image.
    # Write your code here
    for i in range(0,500, 5):

        x1=(0,i)
        x2=(500-i,0)
        cv2.line(image, x1 , x2, color, thickness)
    return image

width = 500
height = 500
thickness = 1
color = (0, 0, 255) # blue color in (r, g, b) mode

image_list = []
image_list.append([draw_lines(width, height, color, thickness), 'lines'])
plotter(image_list, 1, 1, False, 20, 10, 'q6b')