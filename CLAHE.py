def clahe(image):
    '''
    Applys the OpenCV's CLAHE on the input image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The result image.
    '''

    out_image = image.copy()

    ####### your code ########

    clahe = cv2.createCLAHE(clipLimit=40,tileGridSize=(5,5))
    out_image = clahe.apply(out_image)


    ##########################

    return out_image

