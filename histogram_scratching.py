def histogram_stretching(image):
    '''
    Streches the histogram of the input image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The result image that it's histogram be streched.
    '''

    eq_h = compute_histogram(image)
    out_image = image.copy()

    ####### your code ########
    nonzeros = np.nonzero(eq_h)[0]
    f_min = nonzeros[0]
    f_max = nonzeros[-1]
    Min=0
    Max=255

    alpha = ((Max-Min)/(f_max-f_min)) +Min
    out_image = np.round((out_image - f_min)*alpha)
    out_image=out_image.astype(int)

    return out_image

