import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython

"""
Display input data as an image
Outputs:
    fig_no: index for figure call
    sub_axis: index for subplot call
    axis_image: index for imshow call
Inpus:
    data : np.ndarray of shape (height, width) or (n, height, width) or (n, height, width, 3)
    title: string for title of figure
    prev_fig: tuple containing (fig_no, sub_axis, axis_image) from previous display_data() call
"""
def display_data(data, title='', prev_fig=None):
    # normalize input
    data = (data - data.min()) / (data.max() - data.min())

    if len(data.shape) >= 3: #TODO: Allow for condition where data is (height, width, 3)
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                   (1, 1), (1, 1))                 # add some space between filters
                   + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)

        data = np.pad(data, padding, mode='constant', constant_values=1)

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    if prev_fig is None:
        fig_no, sub_axis = plt.subplots(1)
        axis_image = sub_axis.imshow(data, cmap='Greys', interpolation='nearest')
    else:
        (fig_no, sub_axis, axis_image) = prev_fig
        axis_image.set_data(data)

    for axis in fig_no.axes:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)

    fig_no.suptitle(title)
    if prev_fig is None:
        fig_no.show()
    else:
        fig_no.canvas.draw()

    return (fig_no, sub_axis, axis_image)

""" 
Normalize input image to have mean 0 and std 1
Outputs:
    normalized image
Inputs:
    img : numpy ndarray
"""
def normalize_image(img):
    img_l2 = np.sqrt(np.sum(np.power(img, 2.0)))
    return np.vstack([(img[idx,:]-np.mean(img[idx,:]))/np.std(img[idx,:]) for idx in range(img.shape[0])]) / img_l2


""" 
Independently normalize the columns of matrix to be bounded by bound

Outputs:
    column normalized matrix

Inputs:
    matrix : tensorflow tensor
    bound  : flaot32
"""
def normalize_cols(matrix, bound):
    num_rows = matrix.get_shape()[0]
    matrix_eval = matrix.eval()
    max_mat = np.array([np.max(matrix_eval, axis=0),]*num_rows)
    min_mat = np.array([np.min(matrix_eval, axis=0),]*num_rows)
    sub_mat = tf.sub(matrix, tf.constant(min_mat))
    norm_mat = tf.truediv(sub_mat, tf.constant(max_mat - min_mat))
    bound_norm_mat = tf.mul(norm_mat, tf.constant(np.float32(bound)))
    #nump_matrix = np.random.randn(12,5)
    #nump_num_rows = nump_matrix.shape[0]
    #nump_max_mat = np.array([np.max(nump_matrix, axis=0),]*nump_num_rows)
    #nump_min_mat = np.array([np.min(nump_matrix, axis=0),]*nump_num_rows)
    #nump_sub_mat = nump_matrix - nump_min_mat
    #nump_norm_mat = nump_sub_mat / (nump_max_mat - nump_min_mat)
    #nump_bound_norm_mat = norm_mat * bound
    #IPython.embed()
    return bound_norm_mat

""" 
Independently normalize the rows of matrix to be bounded by bound

Outputs:
    column normalized matrix

Inputs:
    matrix : tensorflow tensor
    bound  : flaot32
"""
def normalize_rows(matrix, bound):
    num_cols = matrix.get_shape()[1]
    matrix_eval = matrix.eval()
    max_mat = np.array([np.max(matrix_eval, axis=1),]*num_cols).T
    min_mat = np.array([np.min(matrix_eval, axis=1),]*num_cols).T
    sub_mat = tf.sub(matrix, tf.constant(min_mat))
    norm_mat = tf.truediv(sub_mat, tf.constant(max_mat - min_mat))
    bound_norm_mat = tf.mul(norm_mat, tf.constant(bound))
    #nump_matrix = np.random.randn(12,5)
    #nump_num_cols = nump_matrix.shape[1]
    #nump_max_mat = np.array([np.max(nump_matrix, axis=1),]*nump_num_cols).T
    #nump_min_mat = np.array([np.min(nump_matrix, axis=1),]*nump_num_cols).T
    #nump_sub_mat = nump_matrix - nump_min_mat
    #nump_norm_mat = nump_sub_mat / (nump_max_mat - nump_min_mat)
    #nump_bound_norm_mat = bound * nump_norm_mat
    return bound_norm_mat
