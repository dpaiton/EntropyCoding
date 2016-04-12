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
    output : normalized image
Inputs:
    img : numpy ndarray
    divide_l2 : (default False), set to True if you wish to divide the normalized image by its l2 norm
"""
def normalize_image(img, divide_l2=False):
    if divide_l2:
        img_l2 = np.sqrt(np.sum(np.power(img, 2.0)))
        output = np.vstack([(img[idx,:]-np.mean(img[idx,:]))/np.std(img[idx,:]) for idx in range(img.shape[0])]) / img_l2
    else:
        output = np.vstack([(img[idx,:]-np.mean(img[idx,:]))/np.std(img[idx,:]) for idx in range(img.shape[0])])
    return output

""" 
Independently normalize the columns of matrix to be bounded by bound

Outputs:
    column normalized matrix

Inputs:
    matrix : tensorflow tensor
    bound  : flaot32
"""
def normalize_cols(matrix, bound):
    num_rows = int(matrix.get_shape()[0])
    ones_mat = tf.constant(np.ones([num_rows, 1], dtype=np.float32))
    max_mat = tf.matmul(ones_mat, tf.expand_dims(tf.reduce_max(matrix, reduction_indices=0), 0))
    min_mat = tf.matmul(ones_mat, tf.expand_dims(tf.reduce_min(matrix, reduction_indices=0), 0))
    norm_mat = tf.truediv(tf.sub(matrix, min_mat), tf.sub(max_mat,min_mat))
    bound_norm_mat = tf.mul(norm_mat, tf.constant(np.float32(bound)))
    return bound_norm_mat

""" 
Independently normalize the rows of matrix to be bounded by bound

Outputs:
    row normalized matrix

Inputs:
    matrix : tensorflow tensor
    bound  : flaot32
"""
def normalize_rows(matrix, bound):
    num_cols = int(matrix.get_shape()[1])
    ones_mat = tf.constant(np.ones([1, num_cols], dtype=np.float32))
    max_mat = tf.matmul(tf.expand_dims(tf.reduce_max(matrix, reduction_indices=1), 1), ones_mat)
    min_mat = tf.matmul(tf.expand_dims(tf.reduce_min(matrix, reduction_indices=1), 1), ones_mat)
    norm_mat = tf.truediv(tf.sub(matrix, min_mat), tf.sub(max_mat,min_mat))
    bound_norm_mat = tf.mul(norm_mat, tf.constant(np.float32(bound)))
    return bound_norm_mat

""" 
Normalize the columns of matrix to unit norm
For matrix W of dimensions (m,n),
set the l2 norm of each of the n columns to 1
||W_n||_2 = 1 for all n

Outputs:
    column normalized matrix

Inputs:
    matrix : tensorflow tensor
"""
def l2_normalize_cols(matrix):
    return tf.matmul(matrix, tf.diag(1.0/tf.sqrt(tf.reduce_sum(tf.pow(matrix, 2.0), reduction_indices=0))))

""" 
Normalize the rows of matrix to unit norm
For matrix W of diminsions (m, n),
set the l2 norm of each of the m rows to 1
||W_m||_2 = 1 for all m

Outputs:
    row normalized matrix

Inputs:
    matrix : tensorflow tensor
"""
def l2_normalize_rows(matrix):
    return tf.matmul(tf.diag(1.0/tf.sqrt(tf.reduce_sum(tf.pow(matrix, 2.0), reduction_indices=1))), matrix)
