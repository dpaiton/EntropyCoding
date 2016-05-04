import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython

"""
Display input data as an image with reshaping
Outputs:
    fig_no: index for figure call
    sub_axis: index for subplot call
    axis_image: index for imshow call
Inpus:
    data : np.ndarray of shape (height, width) or (n, height, width) or (n, height, width, 3)
    title: string for title of figure
    prev_fig: tuple containing (fig_no, sub_axis, axis_image) from previous display_data() call
"""
def display_data_tiled(data, title='', prev_fig=None):
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
        cbar = fig_no.colorbar(axis_image)
    else:
        (fig_no, sub_axis, axis_image) = prev_fig
        axis_image.set_data(data)
        axis_image.autoscale()

    #for axis in fig_no.axes:
    #    axis.get_yaxis().set_visible(False)
    #    axis.get_xaxis().set_visible(False)

    fig_no.suptitle(title)
    if prev_fig is None:
        fig_no.show()
    else:
        fig_no.canvas.draw()
        
    return (fig_no, sub_axis, axis_image)

"""
Display input data as an image without reshaping
"""
def display_data(data, title='', prev_fig=None):
    if prev_fig is None:
        fig_no, sub_axis = plt.subplots(1)
        axis_image = sub_axis.imshow(data, cmap='Greys', interpolation='nearest')
        cbar = fig_no.colorbar(axis_image)
    else:
        (fig_no, sub_axis, axis_image) = prev_fig
        axis_image.set_data(data)
        axis_image.autoscale()

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
    img : numpy ndarray - assumes 2D array with first dimension indicating batch dimension.
          normalization is done per image, not across the batch
"""
def normalize_image(img):
    return np.vstack([(img[idx,:]-np.mean(img[idx,:]))/np.std(img[idx,:]) for idx in range(img.shape[0])])

#""" 
#Independently normalize the columns of tensor to be bounded by bound
#
#Outputs:
#    bound_norm_mat : column normalized tensor
#
#Inputs:
#    tensor : 2-D tensorflow tensor
#    bound  : flaot32
#"""
#def normalize_cols(tensor, bound):
#    num_rows = int(tensor.get_shape()[0])
#    ones_mat = tf.constant(np.ones([num_rows, 1], dtype=np.float32))
#    max_mat = tf.matmul(ones_mat, tf.expand_dims(tf.reduce_max(tensor, reduction_indices=0), 0))
#    min_mat = tf.matmul(ones_mat, tf.expand_dims(tf.reduce_min(tensor, reduction_indices=0), 0))
#    norm_mat = tf.truediv(tf.sub(tensor, min_mat), tf.sub(max_mat, min_mat))
#    bound_norm_mat = tf.mul(norm_mat, tf.constant(np.float32(bound)))
#    return bound_norm_mat
#
#""" 
#Independently normalize the rows of tensor to be bounded by bound
#
#Outputs:
#    bound_norm_mat : row normalized tensor
#
#Inputs:
#    tensor : 2-D tensorflow tensor
#    bound  : flaot32
#"""
#def normalize_rows(tensor, bound):
#    num_cols = int(tensor.get_shape()[1])
#    ones_mat = tf.constant(np.ones([1, num_cols], dtype=np.float32))
#    max_mat = tf.matmul(tf.expand_dims(tf.reduce_max(tensor, reduction_indices=1), 1), ones_mat)
#    min_mat = tf.matmul(tf.expand_dims(tf.reduce_min(tensor, reduction_indices=1), 1), ones_mat)
#    norm_mat = tf.truediv(tf.sub(tensor, min_mat), tf.sub(max_mat, min_mat))
#    bound_norm_mat = tf.mul(norm_mat, tf.constant(np.float32(bound)))
#    return bound_norm_mat
#
#""" 
#Normalize the columns of tensor to unit norm
#For tensor W of dimensions (m,n),
#set the l2 norm of each of the n columns to bound
#||W_n||_2 = bound for all n
#
#Outputs:
#    column normalized tensor
#
#Inputs:
#    tensor : 2-D tensorflow tensor
#    bound  : (int) scalar for normalization procedure
#"""
#def l2_normalize_cols(tensor, bound=1.0):
#    return tf.matmul(tensor, tf.diag(bound / tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2.0), reduction_indices=0))))
#
#""" 
#Normalize the rows of tensor to unit norm
#For tensor W of diminsions (m, n),
#set the l2 norm of each of the m rows to bound
#||W_m||_2 = bound for all m
#
#Outputs:
#    row normalized tensor
#
#Inputs:
#    tensor : 2-D tensorflow tensor
#    bound  : (int) scalar for normalization procedure
#"""
#def l2_normalize_rows(tensor, bound=1.0):
#    return tf.matmul(tf.diag(bound / tf.sqrt(tf.reduce_sum(tf.pow(tensor, 2.0), reduction_indices=1))), tensor)
#
#""" 
#Normalize input tensor to be between 0 and bound
#Outputs:
#    normalized tensor
#Inputs:
#    tensor : tensorflow tensor
#"""
#def normalize_tensor(tensor, bound):
#    tensor_max = tf.reduce_max(tensor)
#    tensor_min = tf.reduce_min(tensor)
#    norm_tensor = tf.truediv(tf.sub(tensor, tensor_min), tf.sub(tensor_max, tensor_min))
#    return tf.mul(norm_tensor, tf.constant(np.float32(bound)))
