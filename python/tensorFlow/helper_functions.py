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
