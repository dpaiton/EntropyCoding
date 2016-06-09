import numpy as np
import matplotlib.pyplot as plt
import IPython

"""
Display input data as an image with reshaping
Outputs:
  fig_no: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data : np.ndarray of shape (height, width) or (n, height, width) 
  title: string for title of figure
  prev_fig: tuple containing (fig_no, sub_axis, axis_image) from previous display_data() call
  save_filename: string holding output directory for writing, figures will not display with GUI if set
  TODO: Allow for color weight vis
"""
def display_data_tiled(data, title='', prev_fig=None, save_filename=""):
  # normalize input & remove extra dims
  data = (data / np.max(np.abs(data))).squeeze()

  if len(data.shape) >= 3:
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
      (1, 1), (1, 1))                       # add some space between filters
      + ((0, 0),) * (data.ndim - 3))        # don't pad the last dimension (if there is one)

    data = np.pad(data, padding, mode='constant', constant_values=1)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + \
      tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

  if prev_fig is None:
    fig_no, sub_axis = plt.subplots(1)
    axis_image = sub_axis.imshow(data, cmap='Greys', interpolation='nearest')
    axis_image.set_clim(vmin=-1.0, vmax=1.0)
    cbar = fig_no.colorbar(axis_image)
  else:
    (fig_no, sub_axis, axis_image) = prev_fig
    axis_image.set_data(data)

  fig_no.suptitle(title)
  
  if save_filename == "":
    if prev_fig is None:
      fig_no.show()
    else:
      fig_no.canvas.draw()
  else:
    fig_no.savefig(save_filename, bbox_inches='tight')

  return (fig_no, sub_axis, axis_image)

"""
Display input data as an image without reshaping
Outputs:
  fig_no: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data : np.ndarray of shape (height, width) or (n, height, width) 
  title: string for title of figure
  prev_fig: tuple containing (fig_no, sub_axis, axis_image) from previous display_data() call
  save_filename: string holding output directory for writing, figures will not display with GUI if set
"""
def display_data(data, title='', prev_fig=None, save_filename=""):
  if prev_fig is None:
    fig_no, sub_axis = plt.subplots(1)
    axis_image = sub_axis.imshow(data, cmap='Greys', interpolation='nearest')
    cbar = fig_no.colorbar(axis_image)
  else:
    (fig_no, sub_axis, axis_image) = prev_fig
    axis_image.set_data(data)
    axis_image.autoscale()

  fig_no.suptitle(title)

  if save_filename == "":
    if prev_fig is None:
      fig_no.show()
    else:
      fig_no.canvas.draw()
  else:
    fig_no.saverfig(save_filename, bbox_inches='tight')

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
  im_shp = img.shape[0]
  norm = np.vstack([(img[idx,:]-np.mean(img[idx,:]))/np.std(img[idx,:]) for idx in range(im_shp)])
  return norm
