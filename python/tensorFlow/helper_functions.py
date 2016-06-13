import numpy as np
import matplotlib.pyplot as plt
import IPython

"""
Normalize data 
Outputs:
  data normalized so that when plotted 0 will be midlevel grey
Inputs:
  data : np.ndarray
"""
def normalize_data(data):
  return (data / np.max(np.abs(data))).squeeze()

"""
Pad data with ones for visualization
Outputs:
  padded version of input
Inputs:
  data : np.ndarray
"""
def pad_data(data):
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = (((0, n ** 2 - data.shape[0]),
    (1, 1), (1, 1))                       # add some space between filters
    + ((0, 0),) * (data.ndim - 3))        # don't pad the last dimension (if there is one)
  padded_data = np.pad(data, padding, mode="constant", constant_values=1)
  # tile the filters into an image
  padded_data = padded_data.reshape((n, n) + padded_data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1)))
  padded_data = padded_data.reshape((n * padded_data.shape[1], n * padded_data.shape[3]) + padded_data.shape[4:])
  return padded_data

"""
Save figure for input data as a tiled image
Outputs:
  fig_no: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data : np.ndarray of shape (height, width) or (n, height, width) 
  title: string for title of figure
  save_filename: string holding output directory for writing, figures will not display with GUI if set
"""
def save_data_tiled(data, title="", save_filename=""):
  data = normalize_data(data)
  if len(data.shape) >= 3:
    data = pad_data(data)
  fig_no, sub_axis = plt.subplots(1)
  axis_image = sub_axis.imshow(data, cmap="Greys", interpolation="nearest")
  axis_image.set_clim(vmin=-1.0, vmax=1.0)
  cbar = fig_no.colorbar(axis_image)
  fig_no.suptitle(title, y=1.05)
  if save_filename == "":
    save_filename = "./output.ps"
  fig_no.savefig(save_filename, transparent=True, bbox_inches="tight", pad_inches=0.01)
  plt.close(fig_no)
  return 0

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
  TODO: Allow for color weight vis
"""
def display_data_tiled(data, title="", prev_fig=None):
  data = normalize_data(data)
  if len(data.shape) >= 3:
    data = pad_data(data)
  if prev_fig is None:
    fig_no, sub_axis = plt.subplots(1)
    axis_image = sub_axis.imshow(data, cmap="Greys", interpolation="nearest")
    axis_image.set_clim(vmin=-1.0, vmax=1.0)
    cbar = fig_no.colorbar(axis_image)
  else:
    (fig_no, sub_axis, axis_image) = prev_fig
    axis_image.set_data(data)
  fig_no.suptitle(title, y=1.05)
  if prev_fig is None:
    fig_no.show()
  else:
    fig_no.canvas.draw()
  return (fig_no, sub_axis, axis_image)

"""
Save input data as an image without reshaping
Outputs:
  fig_no: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data : np.ndarray of shape (height, width) or (n, height, width) 
  title: string for title of figure
  save_filename: string holding output directory for writing, figures will not display with GUI if set
"""
def save_data(data, title="", save_filename=""):
  fig_no, sub_axis = plt.subplots(1)
  axis_image = sub_axis.imshow(data, cmap="Greys", interpolation="nearest")
  cbar = fig_no.colorbar(axis_image)
  fig_no.suptitle(title, y=1.05)
  if save_filename == "":
    save_filename = "./output.ps"
  fig_no.savefig(save_filename, transparent=True, bbox_inches="tight", pad_inches=0.01)
  plt.close(fig_no)
  return 0

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
"""
def display_data(data, title="", prev_fig=None):
  if prev_fig is None:
    fig_no, sub_axis = plt.subplots(1)
    axis_image = sub_axis.imshow(data, cmap="Greys", interpolation="nearest")
    cbar = fig_no.colorbar(axis_image)
  else:
    (fig_no, sub_axis, axis_image) = prev_fig
    axis_image.set_data(data)
    axis_image.autoscale()

  fig_no.suptitle(title, y=1.05)

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
  im_shp = img.shape[0]
  norm = np.vstack([(img[idx,:]-np.mean(img[idx,:]))/np.std(img[idx,:]) for idx in range(im_shp)])
  return norm
