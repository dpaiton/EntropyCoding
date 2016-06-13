import os, io, re
import IPython
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt

"""
Load loss file into memory

Output: string containing log file text

Input: string containing the filename of the log file
"""
def load_file(log_file):
  with open(log_file, "r") as f:
    log_text = f.read()
  return log_text

"""
Generate array that has loss values from text

Output: dictionary containing arrays of loss values

Input: string containing log file text
"""
def get_log_outputs(log_text):
  max_iter = float(re.findall("out of (\d+),", log_text)[0])
  batch_iter = np.array([float(val) for val in re.findall("Global batch index is (\d+)", log_text)])

  euclidean_loss = np.array([float(val) for val in re.findall("euclidean loss:\s+(\d+\.?\d*)", log_text)])
  sparse_loss = np.array([float(val) for val in re.findall("sparse loss:\s+(\d+\.?\d*)", log_text)])
  unsupervised_loss = np.array([float(val) for val in re.findall("unsupervised loss:\s+(\d+\.?\d*)", log_text)])
  supervised_loss = np.array([float(val) for val in re.findall("supervised loss:\s+(\d+\.?\d*)", log_text)])
  train_accuracy = np.array([float(val) for val in re.findall("train accuracy:\s+(\d+\.?\d*)", log_text)])
  val_accuracy = np.array([float(val) for val in re.findall("validation accuracy:\s+(\d+\.?\d*)", log_text)])

  return {"max_iter":max_iter, "batch_iter":batch_iter,
    "euclidean_loss":euclidean_loss, "sparse_loss":sparse_loss,
    "unsupervised_loss":unsupervised_loss, "supervised_loss":supervised_loss,
    "train_accuracy":train_accuracy, "val_accuracy":val_accuracy}

## Set config
log_base_path = os.path.expanduser('~')+"/Work/Projects/output/logfiles/"
out_base_path = os.path.expanduser('~')+"/Work/Projects/analysis/"
log_file = log_base_path+"lca_grad_test.log"
model_version = "0"

## Get data
losses = get_log_outputs(load_file(log_file))

## Plot losses
save_filename = out_base_path+"unsupervised_loss_v"+model_version+".ps"

fig_no, sub_axes = plt.subplots(3)
axis_image = [None]*3
axis_image[0] = sub_axes[0].plot(losses["batch_iter"], losses["euclidean_loss"])
axis_image[1] = sub_axes[1].plot(losses["batch_iter"], losses["sparse_loss"])
axis_image[2] = sub_axes[2].plot(losses["batch_iter"], losses["unsupervised_loss"])

fig_no.suptitle("Average Unsupervised Losses per Batch", y=1.05)

sub_axes[0].get_xaxis().set_ticklabels([])
sub_axes[1].get_xaxis().set_ticklabels([])

sub_axes[2].set_xlabel("Batch Number")

sub_axes[0].set_ylabel("Euclidean Loss")
sub_axes[1].set_ylabel("Sparse Loss")
sub_axes[2].set_ylabel("Total Loss")
fig_no.savefig(save_filename, transparent=True, bbox_inches="tight", pad_inches=0.01)

IPython.embed()

#TOOD: Create activation plots similar to those in Rolfe et al
#def plot_connection_summaries(enc_w, dec_w, rec_w
