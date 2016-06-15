import os, io, re
import IPython
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt


#TOOD: Create activation plots similar to those in Rolfe et al
#def plot_connection_summaries(enc_w, dec_w, rec_w

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
  supervised_loss = np.array([float(val) for val in re.findall("[^un]supervised loss:\s+(\d+?\.?\d*)", log_text)])
  train_accuracy = np.array([float(val) for val in re.findall("train accuracy:\s+(\d+\.?\d*)", log_text)])
  val_accuracy = np.array([float(val) for val in re.findall("validation accuracy:\s+(\d+\.?\d*)", log_text)])

  return {"max_iter":max_iter, "batch_iter":batch_iter,
    "euclidean_loss":euclidean_loss, "sparse_loss":sparse_loss,
    "unsupervised_loss":unsupervised_loss, "supervised_loss":supervised_loss,
    "train_accuracy":train_accuracy, "val_accuracy":val_accuracy}

def main(args):
  log_base_path = os.path.expanduser('~')+"/Work/Projects/"+args["model_name"]+"_output/logfiles/"
  out_base_path = os.path.expanduser('~')+"/Work/Projects/"+args["model_name"]+"_analysis/"
  log_file = log_base_path+"v"+args["model_version"]+"_out.log"

  ## Make paths
  if not os.path.exists(out_base_path):
    os.makedirs(out_base_path)

  ## Get data
  stats = get_log_outputs(load_file(log_file))

  ## Plot loss stats
  loss_save_filename = out_base_path+args["model_name"]+"_loss_v"+args["model_version"]+".ps"
  acc_save_filename = out_base_path+args["model_name"]+"_accuracy_v"+args["model_version"]+".ps"

  fig_no, sub_axes = plt.subplots(4)
  axis_image = [None]*4
  axis_image[0] = sub_axes[0].plot(stats["batch_iter"], stats["euclidean_loss"])
  axis_image[1] = sub_axes[1].plot(stats["batch_iter"], stats["sparse_loss"])
  axis_image[2] = sub_axes[2].plot(stats["batch_iter"], stats["supervised_loss"])
  axis_image[3] = sub_axes[3].plot(stats["batch_iter"], stats["unsupervised_loss"]+stats["supervised_loss"])

  # All sub-plots share x tick labels
  sub_axes[0].get_xaxis().set_ticklabels([])
  sub_axes[1].get_xaxis().set_ticklabels([])
  sub_axes[2].get_xaxis().set_ticklabels([])

  # Reduce the number of y tick labels to prevent overcrowding
  sub_axes[0].locator_params(axis="y", nbins=5)
  sub_axes[1].locator_params(axis="y", nbins=5)
  sub_axes[2].locator_params(axis="y", nbins=5)
  sub_axes[3].locator_params(axis="y", nbins=5)

  sub_axes[3].set_xlabel("Batch Number")

  sub_axes[0].set_ylabel("Euclidean")
  sub_axes[1].set_ylabel("Sparse")
  sub_axes[2].set_ylabel("Cross Entropy")
  sub_axes[3].set_ylabel("Total")

  ylabel_xpos = -0.1
  sub_axes[0].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[1].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[2].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[3].yaxis.set_label_coords(ylabel_xpos, 0.5)

  fig_no.suptitle("Average Losses per Batch", y=1.0, x=0.5)

  fig_no.savefig(loss_save_filename, transparent=True)

  ## Plot accuracy stats
  fig_no, sub_axes = plt.subplots(2)
  axis_image = [None]*2
  axis_image[0] = sub_axes[0].plot(stats["batch_iter"], stats["train_accuracy"])
  ##TODO: The x values for the validation plot are approximate.
  ##      It would be better to grab the exact global batch index
  ##      for the corresponding validation accuracy value.
  axis_image[1] = sub_axes[1].plot(\
    np.linspace(stats["batch_iter"][0], stats["batch_iter"][-1], len(stats["val_accuracy"])),
    stats["val_accuracy"])

  sub_axes[1].set_xlabel("Batch Number")

  sub_axes[0].set_ylabel("Train Accuracy")
  sub_axes[1].set_ylabel("Val Accuracy")

  ylabel_xpos = -0.1
  sub_axes[0].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[1].yaxis.set_label_coords(ylabel_xpos, 0.5)

  fig_no.suptitle("Average Accuracy per Batch", y=1.0, x=0.5)

  fig_no.savefig(acc_save_filename, transparent=True)

  #IPython.embed()

if __name__ == "__main__":
  args = dict()

  ## Set config
  args["model_name"] = "lca"
  args["model_version"] = "2"

  main(args)
