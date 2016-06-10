import os
import IPython
import numpy as np
import matplotlib.pyplot as plt

import helper_functions as hf
import load_checkpoint

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main(args):
  ## Setup data structure for checkpoint loading
  checkpoint = load_checkpoint.checkpoint_session(args)

  ## Load checkpoint into data structure
  checkpoint.load()

  ## Extract tensors for analysis
  s = checkpoint.session.graph.get_tensor_by_name("constants/input_data:0")
  y = checkpoint.session.graph.get_tensor_by_name("constants/input_label:0")
  lamb = checkpoint.session.graph.get_tensor_by_name("parameters/sparsity_tradeoff:0")
  phi = checkpoint.session.graph.get_tensor_by_name("weights/phi:0")
  W = checkpoint.session.graph.get_tensor_by_name("weights/W:0")
  recon = checkpoint.session.graph.get_tensor_by_name("output/image_estimate/reconstruction:0")
  accuracy = checkpoint.session.graph.get_tensor_by_name("accuracy_calculation/accuracy/avg_accuracy:0")

  ## Evaluate variables for analysis
  W_eval = checkpoint.session.run(W, feed_dict={})
  l_ = W_eval.shape[0]
  m_ = W_eval.shape[1]

  phi_eval = checkpoint.session.run(phi, feed_dict={}).T

  recon_eval = checkpoint.session.run(recon, feed_dict={lamb:0.1}).T
  (batch_, n_) = recon_eval.shape

  ## Generate plots
  phi_prev_fig = None
  w_prev_fig = None
  recon_prev_fig = None

  phi_prev_fig = hf.display_data_tiled(phi_eval.reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
    title="Layer 1 dictionary", prev_fig=phi_prev_fig,
    save_filename=args["out_path"]+"LCA_Phi.ps")

  w_prev_fig = hf.display_data_tiled(W_eval.reshape(l_, int(np.sqrt(m_)), int(np.sqrt(m_))),
    title="Classification matrix", prev_fig=w_prev_fig, save_filename=args["out_path"]+"LCA_W.ps")

  recon_prev_fig = hf.display_data_tiled(
    recon_eval.reshape(batch_, int(np.sqrt(n_)), int(np.sqrt(n_))),
    title="Reconstructions", prev_fig=recon_prev_fig,
    save_filename=args["out_path"]+"LCA_Recon.ps")

  ## Setup data for testing
  dataset = input_data.read_data_sets("MNIST_data", one_hot=True)
  test_images = hf.normalize_image(dataset.test.images).T # Full test set
  test_labels = dataset.test.labels.T

  ## Compute test accuracy
  # Session definition is fixed to size batch_,
  # so we have to test batch_ images at a time
  test_batch_ = test_images.shape[1]
  test_accuracy = 0.0
  num_mini_batches = 0.0
  test_set_batched = [(test_images[:,start:start+batch_], test_labels[:,start:start+batch_]) for start in range(0, test_batch_, batch_)]
  for mini_batch in test_set_batched:
      if mini_batch[0].shape[1] == batch_:
        test_accuracy += checkpoint.session.run(accuracy, feed_dict={s:mini_batch[0], y:mini_batch[1], lamb:0.1})
        num_mini_batches += 1.0
  test_accuracy /= num_mini_batches
  print("test accuracy: %g"%(test_accuracy))

if __name__ == "__main__":
  chk_dir = os.path.expanduser('~')+"/Work/Projects/output/checkpoints/"

  args = dict()

  # Checkpoint loading 
  args["checkpoint_dir"] = chk_dir
  # TF GraphDef file to load
  args["input_graph"] = chk_dir+"/lca_gradient_graph_v1.pb"
  # TF saver file to load
  args["input_saver"] = chk_dir+"/saver.def"
  # TF variables (checkpoint made with saver.save()) file to load
  args["input_checkpoint"] = chk_dir+"/lca_checkpoint_v1-90"
  # TF GraphDef save name
  args["output_graph"] = chk_dir+"/lca_checkpoint_v1-90.frozen"

  # Path for analysis outputs
  args["out_path"] = os.path.expanduser('~')+"/Work/Projects/analysis/v1_"

  main(args)
