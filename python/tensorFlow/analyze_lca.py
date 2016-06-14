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

  if args["list_nodes"]:
    print("\n".join(node for node in checkpoint.node_names))
  else:
    ## Extract tensors for analysis
    s = checkpoint.session.graph.get_tensor_by_name("constants/input_data:0")
    y = checkpoint.session.graph.get_tensor_by_name("constants/input_label:0")
    lamb = checkpoint.session.graph.get_tensor_by_name("parameters/sparsity_tradeoff:0")
    phi = checkpoint.session.graph.get_tensor_by_name("weights/phi:0")
    W = checkpoint.session.graph.get_tensor_by_name("weights/w:0")
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
    phi_prev_fig = hf.save_data_tiled(phi_eval.reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
      title="Layer 1 dictionary", save_filename=args["out_path"]+"LCA_Phi-"+args["chkpt_iter"]+".ps")

    w_prev_fig = hf.save_data_tiled(W_eval.reshape(l_, int(np.sqrt(m_)), int(np.sqrt(m_))),
      title="Classification matrix", save_filename=args["out_path"]+"LCA_W-"+args["chkpt_iter"]+".ps")

    recon_prev_fig = hf.save_data_tiled(
      recon_eval.reshape(batch_, int(np.sqrt(n_)), int(np.sqrt(n_))),
      title="Reconstructions", save_filename=args["out_path"]+"LCA_Recon-"+args["chkpt_iter"]+".ps")

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
  args["input_graph"] = chk_dir+"/lca_gradient_graph_v0.pb"
  # TF saver file to load
  args["input_saver"] = chk_dir+"/saver.def"
  # Checkpoint iteartion number for loading
  args["chkpt_iter"] = "30002"
  # TF variables (checkpoint made with saver.save()) file to load
  args["input_checkpoint"] = chk_dir+"/lca_checkpoint_v0-"+args["chkpt_iter"]
  # TF GraphDef save name
  args["output_graph"] = chk_dir+"/lca_checkpoint_v0-"+args["chkpt_iter"]+".frozen"

  # Path for analysis outputs
  args["out_path"] = os.path.expanduser('~')+"/Work/Projects/analysis/v0_"

  # Other arguments
  args["list_nodes"] = False

  main(args)
