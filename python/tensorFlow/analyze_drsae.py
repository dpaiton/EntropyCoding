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
    ## Make output dir
    if not os.path.exists(os.path.dirname(args["out_path"])):
      os.makedirs(os.path.dirname(args["out_path"]))
    ## Extract tensors for analysis
    x = checkpoint.session.graph.get_tensor_by_name("constants/input_data:0")
    y = checkpoint.session.graph.get_tensor_by_name("constants/input_label:0")
    lamb = checkpoint.session.graph.get_tensor_by_name("parameters/sparsity_tradeoff:0")
    E = checkpoint.session.graph.get_tensor_by_name("weights/encode_weights/E:0")
    D = checkpoint.session.graph.get_tensor_by_name("weights/decode_weights/D:0")
    S = checkpoint.session.graph.get_tensor_by_name("weights/recurrent_weights/S:0")
    C = checkpoint.session.graph.get_tensor_by_name("weights/classification_weights/C:0")
    bias = checkpoint.session.graph.get_tensor_by_name("bias:0")
    recon = checkpoint.session.graph.get_tensor_by_name("output/image_estimate/reconstruction:0")
    accuracy = checkpoint.session.graph.get_tensor_by_name("accuracy_calculation/accuracy/avg_accuracy:0")

    ## Evaluate variables for analysis
    E_eval = checkpoint.session.run(E, feed_dict={})
    (n_, m_) = E_eval.shape

    D_eval = checkpoint.session.run(D, feed_dict={}).T
    S_eval = checkpoint.session.run(S, feed_dict={})
    C_eval = checkpoint.session.run(C, feed_dict={})
    l_ = C_eval.shape[0]

    b_eval = checkpoint.session.run(bias, feed_dict={})

    recon_eval = checkpoint.session.run(recon, feed_dict={}).T
    batch_ = recon_eval.shape[0]

    ## Generate plots
    e_prev_fig = hf.save_data_tiled(E_eval.reshape(n_, int(np.sqrt(m_)), int(np.sqrt(m_))),
      title="Encoding matrix at time step "+args["chkpt_iter"],
      save_filename=args["out_path"]+"DRSAE_E-"+args["chkpt_iter"]+".ps")

    d_prev_fig = hf.save_data_tiled(D_eval.reshape(n_, int(np.sqrt(m_)), int(np.sqrt(m_))),
      title="Decoding matrix at time step "+args["chkpt_iter"],
      save_filename=args["out_path"]+"DRSAE_D-"+args["chkpt_iter"]+".ps")

    s_prev_fig = hf.save_data_tiled(S_eval,
      title="Explaining-away matrix at time step "+args["chkpt_iter"],
      save_filename=args["out_path"]+"DRSAE_S-"+args["chkpt_iter"]+".ps")

    c_prev_fig = hf.save_data_tiled(C_eval.reshape(l_, int(np.sqrt(n_)), int(np.sqrt(n_))),
      title="Classification matrix at time step "+args["chkpt_iter"],
      save_filename=args["out_path"]+"DRSAE_C-"+args["chkpt_iter"]+".ps")

    b_prev_fig = hf.save_data_tiled(b_eval.reshape(int(np.sqrt(n_)), int(np.sqrt(n_))),
      title="Bias at time step "+args["chkpt_iter"]+"\nEach pixel represents the bias for a neuron",
      save_filename=args["out_path"]+"DRSAE_b-"+args["chkpt_iter"]+".ps")

    recon_prev_fig = hf.save_data_tiled(
      recon_eval.reshape(batch_, int(np.sqrt(m_)), int(np.sqrt(m_))),
      title="Reconstructions for time step "+args["chkpt_iter"],
      save_filename=args["out_path"]+"DRSAE_recon-"+args["chkpt_iter"]+".ps")

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
    test_set_batched = \
      [(test_images[:,start:start+batch_], test_labels[:,start:start+batch_]) for start in range(0, test_batch_, batch_)]
    for mini_batch in test_set_batched:
      if mini_batch[0].shape[1] == batch_:
        test_accuracy += checkpoint.session.run(accuracy,
          feed_dict={x:mini_batch[0], y:mini_batch[1]})
        num_mini_batches += 1.0
    test_accuracy /= num_mini_batches
    print("test accuracy: %g"%(test_accuracy))

if __name__ == "__main__":
  args = dict()

  # Checkpoint loading
  args["chkpt_dir"] = os.path.expanduser('~')+"/Work/Projects/drsae_output/checkpoints/"
  # Checkpoint iteartion number for loading
  args["chkpt_iter"] = "60000"
  # Checkpoint version number
  args["chkpt_ver"] = "1"
  # Schedule number
  args["sched_num"] = "4"
  # TF GraphDef file to load
  args["input_graph"] = args["chkpt_dir"]+"/drsae_graph_v"+args["chkpt_ver"]+".pb"
  # TF saver file to load
  args["input_saver"] = args["chkpt_dir"]+"/drsae_saver_v"+args["chkpt_ver"]+".def"
  # TF variables (checkpoint made with saver.save()) file to load
  args["input_checkpoint"] = args["chkpt_dir"]+"/drsae_checkpoint_v"+args["chkpt_ver"]+"_s"+args["sched_num"]+"-"+args["chkpt_iter"]
  # TF GraphDef save name
  args["output_graph"] = args["chkpt_dir"]+"/drsae_checkpoint_v"+args["chkpt_ver"]+"_s"+args["sched_num"]+"-"+args["chkpt_iter"]+".frozen"
  # Path for analysis outputs
  args["out_path"] = os.path.expanduser('~')+"/Work/Projects/drsae_analysis/v"+args["chkpt_ver"]+"_"
  # Other arguments
  args["list_nodes"] = False

  main(args)
