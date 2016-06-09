import os
import IPython
import numpy as np
import matplotlib.pyplot as plt

import helper_functions as hf
import load_checkpoint

import tensorflow as tf

def main(args):
  checkpoint_sess = load_checkpoint.checkpoint_session(args)

  checkpoint_sess.do_load()

  phi = checkpoint_sess.session.graph.get_tensor_by_name("weights/phi:0")
  W = checkpoint_sess.session.graph.get_tensor_by_name("weights/W:0")

  W_eval = checkpoint_sess.session.run(W, feed_dict={})
  l_ = W_eval.shape[0]
  m_ = W_eval.shape[1]

  w_prev_fig = None
  #w_prev_fig = hf.display_data_tiled(W_eval.reshape(l_, int(np.sqrt(m_)), int(np.sqrt(m_))),
  #  title="Classification matrix", prev_fig=w_prev_fig)

  IPython.embed()

if __name__ == "__main__":
  chk_dir = os.path.expanduser('~')+"/Work/Projects/output/checkpoints/"

  args = dict()

  # Checkpoint loading 
  args["checkpoint_dir"] = chk_dir
  # TF GraphDef file to load
  args["input_graph"] = chk_dir+"/lca_gradient_graph_v0.pb"
  # TF saver file to load
  args["input_saver"] = chk_dir+"/saver.def"
  # TF variables (checkpoint made with saver.save()) file to load
  args["input_checkpoint"] = chk_dir+"/lca_checkpoint_v0_FINAL-50"
  # TF GraphDef save name
  args["output_graph"] = chk_dir+"/lca_checkpoint_v0_FINAL-50.frozen"

  # Path for analysis outputs
  args["out_path"] = os.path.expanduser('~')+"/Work/Projects/analysis/"

  main(args)
