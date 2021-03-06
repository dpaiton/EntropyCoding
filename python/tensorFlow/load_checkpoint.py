import os
from tensorflow.python.client import graph_util
import tensorflow as tf

class checkpoint_session:
  def __init__(self, params):
    if not tf.gfile.Exists(params["input_graph"]):
      print("Input graph file '" + params["input_graph"] + "' does not exist!")
      return -1

    if not tf.gfile.Exists(params["input_saver"]):
      print("Input saver file '" + params["input_saver"] + "' does not exist!")
      return -1

    if not tf.gfile.Glob(params["input_checkpoint"]):
      print("Input checkpoint '" + params["input_checkpoint"] + "' doesn't exist!")
      return -1

    if params["output_graph"] == "":
      params["output_graph"] = "./"

    self.params = params
    self.session = None
    self.graph_def = None
    self.node_names = []
    self.loaded = False

  def load(self):
    # Read binary graph file into graph_def structure
    self.graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(self.params["input_graph"], "rb") as f:
      self.graph_def.ParseFromString(f.read())

    # Strip nodes of device specification, collect list of node names
    for node in self.graph_def.node:
      node.device = ""
      self.node_names.append(node.name)

    # Load the session graph
    _ = tf.import_graph_def(self.graph_def, name="")

    # Initialize the session, restore variables from checkpoint
    self.session = tf.Session()
    with tf.gfile.FastGFile(self.params["input_saver"], "rb") as f:
      saver_def = tf.train.SaverDef()
      saver_def.ParseFromString(f.read())
      saver = tf.train.Saver(saver_def=saver_def)
      saver.restore(self.session, self.params["input_checkpoint"])

    self.loaded = True

  def write_constant_graph_def(self):
    if not os.path.exists(os.path.dirname(self.params["output_graph"])):
      os.makedirs(os.path.dirname(self.params["output_graph"]))
    output_graph_def = graph_util.convert_variables_to_constants(self.session, self.graph_def,
      self.node_names)
    with tf.gfile.GFile(self.params["output_graph"], "wb") as f:
      f.write(output_graph_def.SerializeToString())

  def close_sess(self):
    self.session.close()
