"""
Deep Sparse Coding model

Written by Dylan Paiton
"""

import os, argparse, logging, timeit

import numpy as np
import tensorflow as tf
import helper_functions as hf

import IPython

#For MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

## Logging
LOG_FORMAT = "%(filename)s:%(funcName)s:%(asctime)s.%(msecs)03d -- %(message)s"

## Argument parsing
parser = argparse.ArgumentParser(description="Apply a hierarchical sparse coding model to data.",
        usage="deep_sparse_coding.py -b <num_batches>")
parser.add_argument("-b", "--num-batches",
        default=1000, type=int, required=True,
        help="number of batches to present for training")
parser.add_argument("-g", "--graph-output",
        default="checkpoints", type=str, required=False,
        help="output directory for graph saving")
parser.add_argument("-c", "--checkpoint-dir",
        default="checkpoints", type=str, required=False,
        help="output directory for checkpoints")
parser.add_argument("-cf", "--checkpiont-frequency",
        default=-1, type=int, required=False,
        help="time step frequency to write checkpoints; -1 for never")
parser.add_argument("-pw", "--prefix-write",
        default="", type=str, required=False,
        help="model prefix for saving")
parser.add_argument("-pr", "--prefix-read",
        default="", type=str, required=False,
        help="model prefix for reading")
parser.add_argument("-d", "--device",
        default="cpu:0", type=str, required=False,
        help="specify hardware; can be 'cpu:0', 'gpu:0', 'gpu:1'")
parser.add_argument("-s", "--show-plots",
        action="store_true", required=False,
        help="set to show plots")
parser.add_argument("-sd", "--status-display",
        default=-1, type=int, required=False,
        help="time step frequency to display text outputs; -1 for never")
parser.add_argument("-v", "--verbose",
        action="store_true", required=False,
        help="set for verbose outputs")

"""
Construct a tensorflow constant tensor of zeros
inputs:
        shape : int or sequence of ints
            Shape of new tensor
        name : string, optional
            Name of tensor
returns:
        Tensorfor constant with given shape and name, filled with zeros
"""
def zeros_constant(shape, name=""):
    return tf.constant(np.zeros(shape, dtype=np.float32), name=name)

"""
Construct a tensorflow constant tensor that is the identity matrix
inputs:
        shape : int
            New tensor will be a 2-dimensional matrix of size [shape,shape]
        name : string, optional
            Name of tensor
returns:
        Tensorfor constant that containst the identity matrix
"""
def identity_matrix(shape, name=""):
    return tf.constant(np.identity(shape, dtype=np.float32), name=name)

"""
Dimension specification:
    b    : number of data samples in a mini-batch
    l    : number of layers
    m[l] : number of neurons in a given layer l={0..L},
           where m[0] gives the number of data points
           (e.g. pixels) in a data sample
"""
class deepSparseCodingNetwork:

    def __init__(self, dataObj, params):
        self.dataObj = dataObj
        self.sess = tf.Session()
        self.neuron_list = [None] * len(params["m"])
        self.weight_list = [None] * (len(params["m"]) - 1)
        self.params = params
        self.makeDirectories()
        self.buildModel()

    def makeDirectories(self):
        graph_dir = os.path.abspath(os.path.dirname(self.params["graph_file"]))
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        checkpoint_dir = os.path.abspath(os.path.dirname(self.params["checkpoint_dir"]))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def buildModel(self):
        with tf.name_scope("labels") as scope:
            self.y = tf.placeholder(dtype=tf.float32,
                    shape=[self.params["b"], self.params["m"][-1]], name="data_labels")

        ## L layers, first of which is data
        for layer in range(self.params["l"]):
            if layer == 0: # data layer
                self.neuron_list[layer] = tf.placeholder(dtype=tf.float32,
                        shape=[self.params["b"], self.params["m"][layer]], name="input_data")
            else:
                with tf.name_scope("l"+str(layer)+"_neurons") as scope:
                    self.neuron_list[layer] = tf.Variable(np.zeros([self.params["b"], self.params["m"][layer]],
                        dtype=np.float32), trainable=False, name="a_"+str(layer))

                with tf.name_scope("weights_"+str(layer-1)+str(layer)) as scope:
                    weight_init = hf.l2_normalize_cols(tf.truncated_normal([self.params["m"][layer-1], self.params["m"][layer]],
                        mean=0.0, stddev=np.sqrt(1.0), dtype=tf.float32, name="w_"+str(layer)+"_init"))
                    self.weight_list[layer-1] = tf.Variable(weight_init, trainable=True, name="w_"+str(layer-1))

        with tf.name_scope("output") as scope:
            with tf.name_scope("label_estimate"):
                self.neuron_list[-1].assign(tf.nn.softmax(tf.matmul(self.neuron_list[-2], self.weight_list[-1],
                    name="classify"), name="softmax"))
            with tf.name_scope("image_estimate"):
                self.x_ = tf.matmul(self.neuron_list[1], tf.transpose(self.weight_list[0]), name="reconstruction")

        self.saver = tf.train.Saver()

    def initSess(self):
        self.sess.run(tf.initialize_all_variables())

    def closeSess(self):
        self.sess.close()

    def writeGraph(self):
        output_dir = os.path.abspath(os.path.dirname(self.params["graph_file"]))
        output_file_name = os.path.basename(output_file)
        tf.train.write_graph(self.sess.graph_def, output_dir, output_file_name, as_text=True)
        tf.train.SummaryWriter(output_dir, sess.graph)

    def saveModel(self, time_step):
        output_dir = os.path.abspath(self.params["checkpoint_dir"])
        save_path = self.saver.save(sess=self.sess,
                save_path=output_dir,
                global_step=time_step,
                latest_filename=self.params["checkpoint_dir"]+self.params["checkpoint_write_prefix"]+"_latest.log")
        print("Model saved to file %s"%save_path)

    def loadModel(self, time_step):
        self.saver.restore(self.sess,
                self.params["checkpoint_dir"]+self.params["checkpoint_read_prefix"]+"-"+str(time_step))

   #def computeCoefficients(self):
   #def updateWeights(self):

"""
Entry point
"""
def main(args):
    ## Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S", level=level)

    ## Model parameters
    params = {}
    params["b"] = 60                # Number of data samples (e.g. images) in a batch
    params["m"] = [784, 484, 10]    # Number of neurons in each layer
    params["l"] = 3                 # Number of levels in network, including data layer
    params["num_steps"] = 20        # Number of steps
    params["learning_rate"] = 0.001 # Learning rate for gradient updates
    params["num_batches"] = args.num_batches
    params["checkpoint_dir"] = args.checkpoint_dir
    params["checkpoint_write_prefix"] = args.prefix_write
    params["checkpoint_read_prefix"] = args.prefix_read
    params["graph_file"] = args.graph_output
    params["show_plots"] = args.show_plots
    params["status_display"] = args.status_display
    if args.device is "cpu:0" or "gpu:0" or "gpu:1":
        params["device"] = "/"+args.device
    else:
        logging.warning("Input device specification must be 'cpu:0', 'gpu:0', or 'gpu:1'.")
        params["device"] = "/cpu:0"

    ## Set random seed
    tf.set_random_seed(1234567890)

    ## Input data
    dataset = input_data.read_data_sets("MNIST_data", one_hot=True)
    net = deepSparseCodingNetwork(dataset, params)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
