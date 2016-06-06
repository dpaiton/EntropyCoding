import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
import random
import IPython
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python import ops
from google.protobuf import text_format

class RestoredVariable(tf.Variable):
    """
    A variable restored from disk
    """
    def __init__(self, name, trainable=False, collections=None, graph=None):
        if graph is None:
            graph = tf.get_default_graph()

        if collections is None:
            collections = [ops.GraphKeys.VARIABLES]
        if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
            # pylint: disable=g-no-augmented-assignment
            #
            # Pylint wants us to write collections += [...TRAINABLE_VARIABLES] which
            # is not the same (it modifies the list in place.)  Here, we only want to
            # modify the value of the variable, not the list.
            collections = collections + [ops.GraphKeys.TRAINABLE_VARIABLES]
            # pylint: enable=g-no-augmented-assignment

        self._variable = graph.as_graph_element(name).outputs[0]
        self._snapshot = graph.as_graph_element(name + '/read').outputs[0]
        self._initializer_op = graph.as_graph_element(name + '/Assign')

        i_name = name + '/Initializer/'
        keys = [k for k in graph._nodes_by_name.keys() if k.startswith(i_name) and '/' not in k[len(i_name):] ]
        if len(keys) != 1:
            raise ValueError('Could not find initializer for variable', keys)

        self._initial_value = None #initial_value node

        for key in collections:
            graph.add_to_collection(key, self)
        self._save_slice_info = None

graph_file =  "checkpoints/drsae_graph.pb"
checkpoint_file = "checkpoints/v0.01_s1_drsae_model-20000"
var_list = ["E", "D", "S", "b", "C"]

sess = tf.InteractiveSession()

with gfile.FastGFile(graph_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

#sess.graph.as_default()
IPython.embed()
res = tf.import_graph_def(graph_def)

restored_vars = []
for node in graph_def.node:
    if node.op == "Variable":
        restored_vars.append(RestoredVariable(node.name))

saver = tf.train.Saver(var_list=restored_vars, name='restored-' + ('%016x' % random.randrange(16**16)))

saver.restore(tf.get_default_session(), checkpoint_file)


#graph_def = graph_pb2.GraphDef()
#
#with open(graph_file, "rb") as f:
#    graph_def.ParseFromString(f.read)
#    #text_format.Merge(f.read(), graph_def)
#
#saver = tf.train.Saver()
#
#saver.restore(sess, )

IPython.embed()
