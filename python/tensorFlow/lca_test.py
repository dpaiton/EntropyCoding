import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
import os
import IPython
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

## User-defined parameters
n_    = 784         # Num_pixels
m_    = 400         # Num_elements
lamb_ = 0.50        # Threshold potential (sparseness penalty scaling)
dt_   = 0.001       # [s] discrete time constant
tau_  = 0.01        # [s] LCA time constant
lr_   = 0.05        # Learning rate for weight updates (will be divided by batch_)
batch_ = 100        # Number of images in a batch
num_steps_ = 20     # Number of steps to run LCA
num_trials_ = 30000 # Number of batches to learn weights
display_ = 100      # How often to display status updates

tf.set_random_seed(1234567890)
np.random.seed(1234567890)

## Setup data
dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

## Setup constants & placeholders
s = tf.placeholder(tf.float32, shape=[n_, None], name="input") # Placeholder for data
eta = tf.placeholder(tf.float32, shape=(), name="lca_update")       # Placeholder for LCA update rate
weight_lr = tf.placeholder(tf.float32, shape=(), name="weight_lr") # Placeholder for Phi update rule

## Initialize membrane potential
u = tf.Variable(np.zeros([m_, batch_], dtype=np.float32), trainable=False, name="membrane_potential")
a = tf.select(tf.greater(u, lamb_), u-lamb_, tf.zeros(shape=tf.shape(u)), name="thresholded_u")

## Initialize dictionary
phi = tf.Variable(tf.nn.l2_normalize(tf.truncated_normal([n_, m_], mean=0.0,
  stddev=1.0, dtype=tf.float32, name="phi_init"), dim=1, name="row_l2_norm"), name="phi")

## Reconstruction variable
s_ = tf.matmul(phi, a, name="reconstruction")

## Loss functions
euclidean_loss = 0.5 * tf.reduce_sum(tf.pow(tf.sub(s, s_), 2.0))
sparse_loss = lamb_ * tf.reduce_sum(tf.abs(a))
unsupervised_loss = euclidean_loss + sparse_loss

## Discritized membrane update rule
du = tf.sub(tf.sub(tf.matmul(tf.transpose(phi), s),
  tf.matmul(tf.matmul(tf.transpose(phi), phi)
  - tf.constant(np.identity(int(phi.get_shape()[1]), dtype=np.float32), name="identity_matrix"), a)), u)

step_lca = tf.group(u.assign_add(eta * du))

## Discritized weight update rule
phi_optimizer = tf.train.GradientDescentOptimizer(weight_lr, name="grad_optimizer")
auto_gradient = phi_optimizer.compute_gradients(unsupervised_loss, var_list=[phi])

manual_gradient = -tf.matmul(tf.sub(s, s_), tf.transpose(a))

#step_phi = phi_optimizer.apply_gradients([(manual_gradient, phi)])
step_phi = phi_optimizer.apply_gradients(auto_gradient)

## Weight normalization
normalize_phi = tf.group(phi.assign(tf.nn.l2_normalize(phi, dim=1,
  epsilon=1e-12, name="row_l2_norm")), name="do_normalization")

pSNRdB = tf.mul(10.0, tf.log(255.0**2
  / ((1.0 / float(n_)) * tf.reduce_sum(tf.pow(tf.sub(s, s_), 2.0)))),
  name="recon_quality")

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for trial in range(num_trials_):
    norm_input = hf.normalize_image(dataset.train.next_batch(batch_)[0]).T # don't need labels
    for t in range(num_steps_):
      step_lca.run({s:norm_input, eta:dt_/tau_})
    step_phi.run({s:norm_input, weight_lr:float(lr_)/float(batch_)})
    normalize_phi.run()
    if trial % display_ == 0:
      sparsity = 100*np.count_nonzero(a.eval())/np.float32(np.size(a.eval()))
      print("Finished trial %g"%(trial))
      print("\tmax val of u:\t%g"%(u.eval().max()))
      print("\tpercent active:\t%g"%(sparsity))
      print("\tRecon pSNRdB:\t%g"%(pSNRdB.eval({s:norm_input})))
      print("\teuclidean loss:\t%g"%(euclidean_loss.eval({s:norm_input, u:u.eval(), phi:phi.eval()})))
      print("\tsparse loss:\t%g"%(sparse_loss.eval({u:u.eval()})))
      print("\ttotal loss:\t%g"%(unsupervised_loss.eval({s:norm_input, u:u.eval(), phi:phi.eval()})))
      _ = hf.save_data_tiled(tf.transpose(phi).eval().reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
        title='Dictionary for trial number '+str(trial),
        save_filename="outputs/phi_"+str(trial)+".pdf")
      #IPython.embed()
      _ = hf.save_data_tiled(tf.transpose(auto_gradient[0][0]).eval({s:norm_input, weight_lr:float(lr_)/float(batch_)}).reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
        title='TF computed phi gradient for trial number '+str(trial),
        save_filename="outputs/dphi_auto_"+str(trial)+".pdf")
      _ = hf.save_data_tiled(tf.transpose(manual_gradient).eval({s:norm_input, weight_lr:float(lr_)/float(batch_)}).reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
        title='Manually computed phi gradient for trial number '+str(trial),
        save_filename="outputs/dphi_manual_"+str(trial)+".pdf")

  IPython.embed()
