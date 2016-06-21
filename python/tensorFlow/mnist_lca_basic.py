import os
import numpy as np
import helper_functions as hf
import IPython

import tensorflow as tf # always import tensorflow after primary imports
from tensorflow.examples.tutorials.mnist import input_data

## User-defined parameters
# Dimensions
n_     = 784         # number of pixels
m_     = 400         # number of elements
l_     = 10          # number of categories
batch_ = 60          # number of images in a batch

# LCA parameters
dt_     = 0.001      # [s] discrete time constant
tau_    = 0.01       # [s] LCA time constant

# Learning Schedule
#lambda_ = 1.8
lambda_ = 0.005
learning_rate_ = 0.001
num_steps_ = 40
num_batches_ = 20000

# Display & Output
version = "0"           # Append a version number to runs
checkpoint_ = -1        # How often to checkpoint
stats_display_ = 50     # How often to print updates to stdout
generate_plots_ = 200   # How often to generate plots for display or saving
checkpoint_base_path = os.path.expanduser('~')+"/Work/Projects/lca_basic_output/"

# Other
device_ = "/cpu:0"
tf.set_random_seed(1234567890)
eps = 1e-12

## Setup data
dataset = input_data.read_data_sets("MNIST_data", one_hot=True)

## Setup constants & placeholders
with tf.name_scope("constants") as scope:
  s = tf.placeholder(tf.float32, shape=[n_, None], name="input_data")  # Placeholder for data
  y = tf.placeholder(tf.float32, shape=[l_, None], name="input_label") # Placeholder for ground truth

with tf.name_scope("hyper_parameters") as scope:
  eta = tf.placeholder(tf.float32, shape=(), name="lca_update_rate")       # Placeholder for LCA update rate (dt/tau)
  lamb = tf.placeholder(tf.float32, shape=(), name="sparsity_tradeoff")    # Placeholder for sparsity loss tradeoff

## Initialize membrane potential
with tf.name_scope("dynamic_variables") as scope:
  # Internal state variable for sub-threshold dynamics
  u = tf.Variable(tf.zeros(shape=tf.pack([m_, tf.shape(s)[1]]), dtype=tf.float32, name="u_init"),
    trainable=False, validate_shape=False, name="u")
  # Soft threshold function is applied to u to produce a = T(u)
  Tu = tf.select(tf.greater(u, lamb), u-lamb, tf.zeros(shape=tf.shape(u), dtype=tf.float32))

## Initialize weights
with tf.name_scope("weights") as scope:
  weight_init_mean = 0.0
  weight_init_var = 1.0
  phi = tf.Variable(tf.truncated_normal([n_, m_], mean=weight_init_mean,
    stddev=np.sqrt(weight_init_var), dtype=tf.float32, name="phi_init"), trainable=True, name="phi")

## Op for renormalizing weights after update
with tf.name_scope("normalize_weights") as scope:
  norm_phi = phi.assign(tf.nn.l2_normalize(phi, dim=1, epsilon=eps, name="row_l2_norm"))
  normalize_weights = tf.group(norm_phi, name="do_normalization")

with tf.name_scope("output") as scope:
  with tf.name_scope("image_estimate"):
    s_ = tf.matmul(phi, Tu, name="reconstruction")

with tf.name_scope("loss") as scope:
  with tf.name_scope("unsupervised"):
    euclidean_loss = 0.5 * tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(s, s_), 2.0)))
    sparse_loss = lamb * tf.reduce_sum(tf.abs(Tu))
    unsupervised_loss = euclidean_loss + sparse_loss
  total_loss = unsupervised_loss

with tf.name_scope("update_u") as scope:
  ## Discritized membrane update rule
  # du = b - u - G * T(u)
  # du = <phi.T,s> - u - (<phi.T,phi> - I) * T(u)
  du = (tf.matmul(tf.transpose(phi), s) -
    tf.matmul(tf.matmul(tf.transpose(phi), phi) -
    tf.constant(np.identity(int(phi.get_shape()[1])),
    dtype=tf.float32, name="identity_matrix"), Tu) -
    u)
  du_auto = -(tf.gradients(euclidean_loss, Tu)[0] + (u - Tu))

  ## Op to update the state
  step_lca = tf.group(u.assign_add(eta * du), name="do_update_u")

  ## Op to clear u
  clear_u = tf.group(u.assign(tf.zeros(shape=tf.pack([m_, tf.shape(s)[1]]),
    dtype=tf.float32, name="zeros")))

## Weight update method
with tf.name_scope("Optimizer") as scope:
  beta_1_  = 0.9
  beta_2_  = 0.999
  epsilon_ = 1e-7
  train_weights = tf.train.AdamOptimizer(learning_rate_, beta_1_, beta_2_, epsilon_,
    name="adam_optimizer").minimize(total_loss, var_list=[phi], name="adam_minimzer")
  #train_weights = tf.train.GradientDescentOptimizer(0.001,
  #  name="gradient_descent_optimizer").minimize(unsupervised_loss, var_list=[phi], name="minimize")

## Checkpointing & graph output
if checkpoint_ > 0:
  if not os.path.exists(checkpoint_base_path+"/checkpoints"):
    os.makedirs(checkpoint_base_path+"/checkpoints")
  saver = tf.train.Saver()
  saver_def = saver.as_saver_def()
  with open(checkpoint_base_path+"/checkpoints/lca_gradient_saver_v"+version+".def", 'wb') as f:
    f.write(saver_def.SerializeToString())

## Initialization
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
  with tf.device(device_):
    global_step = 0

    ## Run session, passing empty arrays to set up network size
    sess.run(init_op,
      feed_dict={s:np.zeros((n_, batch_), dtype=np.float32),
      y:np.zeros((l_, batch_), dtype=np.float32)})

    for step in range(num_batches_):
      if step == 0:
        tf.train.write_graph(sess.graph_def, checkpoint_base_path+"/checkpoints",
          "lca_gradient_graph_v"+version+".pb", as_text=False)

      ## Load in data
      batch = dataset.train.next_batch(batch_)
      input_image = hf.normalize_image(batch[0]).T
      input_label = batch[1].T

      ## Normalize weights
      normalize_weights.run()

      ## Perform inference
      clear_u.run({s:input_image})
      for t in range(num_steps_):
        step_lca.run({s:input_image, y:input_label, eta:dt_/tau_, lamb:lambda_})

      train_weights.run({\
        s:input_image,
        y:input_label,
        lamb:lambda_})

      ## Print statistics about run to stdout
      if global_step % stats_display_ == 0 and stats_display_ > 0:
        sparsity = 100 * np.count_nonzero(Tu.eval({lamb:lambda_})) / (m_ * batch_)

        print("\nGlobal batch index is %g"%global_step)
        print("Completed step %g out of %g, max val of u is %g, num active of T(u) was %g percent"%(step,
          num_batches_, u.eval().max(), sparsity))
        print("\teuclidean loss:\t\t%g"%(euclidean_loss.eval({s:input_image, lamb:lambda_})))
        print("\tsparse loss:\t\t%g"%(sparse_loss.eval({s:input_image, lamb:lambda_})))
        print("\tunsupervised loss:\t%g"%(unsupervised_loss.eval({s:input_image, lamb:lambda_})))

      if global_step % generate_plots_ == 0 and generate_plots_ > 0:
        plot_out_dir = checkpoint_base_path+"/vis/"
        if not os.path.exists(plot_out_dir):
          os.makedirs(plot_out_dir)

        s_status = hf.save_data_tiled(
          tf.transpose(s_).eval({lamb:lambda_}).reshape(batch_, int(np.sqrt(n_)), int(np.sqrt(n_))),
          title="Reconstructions in step "+str(global_step),
          save_filename=plot_out_dir+"recon_v"+version+"-"+str(global_step).zfill(5)+".ps")

        phi_status = hf.save_data_tiled(
          tf.transpose(phi).eval().reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
          title="Dictionary for step "+str(global_step),
          save_filename=plot_out_dir+"phi_v"+version+"-"+str(global_step).zfill(5)+".ps")

      global_step += 1

    #IPython.embed()
