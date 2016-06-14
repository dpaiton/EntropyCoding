import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')

import os
import numpy as np
import helper_functions as hf
import IPython

import tensorflow as tf # always import tensorflow after primary imports
from tensorflow.examples.tutorials.mnist import input_data
import lca_schedule as scheduler

## User-defined parameters
# Dimensions
n_     = 784        # number of pixels
m_     = 400        # number of elements
l_     = 10         # number of categories
batch_ = 60         # number of images in a batch

# LCA parameters
dt_    = 0.001      # [s] discrete time constant
tau_   = 0.01       # [s] LCA time constant
thresh_ = "soft"    # Type of thresholding for LCA -> can be "hard" or "soft"

# ADADELTA parameters
#learning_rate_ = 0.001
#rho_ = 0.95
#adadelta_epsilon_ = 1e-8

# ADAM parameters
beta_1_ = 0.9
beta_2_ = 0.999
epsilon_ = 1e-7

# Checkpointing
version = "1"           # Append a version number to runs
checkpoint_ = 10000     # How often to checkpoint
checkpoint_base_path = os.path.expanduser('~')+"/Work/EntropyCoding/python/tensorFlow/output/"

# Display & Output
stats_display_ = 100    # How often to print updates to stdout
display_plots_ = False  # Display plots
save_plots_ = True      # Save plots to disc
generate_plots_ = 1000  # How often to generate plots for display or saving
val_test_ = 100         # How often to run the validation test

# Other
device_ = "/cpu:0"
tf.set_random_seed(1234567890)
eps = 1e-12

## Helper functions
def b(phi, s):
  """
  Driving input for LCA model
      b_m = <phi_m^T, s(t)>
  """
  return tf.matmul(tf.transpose(phi), s)

def G(phi):
  """
  Lateral inhibition
      G_m,n = <phi_m, phi_n> - I
  where I is the identity matrix and prevents a neuron from inhibiting itself
  """
  return tf.matmul(tf.transpose(phi), phi) -\
    tf.constant(np.identity(int(phi.get_shape()[1])), dtype=tf.float32, name="identity_matrix")

def T(u, lamb, thresh_type="soft"):
  """
  Soft threshold function
  T(u) = 0        for u <  lambda
         u-lambda for u >= lambda

  Hard threshold function
  T(u) = 0 for u <  lambda
         u for u >= lambda

  TODO: Define as in eq 3.5 of Rozell et al. 2008 for generality
  """
  # tf.select(condition, a, e) generates a new variable with subsets of a and e, based on condition
  # select from a if condition is true; select from e if condition is false
  # here I assign a to be u-lambda and e to be a tensor of zeros, this will perform thresholding function
  if thresh_type is "soft":
    return tf.select(tf.greater_equal(u, lamb),
      u-lamb, tf.zeros(shape=tf.shape(u), dtype=tf.float32))
  else:
    return tf.select(tf.greater_equal(u, lamb),
      u, tf.zeros(shape=tf.shape(u), dtype=tf.float32))

def compute_recon(phi, a):
  """
  Reconstruction of the input is a weighted sum of basis vectors
  recon = <a, phi>
  """
  return tf.matmul(phi, a, name="reconstruction")

## Setup data
dataset = input_data.read_data_sets("MNIST_data", one_hot=True)

## Setup constants & placeholders
with tf.name_scope("constants") as scope:
  s = tf.placeholder(tf.float32, shape=[n_, None], name="input_data")  # Placeholder for data
  y = tf.placeholder(tf.float32, shape=[l_, None], name="input_label") # Placeholder for ground truth

with tf.name_scope("parameters") as scope:
  eta = tf.placeholder(tf.float32, shape=(), name="LCA_update_rate")       # Placeholder for LCA update rate (dt/tau)
  lamb = tf.placeholder(tf.float32, shape=(), name="sparsity_tradeoff")    # Placeholder for sparsity loss tradeoff
  gamma = tf.placeholder(tf.float32, shape=(), name="supervised_tradeoff") # Placeholder for supervised loss tradeoff
  lr = tf.placeholder(tf.float32, shape=(), name="weight_learning_rate")   # Placeholder for Phi update rule

## Initialize membrane potential
with tf.name_scope("dynamic_variables") as scope:
  u = tf.Variable(tf.zeros(shape=tf.pack([m_, tf.shape(s)[1]]), dtype=tf.float32, name="u_init"),
    trainable=False, validate_shape=False, name="u")

with tf.name_scope("weights") as scope:
  ## Initialize dictionary
  # Truncated normal distribution is a standard normal distribution with specified mean and standard
  # deviation, except that values whose magnitude is more than 2 standard deviations from the mean
  # are dropped and re-picked.
  weight_init_mean = 0.0
  weight_init_var = 1.0

  phi = tf.Variable(tf.truncated_normal([n_, m_], mean=weight_init_mean,
    stddev=np.sqrt(weight_init_var), dtype=tf.float32, name="phi_init"), trainable=True, name="phi")

  w = tf.Variable(tf.truncated_normal([l_, m_], mean=weight_init_mean,
    stddev=np.sqrt(weight_init_var), dtype=tf.float32, name="W_init"), trainable=True, name="w")

with tf.name_scope("normalize_weights") as scope:
  norm_phi = phi.assign(tf.nn.l2_normalize(phi, dim=1, epsilon=eps, name="row_l2_norm"))
  norm_w = w.assign(tf.nn.l2_normalize(w, dim=1, epsilon=eps, name="row_l2_norm"))
  normalize_weights = tf.group(norm_phi, norm_w, name="do_normalization")

with tf.name_scope("output") as scope:
  with tf.name_scope("image_estimate"):
    s_ = compute_recon(phi, T(u, lamb, thresh_type=thresh_))
  with tf.name_scope("label_estimate"):
    ## TODO: Test w/ and w/out normalization
    #y_ = tf.nn.softmax(tf.matmul(w, tf.nn.l2_normalize(T(u, lamb, thresh_type=thresh_),
    #  dim=0, epsilon=1e-12, name="col_l2_norm"), name="classify"), name="softmax")
    y_ = tf.nn.softmax(tf.matmul(w, T(u, lamb, thresh_type=thresh_),
      name="classify"), name="softmax")

with tf.name_scope("update_u") as scope:
  ## Discritized membrane update rule
  du = ((1 - eta) * u + eta * (b(phi, s) -
    tf.matmul(G(phi), T(u, lamb, thresh_type=thresh_)) -
    gamma * tf.matmul(tf.transpose(w), tf.mul(y, y_))))

  ## Operation to update the state
  step_lca = tf.group(u.assign(du), name="do_update_u")

  ## Operation to clear u
  clear_u = tf.group(u.assign(tf.zeros(shape=tf.pack([m_, tf.shape(s)[1]]), dtype=tf.float32, name="zeros")))

with tf.name_scope("loss") as scope:
  with tf.name_scope("unsupervised"):
    euclidean_loss = 0.5 * tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(s, s_), 2.0)))
    sparse_loss = lamb * tf.reduce_sum(tf.abs(T(u, lamb, thresh_type=thresh_)))
    unsupervised_loss = euclidean_loss + sparse_loss
  with tf.name_scope("supervised"):
    with tf.name_scope("cross_entropy_loss"):
      cross_entropy_loss = gamma * -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
    supervised_loss = cross_entropy_loss
  total_loss = unsupervised_loss + supervised_loss

with tf.name_scope("accuracy_calculation") as scope:
  with tf.name_scope("prediction_bools"):
    correct_prediction = tf.equal(tf.argmax(y_, 0), tf.argmax(y, 0), name="individual_accuracy")
  with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="avg_accuracy")

## Load in scheduler
schedules = scheduler.schedule().blocks

## Weight update method
with tf.name_scope("Optimizer") as scope:
  var_lists = [[phi] if sch["prefix"] == "unsupervised" else [w] if sch["prefix"] == "supervised" else [phi, w] for sch in schedules]
  #train_weights = [tf.train.adadeltaoptimizer(lr, rho_, adadelta_epsilon_,
  #  name="adadelta_optimizer").minimize(total_loss, var_list=var_lists[sch_no],
  #  name="adadelta_minimzer") for sch_no in range(len(schedules))]
  train_weights = [tf.train.AdamOptimizer(lr, beta_1_, beta_2_, epsilon_,
    name="adam_optimizer").minimize(total_loss, var_list=var_lists[sch_no],
    name="adam_minimzer") for sch_no in range(len(schedules))]

## Checkpointing & graph output
if checkpoint_ != -1:
  if not os.path.exists(checkpoint_base_path+"/checkpoints"):
    os.makedirs(checkpoint_base_path+"/checkpoints")
  saver = tf.train.Saver()
  saver_def = saver.as_saver_def()
  with open(checkpoint_base_path+"/checkpoints/lca_gradient_saver_v"+version+".def", 'wb') as f:
    f.write(saver_def.SerializeToString())

## Initialization
init_op = tf.initialize_all_variables()

if display_plots_:
  w_prev_fig = None
  phi_prev_fig = None
  recon_prev_fig = None

with tf.Session() as sess:
  with tf.device(device_):
    global_step = 0

    ## Run session, passing empty arrays to set up network size
    sess.run(init_op,
      feed_dict={s:np.zeros((n_, batch_), dtype=np.float32),
      y:np.zeros((l_, batch_), dtype=np.float32)})

    for sched_no, schedule in enumerate(schedules):
      print("\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
      print("Beginning schedule:")
      print(schedule)
      print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
      lambda_ = schedule["lambda"]               # Sparsity tradeoff
      gamma_ = schedule["gamma"]                 # Supervised loss tradeoff
      learning_rate_ = schedule["learning_rate"] # Learning rate for SGD
      num_steps_ = schedule["num_steps"]         # Number of time steps for enoding
      num_batches_ = schedule["num_batches"]     # Number of batches to learn weights

      for trial in range(num_batches_):
        batch = dataset.train.next_batch(batch_)
        input_image = hf.normalize_image(batch[0]).T
        input_label = batch[1].T

        if trial == 0 and sched_no == 0:
          tf.train.write_graph(sess.graph_def, checkpoint_base_path+"/checkpoints",
            "lca_gradient_graph_v"+version+".pb", as_text=False)

        ## Normalize weights
        normalize_weights.run()

        ## Converge network
        clear_u.run({s:input_image, y:input_label, eta:dt_/tau_, lamb:lambda_, gamma:gamma_})
        for t in range(num_steps_):
          step_lca.run({s:input_image, y:input_label, eta:dt_/tau_, lamb:lambda_, gamma:gamma_})

        ## Run update method
        train_weights[sched_no].run({\
          s:input_image,
          y:input_label,
          lr:learning_rate_,
          lamb:lambda_,
          gamma:gamma_})

        ## Print statistics about run to stdout
        if global_step % stats_display_ == 0 and stats_display_ != 0:
          sparsity = 100 * np.count_nonzero(T(u, lamb, thresh_).eval({lamb:lambda_})) / (m_ * batch_)
          train_accuracy = accuracy.eval({s:input_image, y:input_label, lamb:lambda_})
          print("\nGlobal batch index is %g"%global_step)
          print("Finished trial %g out of %g, max val of u is %g, num active of a was %g percent"%(trial+1,
            num_batches_, u.eval().max(), sparsity))
          print("\teuclidean loss:\t\t%g"%(euclidean_loss.eval({s:input_image, lamb:lambda_})))
          print("\tsparse loss:\t\t%g"%(sparse_loss.eval({s:input_image, lamb:lambda_})))
          print("\tunsupervised loss:\t%g"%(unsupervised_loss.eval({s:input_image, lamb:lambda_})))
          print("\tsupervised loss:\t%g"%(supervised_loss.eval({s:input_image,
            y:input_label, gamma:gamma_, lamb:lambda_})))
          print("\ttrain accuracy:\t\t%g"%(train_accuracy))

        ## Create plots for visualizing network
        if global_step % generate_plots_ == 0 and generate_plots_ != -1:
          #TODO: plot weight gradients
          if display_plots_:
            w_prev_fig = hf.display_data_tiled(w.eval().reshape(l_, int(np.sqrt(m_)), int(np.sqrt(m_))),
              title="Classification matrix at trial number "+str(global_step), prev_fig=w_prev_fig)
            recon_prev_fig = hf.display_data_tiled(
              tf.transpose(s_).eval({lamb:lambda_}).reshape(batch_, int(np.sqrt(n_)), int(np.sqrt(n_))),
              title="Reconstructions in trial "+str(global_step), prev_fig=recon_prev_fig)
            phi_prev_fig = hf.display_data_tiled(tf.transpose(phi).eval().reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
              title="Dictionary for trial "+str(global_step), prev_fig=phi_prev_fig)
          if save_plots_:
            plot_out_dir = checkpoint_base_path+"/vis/"
            if not os.path.exists(plot_out_dir):
              os.makedirs(plot_out_dir)
            w_status = hf.save_data_tiled(
              w.eval().reshape(l_, int(np.sqrt(m_)), int(np.sqrt(m_))),
              title="Classification matrix at trial number "+str(global_step),
              save_filename=plot_out_dir+"class_tr-"+str(global_step).zfill(5)+".ps")
            s_status = hf.save_data_tiled(
              tf.transpose(s_).eval({lamb:lambda_}).reshape(batch_, int(np.sqrt(n_)), int(np.sqrt(n_))),
              title="Reconstructions in trial "+str(global_step),
              save_filename=plot_out_dir+"recon_tr-"+str(global_step).zfill(5)+".ps")
            phi_status = hf.save_data_tiled(
              tf.transpose(phi).eval().reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
              title="Dictionary for trial "+str(global_step),
              save_filename=plot_out_dir+"phi_tr-"+str(global_step).zfill(5)+".ps")

        ## Test network on validation dataset
        if global_step % val_test_ == 0 and val_test_ != -1:
          val_image = hf.normalize_image(dataset.validation.images).T
          val_label = dataset.validation.labels.T
          with tf.Session() as temp_sess:
            temp_sess.run(init_op, feed_dict={s:val_image, y:val_label})
            for t in range(num_steps_):
              temp_sess.run(step_lca, feed_dict={s:val_image, y:val_label, eta:dt_/tau_, lamb:lambda_, gamma:0})
            val_accuracy = temp_sess.run(accuracy, feed_dict={s:val_image, y:val_label, lamb:lambda_})
          print("\t---validation accuracy: %g"%(val_accuracy))

        ## Write checkpoint to disc
        if global_step % checkpoint_ == 0 and checkpoint_ != -1:
          saver.save(sess, checkpoint_base_path+"/checkpoints/lca_checkpoint_v"+version, global_step=global_step)

        global_step += 1

    ## Write final checkpoint regardless of specified interval
    if checkpoint_ != -1:
      saver.save(sess, checkpoint_base_path+"/checkpoints/lca_checkpoint_v"+version+"_FINAL", global_step=global_step)

    with tf.Session() as temp_sess:
      temp_sess.run(init_op, feed_dict={s:dataset.test.images.T, y:dataset.test.labels.T})
      for t in range(num_steps_):
        temp_sess.run(step_lca, feed_dict={s:dataset.test.images.T, y:dataset.test.labels.T, eta:dt_/tau_, lamb:lambda_, gamma:0})
      test_accuracy = temp_sess.run(accuracy, feed_dict={s:dataset.test.images.T, y:dataset.test.images.T, lamb:lambda_})
      print("Final accuracy: %g"%test_accuracy)

    IPython.embed()
