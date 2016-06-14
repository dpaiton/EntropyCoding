"""
    This code implements the model described in
    JT Rolfe, Y Lecun (2013) - Discriminative Recurrent Sparse Auto-Encoders

    Code written by Dylan Paiton
"""
import os
import numpy as np
import helper_functions as hf
import IPython


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import drsae_schedule as scheduler

## User-defined parameters
# Dimensions
m_ = 784               # Number of pixels
n_ = 400               # Number of hidden units
l_ = 10                # Number of categories
batch_ = 60            # Number of images in a batch

train_display_ = 20    # How often to update training stats outputs
val_display_ = 50      # How often to update validation stats outputs
display_plots_ = False # If True, plots will display on train_display_ intervals
device_ = "/cpu:0"     # Specify hardware; can be "/cpu:0", "/gpu:0", "/gpu:1"

## Checkpointing
# Writing
checkpoint_ = 1000    # How often to checkpoint weights. -1 for no checkpointing
checkpoint_write_prefix_ = "v0"
checkpoint_base_path = os.path.expanduser('~')+"/Work/EntropyCoding/python/tensorFlow/output/"

# Reading
load_checkpoint_ = False
global_batch_index_ = 20000
checkpoint_sched_no_ = 1
checkpoint_read_prefix_ = "v0"

eps = 1e-12

tf.set_random_seed(1234567890)

## Input data
dataset = input_data.read_data_sets("MNIST_data", one_hot=True)

## Placeholders, constants, parameters
with tf.name_scope("Constants") as scope:
  x = tf.placeholder(dtype=tf.float32, shape=[m_, None], name="input_data")   # Image data
  y = tf.placeholder(dtype=tf.float32, shape=[l_, None], name="input_label")  # Image labels

with tf.name_scope("Parameters") as scope:
  lamb = tf.placeholder(dtype=tf.float32, shape=(), name="sparsity_tradeoff")    # Sparsity tradeoff
  gamma = tf.placeholder(dtype=tf.float32, shape=(), name="supervised_tradeoff") # Supervised loss tradeoff
  lr = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate")          # Gradient descent learning rate

## Learned variables - to be trained with backprop
# Encoding weights
with tf.name_scope("encode_weights") as scope:
  E = tf.Variable(tf.nn.l2_normalize(tf.truncated_normal([n_, m_], mean=0.0, stddev=np.sqrt(1.0),
    dtype=tf.float32, name="E_init"), dim=1, epsilon=eps, name="row_l2_norm"),
    trainable=True, name="encode_weights")

# Decoding weights
with tf.name_scope("decode_weights") as scope:
  D = tf.Variable(tf.nn.l2_normalize(tf.truncated_normal([m_, n_], mean=0.0, stddev=np.sqrt(1.0),
    dtype=tf.float32, name="D_init"), dim=0, epsilon=eps, name="col_l2_norm"),
    trainable=True, name="D")

# Explaining away weights
with tf.name_scope("recurrent_weights") as scope:
  S = tf.Variable(tf.truncated_normal([n_, n_], mean=0.0, stddev=np.sqrt(1e-4),
    dtype=tf.float32, name="S_init"), trainable=True, name="S")

# Classification matrix
with tf.name_scope("classification_weights") as scope:
  C = tf.Variable(tf.truncated_normal([l_, n_], mean=0.0, stddev=np.sqrt(1.0),
    dtype=tf.float32, name="C_init"), trainable=True, name="C")

# Bias
b = tf.Variable(np.zeros([n_, 1], dtype=np.float32), trainable=True, name="bias")

## Dynamic variables - not to be trained with backrpop
#TODO: Plot sorted hist of average amplitude of Zs across whole dataset
z = tf.Variable(tf.zeros(shape=tf.pack([n_, tf.shape(x)[1]]), dtype=tf.float32, name="z_init"),
    trainable=False, validate_shape=False, name="z")

# Discretized update rule: z(t+1) = ReLU(x * E + z(t) * S - b)
with tf.name_scope("update_z") as scope:
  zT = tf.nn.relu(tf.matmul(E, x, name="encoding_transform") +\
    tf.matmul(S, z, name="explaining_away") -\
    tf.matmul(b, tf.ones(shape=tf.pack([1, tf.shape(z)[1]])), name="bias_replication"), name="update_z")

  step_z = tf.group(z.assign(zT), name="do_update_z")

with tf.name_scope("reset_z") as scope:
  reset_z = tf.group(z.assign(tf.zeros(shape=tf.pack([n_, tf.shape(z)[1]]), dtype=tf.float32, name="zeros")),
    name="do_reset_z")

## Network outputs
with tf.name_scope("output") as scope:
  with tf.name_scope("label_estimate"):
    # TODO: Look at values coming out of y_, what is the range?
    # Control the norm of the cols of C to get y_ to be in a desired range.
    # should l2 normalize C and then rescale. as scalar on C increases to inf, you're
    # pushing it to winner take all
    # TODO: try with & without normalization for DrSAE++
    ## Plot softmax output for each trial
    y_ = tf.nn.softmax(tf.matmul(C, tf.nn.l2_normalize(zT, dim=0, epsilon=eps, name="col_l2_norm"),
      name="classify"), name="softmax") # label output
  with tf.name_scope("image_estimate"):
    ## TODO: plot average SNRdb of recons in a whole set
    x_ = tf.matmul(D, zT, name="reconstruction")  # reconstruction

## Loss fucntions
with tf.name_scope("loss") as scope:
  with tf.name_scope("unsupervised_loss"):
    with tf.name_scope("euclidean_loss"):
      euclidean_loss = 0.5 * tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(x, x_), 2.0)))
    with tf.name_scope("sparse_loss"):
      sparse_loss = lamb * tf.reduce_sum(tf.abs(zT))
    unsupervised_loss = euclidean_loss + sparse_loss
  with tf.name_scope("supervised_loss"):
    with tf.name_scope("cross_entropy_loss"):
      cross_entropy_loss = gamma * -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
    supervised_loss = cross_entropy_loss
  total_loss = unsupervised_loss + supervised_loss

## Normalizing operations
with tf.name_scope("normalize_weights") as scope:
  norm_E = E.assign(tf.nn.l2_normalize(E, dim=1, epsilon=eps, name="row_l2_norm"))
  norm_D = D.assign(tf.nn.l2_normalize(D, dim=0, epsilon=eps, name="col_l2_norm"))
  norm_C = C.assign(tf.nn.l2_normalize(C, dim=1, epsilon=eps, name="row_l2_norm"))
  normalize_weights = tf.group(norm_E, norm_D, norm_C, name="do_weight_normalization")

## Load in scheduler
schedules = scheduler.schedule().blocks

## Weight update method
beta_1_ = 0.9
beta_2_ = 0.999
epsilon_ = 1e-7
train_steps = [tf.train.AdamOptimizer(lr, beta_1_, beta_2_, epsilon_,
  name="adam_update").minimize(total_loss,
  var_list=[E, D, S, b, C], name="adam_minimizer") for sch_no in range(len(schedules))]

## Accuracy functions
with tf.name_scope("accuracy_calculation") as scope:
  with tf.name_scope("prediction_bools"):
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
  with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Plotting figures
e_prev_fig = None
d_prev_fig = None
s_prev_fig = None
c_prev_fig = None
b_prev_fig = None
z_prev_fig = None
recon_prev_fig = None

## Checkpointing & graph output
if checkpoint_ != -1:
  if not os.path.exists(checkpoint_base_path+"/checkpoints"):
    os.makedirs(checkpoint_base_path+"/checkpoints")
  #save_list = {"E":E, "D":D, "S":S, "b":b, "C":C}
  #saver = tf.train.Saver(save_list)
  saver = tf.train.Saver()
  saver_def = saver.as_saver_def()
  with open(checkpoint_base_path+"/checkpoints/saver.def", 'wb') as f:
    f.write(saver_def.SerializeToString())

## Initialization
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
  with tf.device(device_):
    global_batch_timer = 0

    sess.run(init_op, 
      feed_dict={x:np.zeros((m_, batch_), dtype=np.float32),
      y:np.zeros((l_, batch_), dtype=np.float32)})
    if load_checkpoint_:
      saver.restore(sess,
        checkpoint_base_path+"/checkpoints/"+\
        checkpoint_read_prefix_+"_s"+str(checkpoint_sched_no_)+"_drsae_model-"+str(global_batch_index_))

    for sched_no, schedule in enumerate(schedules):
      if load_checkpoint_ and sched_no < checkpoint_sched_no_:
        global_batch_timer += schedule["num_batches"]-1
        continue
      
      print("\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
      print("Beginning schedule:")
      print(schedule)
      print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
      lambda_ = schedule["lambda"]               # Sparsity tradeoff
      gamma_ = schedule["gamma"]                 # Supervised loss tradeoff
      learning_rate_ = schedule["learning_rate"] # Learning rate for SGD
      num_steps_ = schedule["num_steps"]         # Number of time steps for enoding
      num_batches_ = schedule["num_batches"]     # Number of batches to learn weights

      for batch_idx in range(num_batches_):
        ## Write graph to text file
        if batch_idx == 0 and sched_no == 0:
          #TODO: setup versioning in graph_def
          tf.train.write_graph(sess.graph_def, checkpoint_base_path+"/checkpoints",
            "drsae_graph.pb", False)
          
        ## Load data
        batch = dataset.train.next_batch(batch_)
        if load_checkpoint_ and batch_idx < (global_batch_index_ - global_batch_timer):
          #TODO: advance batch pointer so this doesn't have to be done in a loop
          continue

        input_image = hf.normalize_image(batch[0]).T
        input_label = batch[1].T

        ## Normalize weight matrices
        normalize_weights.run()

        ## First find image code, z
        reset_z.run()
        for t in range(int(num_steps_)):
          step_z.run({x:input_image})

        ## Use converged z to compute loss & update weights
        train_steps[sched_no].run({\
          x:input_image,
          y:input_label,
          lr:learning_rate_,
          lamb:lambda_,
          gamma:gamma_})

        if train_display_ != -1 and global_batch_timer % train_display_ == 0:
          train_accuracy = accuracy.eval({x:input_image, y:input_label})
          if display_plots_:
            c_prev_fig = hf.display_data_tiled(C.eval().reshape(l_, int(np.sqrt(n_)), int(np.sqrt(n_))),
              title="Classification matrix at trial number "+str(global_batch_timer), prev_fig=c_prev_fig)
            z_prev_fig = hf.display_data(tf.transpose(z).eval(), title="Z values", prev_fig=z_prev_fig)
            b_prev_fig = hf.display_data_tiled(b.eval().reshape(int(np.sqrt(n_)), int(np.sqrt(n_))),
              title="Bias at trial number "+str(global_batch_timer)+"\nEach pixel represents the bias for a neuron",
              prev_fig=b_prev_fig)
            s_prev_fig = hf.display_data_tiled(S.eval(),
              title="Explaining-away matrix at trial number "+str(global_batch_timer), prev_fig=s_prev_fig)
            d_prev_fig = hf.display_data_tiled(tf.transpose(D).eval().reshape(n_, int(np.sqrt(m_)), int(np.sqrt(m_))),
              title="Decoding matrix at trial number "+str(global_batch_timer), prev_fig=d_prev_fig)
            e_prev_fig = hf.display_data_tiled(E.eval().reshape(n_, int(np.sqrt(m_)), int(np.sqrt(m_))),
              title="Encoding matrix at trial number "+str(global_batch_timer), prev_fig=e_prev_fig)
            recon_prev_fig = hf.display_data_tiled(
              tf.transpose(x_).eval({x:input_image}).reshape(batch_, int(np.sqrt(m_)), int(np.sqrt(m_))),
              title="Reconstructions for trial number "+str(global_batch_timer), prev_fig=recon_prev_fig)

          perc_active = 100*np.count_nonzero(z.eval())/np.float32(np.size(z.eval()))
          print("\nGlobal iteration number is %g"%global_batch_timer)
          print("\tCompleted batch number %g out of %g in current schedule"%(global_batch_timer, num_batches_))
          print("\tpercent active:\t\t%g"%(perc_active))
          print("\teuclidean loss:\t\t%g"%(euclidean_loss.eval({x:input_image})))
          print("\tsparse loss:\t\t%g"%(sparse_loss.eval({x:input_image, lamb:lambda_})))
          print("\tunsupervised loss:\t%g"%(unsupervised_loss.eval({x:input_image, lamb:lambda_})))
          print("\tcross-entropy loss:\t%g"%(cross_entropy_loss.eval({x:input_image, y:input_label, gamma:gamma_})))
          print("\tsupervised loss:\t%g"%(supervised_loss.eval({x:input_image, y:input_label, gamma:gamma_})))
          print("\ttrain accuracy:\t\t%g"%(train_accuracy))

        if val_display_ != -1 and global_batch_timer % val_display_ == 0:
          val_image = hf.normalize_image(dataset.validation.images).T
          val_label = dataset.validation.labels.T
          with tf.Session() as temp_sess:
            temp_sess.run(init_op, feed_dict={x:val_image, y:val_label})
            for t in range(num_steps_):
              temp_sess.run(step_z, feed_dict={x:val_image})
            val_accuracy = temp_sess.run(accuracy, feed_dict={x:val_image, y:val_label})
          print("\t---validation accuracy: %g"%(val_accuracy))

        if checkpoint_ != -1 and global_batch_timer % checkpoint_ == 0:
          output_path = checkpoint_base_path+"/checkpoints/"+checkpoint_write_prefix_+schedule["prefix"]+"_s"+str(sched_no)+"_drsae_model"
          save_path = saver.save(sess=sess,
              save_path=output_path,
              global_step=global_batch_timer,
              latest_filename=checkpoint_write_prefix_+"_latest.log")
          print("Model saved in file %s"%save_path)

        global_batch_timer += 1

    print("Model has finished learning schedule.")
    IPython.embed()
