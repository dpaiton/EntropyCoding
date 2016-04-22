"""
    This code implements a modification of the model described in
    JT Rolfe, Y Lecun (2013) - Discriminative Recurrent Sparse Auto-Encoders

    Dylan Paiton
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
import os
import IPython
from tensorflow.examples.tutorials.mnist import input_data
import drsae_schedule as scheduler

## User-defined parameters
m_ = 784               # Number of pixels
n_ = 484               # Number of hidden units
l_ = 10                # Number of categories
batch_ = 60            # Number of images in a batch
train_display_ = 100   # How often to update training stats outputs
val_display_ = 100     # How often to update validation stats outputs
display_plots_ = False # If True, plots will display on train_display_ intervals
device_ = "/cpu:0"     # Specify hardware; can be "/cpu:0", "/gpu:0", "/gpu:1"

## Checkpointing
checkpoint_ = 10000    # How often to checkpoint weights. -1 for no checkpointing
checkpoint_write_prefix_ = "v0.01"
load_checkpoint_ = False
checkpoint_trial_ = 10000
#checkpoint_read_prefix_ = "unsup_encode_0.001sparse"
#checkpoint_read_prefix_ = "unsup_encode_b_s_d_0.001sparse"
#checkpoint_read_prefix_ = "unsup_encode_b_s_d_0.01sparse" # finished 10000
#checkpoint_read_prefix_ = "sup_e_b_s_d_c_l0.01_g0.1" # finished 15000 iterations
#checkpoint_read_prefix_ = "sup_e_b_s_d_c_l0.01_g0.2" # finished 60000 iterations - nan after 57000 (96% train acc)
checkpoint_read_prefix_ = "unsup_encode_b_s_d_0.01sparse" 


if load_checkpoint_:
    trial_start = checkpoint_trial_
    assert(trial_start < num_trials_)
else:
    trial_start = 0

tf.set_random_seed(1234567890)

## Input data
dataset = input_data.read_data_sets("MNIST_data", one_hot=True)

## Placeholders & Constants
with tf.name_scope("Constants") as scope:
    x = tf.placeholder(dtype=tf.float32, shape=[batch_, m_], name="input_data")  # Image data
    y = tf.placeholder(dtype=tf.float32, shape=[batch_, l_], name="input_label")  # Image labels
    zeros = tf.constant(np.zeros([batch_, n_], dtype=np.float32), name="zeros_matrix")
    identity_mat = tf.constant(np.identity(n_, dtype=np.float32), name="identity_matrix")

with tf.name_scope("Parameters") as scope:
    lamb = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate") # Gradient descent learning rate
    gamma = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate") # Gradient descent learning rate
    eta = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate") # Gradient descent learning rate
    lr = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate") # Gradient descent learning rate
    batch = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate") # Gradient descent learning rate
    num_steps = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate") # Gradient descent learning rate
    num_trials = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate") # Gradient descent learning rate

## Weight vectors
# Encoding weights
with tf.name_scope("encode_weights") as scope:
    E = tf.Variable(hf.l2_normalize_rows(tf.truncated_normal([n_, m_], mean=0.0,
        stddev=1.00, dtype=tf.float32, name="E_init")), trainable=True, name="encode_weights")

# Decoding weights
with tf.name_scope("decode_weights") as scope:
    alpha_1 = tf.Variable(np.float32(1.0), trainable=True, name="alpha_1")
    D_W = tf.Variable(tf.truncated_normal([m_, n_], mean=0.0, stddev=np.sqrt(0.00001),
        dtype=tf.float32, name="D_W_init"), trainable=True, name="D_W")
    D = tf.sigmoid(alpha_1) * tf.transpose(E) + (1-tf.sigmoid(alpha_1)) * D_W

# Explaining away weights
with tf.name_scope("explaining_away_weights") as scope:
    alpha_2 = tf.Variable(np.float32(1.0), trainable=True, name="alpha_2")
    S_W = tf.Variable(tf.truncated_normal([n_, n_], mean=0.0, stddev=np.sqrt(0.00001),
        dtype=tf.float32, name="S_W_init"), trainable=True, name="S_W")
    S = tf.sigmoid(alpha_2) * (identity_mat - tf.matmul(E, tf.transpose(E), name="Gramian")) + (1 - tf.sigmoid(alpha_2)) * S_W

# Classification matrix
with tf.name_scope("classification_weights") as scope:
    C = tf.Variable(tf.truncated_normal([l_, n_], mean=0.0, stddev=1.0,
        dtype=tf.float32, name="C_init"), trainable=True, name="C")

# Bias
b = tf.Variable(np.zeros([1, n_], dtype=np.float32), trainable=True, name="bias")

## Dynamic variables
z = tf.Variable(np.zeros([batch_, n_], dtype=np.float32), trainable=False, name="z")

# Discretized update rule: z(t+1) = ReLU(eta * (x E^T + relu(z(t)) (S-I) - b))
with tf.name_scope("update_z") as scope:
    zT = tf.nn.relu(tf.matmul(x, tf.transpose(E), name="encoding_transform") + \
        tf.matmul(z, S, name="explaining_away_transform") - \
        tf.matmul(tf.constant(np.ones([batch_, 1], dtype=np.float32)), b, name="bias_replication"),
        name="zT")
    step_z = tf.group(z.assign(eta * zT))

# Normalized z for classification
with tf.name_scope("normalize_z") as scope:
    norm_z = tf.truediv(zT, tf.sqrt(tf.reduce_sum(tf.pow(zT, 2.0))))

## Network outputs
with tf.name_scope("output") as scope:
    with tf.name_scope("label_estimate"):
        y_ = tf.nn.softmax(tf.matmul(norm_z, tf.transpose(C), name="classify"), name="softmax") # label output
    with tf.name_scope("image_estimate"):
        x_ = tf.matmul(zT, tf.transpose(D), name="reconstruction")  # reconstruction

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
            cross_entropy_loss = gamma * -tf.reduce_sum(y * tf.log(y_))
        supervised_loss = cross_entropy_loss
    total_loss = unsupervised_loss + supervised_loss

## Weight update method
train_step = tf.train.GradientDescentOptimizer(lr).minimize(total_loss,
        var_list=[E, alpha_1, D_W, alpha_2, S_W, b, C])

## Accuracy functions
with tf.name_scope("accuracy_calculation") as scope:
    with tf.name_scope("prediction_bools"):
        correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Plotting figures
e_prev_fig = None
d_prev_fig = None
s_prev_fig = None
c_prev_fig = None
b_prev_fig = None
recon_prev_fig = None

## Initialization
init_op = tf.initialize_all_variables()

## Load in scheduler
schedules = scheduler.schedule().blocks

if checkpoint_ != -1:
    saver = tf.train.Saver()
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

with tf.Session() as sess:
    with tf.device(device_): 
        sess.run(init_op)

        if load_checkpoint_:
            saver.restore(sess, "checkpoints/"+checkpoint_read_prefix_+"_drsae_model-"+str(checkpoint_trial_))

        ## Write graph to text file
        tf.train.write_graph(sess.graph_def, "checkpoints", "drsae_graph.pb", False)
        tf.train.SummaryWriter("checkpoints", sess.graph_def)

        global_batch_timer = 0
        for sched_no, schedule in enumerate(schedules):
            print("\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
            print("Beginning schedule:")
            print(schedule)
            print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
            lambda_ = schedule["lambda"]               # Sparsity tradeoff
            gamma_ = schedule["gamma"]                 # Supervised loss tradeoff
            eta_ = schedule["eta"]                     # Time constant for z update
            learning_rate_ = schedule["learning_rate"] # Learning rate for SGD
            num_steps_ = schedule["num_steps"]         # Number of time steps for enoding
            num_trials_ = schedule["num_trials"]       # Number of batches to learn weights
            for batch_idx in range(num_trials_):
                ## Load data
                batch = dataset.train.next_batch(batch_)
                global_batch_timer += 1
                if batch_idx < trial_start:
                    #TODO: advance batch pointer so this doesn't have to be done in a loop
                    continue

                input_image = hf.normalize_image(batch[0], divide_l2=False)

                ## Normalize weight matrices
                E.assign(hf.l2_normalize_rows(E))
                D_W.assign(hf.normalize_cols(D_W, 1.0))
                C.assign(hf.normalize_rows(C, 1.0))

                ## First find image code, z
                z.assign(zeros)
                for t in range(int(num_steps_)):
                    step_z.run({x:input_image, eta:eta_})

                ## Use converged zT to compute loss & update weights
                train_step.run({\
                    x:input_image,
                    y:batch[1],
                    lr:learning_rate_,
                    lamb:lambda_,
                    gamma:gamma_})

                if train_display_ != -1 and global_batch_timer % train_display_ == 0:
                    train_accuracy = accuracy.eval({x:input_image, y:batch[1]})
                    if display_plots_:
                        c_prev_fig = hf.display_data(C.eval().reshape(l_, int(np.sqrt(n_)), int(np.sqrt(n_))),
                            title="Classification matrix at trial number "+str(global_batch_timer), prev_fig=c_prev_fig)
                        b_prev_fig = hf.display_data(b.eval().reshape(int(np.sqrt(n_)), int(np.sqrt(n_))),
                            title="Bias at trial number "+str(global_batch_timer)+"\nEach pixel represents the bias for a neuron",
                            prev_fig=b_prev_fig)
                        s_prev_fig = hf.display_data(S.eval(),
                            title="Explaining-away matrix at trial number "+str(batch_idx), prev_fig=s_prev_fig)
                        d_prev_fig = hf.display_data(tf.transpose(D).eval().reshape(n_, int(np.sqrt(m_)), int(np.sqrt(m_))),
                            title="Decoding matrix at trial number "+str(batch_idx), prev_fig=d_prev_fig)
                        e_prev_fig = hf.display_data(E.eval().reshape(n_, int(np.sqrt(m_)), int(np.sqrt(m_))),
                            title="Encoding matrix at trial number "+str(batch_idx), prev_fig=e_prev_fig)
                        recon_prev_fig = hf.display_data(
                            x_.eval({x:input_image}).reshape(batch_, int(np.sqrt(m_)), int(np.sqrt(m_))),
                            title="Reconstructions for trial number "+str(batch_idx), prev_fig=recon_prev_fig)

                    print("\nCompleted batch number %g out of %g"%(batch_idx, num_trials_))
                    print("\ttrain accuracy:\t\t%g"%(train_accuracy))
                    print("\talpha_1 value:\t\t%g"%(1.0/(1.0+np.exp(-alpha_1.eval()))))
                    print("\talpha_2 value:\t\t%g"%(1.0/(1.0+np.exp(-alpha_2.eval()))))
                    print("\teuclidean loss:\t\t%g"%(euclidean_loss.eval({x:input_image})))
                    print("\tsparse loss:\t\t%g"%(sparse_loss.eval({x:input_image, lamb:lambda_})))
                    print("\tunsupervised loss:\t%g"%(unsupervised_loss.eval({x:input_image, lamb:lambda_})))
                    print("\tcross-entropy loss:\t%g"%(cross_entropy_loss.eval({x:input_image, y:batch[1], gamma:gamma_})))
                    print("\tsupervised loss:\t%g"%(supervised_loss.eval({x:input_image, y:batch[1], gamma:gamma_})))

                if val_display_ != -1 and global_batch_timer % val_display_ == 0:
                    val_batch = dataset.validation.next_batch(batch_)
                    val_image = hf.normalize_image(batch[0], divide_l2=False)
                    val_accuracy = accuracy.eval({x:val_image, y:val_batch[1]})
                    print("---validation accuracy: %g"%(val_accuracy))

                if checkpoint_ != -1 and global_batch_timer % checkpoint_ == 0:
                    output_path = "checkpoints/"+checkpoint_write_prefix_+schedule["prefix"]+"_s"+str(sched_no)+"_drsae_model"
                    save_path = saver.save(sess=sess,
                        save_path=output_path,
                        global_step=global_batch_timer,
                        latest_filename=checkpoint_write_prefix_+"_latest.log")
                    print("Model saved in file %s"%save_path)

        print("Model has finished learning schedule.")
        IPython.embed()
