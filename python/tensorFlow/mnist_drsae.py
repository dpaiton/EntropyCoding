import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
import IPython
from tensorflow.examples.tutorials.mnist import input_data

## User-defined parameters
m_ = 784               # Number of pixels
n_ = 400               # Number of hidden units
l_ = 10                # Number of categories
lambda_ = 1.0          # Sparsity tradeoff
gamma_ = 0.2           # Supervised loss tradeoff
eta_ = 0.01            # Time constant for z update
batch_ = 60            # Number of images in a batch
learning_rate_ = 0.05  # Learning rate for SGD
num_steps_ = 11        # Number of time steps for enoding
num_trials_ = 1000000  # Number of batches to learn weights
train_display_ = 100   # How often to update training stats outputs
val_display_ = -1      # How often to update validation stats outputs
checkpoint_ = 50000    # How often to checkpoint weights

## Input data
dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

## Placeholders & Constants
x = tf.placeholder(dtype=tf.float32, shape=[batch_, m_])  # Data
y = tf.placeholder(dtype=tf.float32, shape=[batch_, l_])  # Image labels
gamma = tf.placeholder(dtype=tf.float32) # Supervised loss tradeoff hyper-parameter
zeros = tf.constant(np.zeros([batch_, n_], dtype=np.float32))
identity_mat = tf.constant(np.identity(n_, dtype=np.float32))

## Weight vectors
# Encoding weights
E = tf.Variable(hf.l2_normalize_rows(tf.truncated_normal([n_, m_], mean=0.0, stddev=1.0,
    dtype=tf.float32)), trainable=True, name="encode_weights")

# Decoding weights
#D = tf.Variable(tf.truncated_normal([m_, n_], mean=0.0, stddev=1.0,
#    dtype=tf.float32), trainable=True)
alpha_1 = tf.Variable(np.float32(1.0), trainable=True, name="alpha_1")
D_W = tf.Variable(tf.truncated_normal([m_, n_], mean=0.0, stddev=np.sqrt(0.01),
    dtype=tf.float32), trainable=True, name="decode_weights")
D = tf.abs(alpha_1) * tf.transpose(E) + D_W


# Explaining away weights
#S = tf.Variable(tf.truncated_normal([n_, n_], mean=0.0, stddev=1.0,
#    dtype=tf.float32), trainable=True)
alpha_2 = tf.Variable(np.float32(1.0), trainable=True, name="alpha_2")
S_W = tf.Variable(tf.truncated_normal([n_, n_], mean=0.0, stddev=np.sqrt(0.01),
    dtype=tf.float32), trainable=True, name="explaining_away_weights")
S = tf.abs(alpha_2) * (tf.matmul(E, tf.transpose(E)) - identity_mat) + S_W

# Classification matrix
C = tf.Variable(tf.truncated_normal([l_, n_], mean=0.0, stddev=1.0,
    dtype=tf.float32), trainable=True, name="classification_weights")

# Bias
b = tf.Variable(tf.truncated_normal([1, n_], mean=0.0, stddev=1.0,
    dtype=tf.float32), trainable=True, name="bias")

## Dynamic variables
z = tf.Variable(np.zeros([batch_, n_], dtype=np.float32), trainable=False, name="z")

# Discretized update rule: z(t+1) = eta * (x E^T + relu(z(t)) (S-I) - b)
dz = eta_ * (\
    tf.matmul(x, tf.transpose(E)) + \
    tf.matmul(tf.nn.relu(z), S) - \
    tf.matmul(tf.constant(np.ones([batch_, 1]), dtype=tf.float32), b))
step_z = tf.group(z.assign(tf.nn.relu(dz)))
#step_z = tf.group(z.assign_add(tf.nn.relu(dz)))

# Final z value will be the same as one update state
z_T = dz

# Normalized z for classification
norm_z = tf.truediv(z_T, tf.sqrt(tf.reduce_sum(tf.pow(z_T, 2.0))))

## Network outputs
y_ = tf.nn.softmax(tf.matmul(norm_z, tf.transpose(C))) # label output
x_ = tf.matmul(z_T, tf.transpose(D))  # reconstruction

## Loss fucntions
cross_entropy_loss = -tf.reduce_sum(y * tf.log(y_)) # reduce_sum sums across all dim (images in minibatch, classes)
euclidean_loss = 0.5 * tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(x, x_), 2.0)))
sparse_loss = tf.reduce_sum(tf.abs(z_T))
unsupervised_loss = euclidean_loss + lambda_ * sparse_loss
supervised_loss = gamma_ * cross_entropy_loss
total_loss = unsupervised_loss + supervised_loss

## Weight update method
train_step = tf.train.GradientDescentOptimizer(learning_rate_).minimize(total_loss)

## Accuracy functions
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Plotting figures
e_prev_fig = None
d_prev_fig = None
s_prev_fig = None
c_prev_fig = None
b_prev_fig = None

init_op = tf.initialize_all_variables()
if checkpoint_ != -1:
    saver = tf.train.Saver(max_to_keep=int(np.float32(num_trials_)/np.float32(checkpoint_)))
    import os
    os.makedirs('checkpoints')

with tf.Session() as sess:
    with tf.device('/cpu:0'): # specify hardware, could be cpu:0, gpu:0, gpu:1, etc
        sess.run(init_op)
        for batch_idx in range(num_trials_):
            ## Load data
            batch = dataset.train.next_batch(batch_)
            input_image = hf.normalize_image(batch[0], divide_l2=True)

            """
            Need to normalize weight matrices
            per Rolfe & Lecun (2013)
             Magnitude of rows of E are bounded by 1.25/num_steps_
                ***Changing to be l2 norm of each basis vector in E is 1 (from rozell et al 2008)
             Magnitude of columns of D are bounded by 1
             Magnitude of rows of C are bounded by 5
            """
            #E = hf.normalize_rows(E, 1.25/np.float32(num_steps_))
            E = hf.l2_normalize_rows(E)
            D_W = hf.normalize_cols(D_W, 1.0)
            C = hf.normalize_rows(C, 5.0)

            ## First find image code, z
            z.assign(zeros)
            for t in range(int(num_steps_ - 1)):
                step_z.run({x:input_image})

            ## Use converged z to compute loss & update weights
            train_step.run({x:input_image, y:batch[1], gamma:0.0})

            if train_display_ != -1 and batch_idx % train_display_ == 0:
                train_accuracy = accuracy.eval({x:input_image, y:batch[1]})
                print('Completed batch number %g out of %g'%(batch_idx, num_trials_))
                print('\ttrain accuracy:\t\t%g'%(train_accuracy))
                print('\teuclidean loss:\t\t%g'%(euclidean_loss.eval({x:input_image})))
                print('\tsparse loss:\t\t%g'%(sparse_loss.eval({x:input_image})))
                print('\tcross-entropy loss:\t%g'%(cross_entropy_loss.eval({x:input_image, y:batch[1]})))
                print('\talpha_1 value:\t\t%g'%(alpha_1.eval()))
                print('\talpha_2 value:\t\t%g'%(alpha_2.eval()))

            if val_display_ != -1 and batch_idx % val_display_ == 0:
                e_prev_fig = hf.display_data(E.eval().reshape(n_, int(np.sqrt(m_)), int(np.sqrt(m_))),
                    title='Encoding matrix at trial number '+str(batch_idx), prev_fig=e_prev_fig)
                d_prev_fig = hf.display_data(tf.transpose(D).eval().reshape(n_, int(np.sqrt(m_)), int(np.sqrt(m_))),
                    title='Decoding matrix at trial number '+str(batch_idx), prev_fig=d_prev_fig)
                s_prev_fig = hf.display_data(S.eval(),
                    title='Explaining-away matrix at trial number '+str(batch_idx), prev_fig=s_prev_fig)
                c_prev_fig = hf.display_data(C.eval().reshape(l_, int(np.sqrt(n_)), int(np.sqrt(n_))),
                    title='Classification matrix at trial number '+str(batch_idx), prev_fig=c_prev_fig)
                b_prev_fig = hf.display_data(b.eval().reshape(int(np.sqrt(n_)), int(np.sqrt(n_))),
                    title='Bias at trial number '+str(batch_idx)+'\nEach pixel represents the bias for a neuron',
                    prev_fig=b_prev_fig)
                batch = dataset.validation.next_batch(batch_)
                val_accuracy = accuracy.eval({x:input_image, y:batch[1]})
                print('---validation accuracy %g'%(val_accuracy))

            if checkpoint_ != -1 and batch_idx % checkpoint_ == 0:
                save_path = saver.save(sess=sess,
                    save_path='checkpoints/drsae_model',
                    global_step=batch_idx,
                    latest_filename='latest.log')
                print("Model saved in file %s"%save_path)

print('Model has finished learning.')
IPython.embed()
