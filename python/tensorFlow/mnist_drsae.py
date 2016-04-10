import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
import IPython
from tensorflow.examples.tutorials.mnist import input_data

## User-defined parameters
m_ = 784               # num_pixels
n_ = 400               # num_hidden_units
l_ = 10                # num_categories
lambda_ = 0.1          # sparsity tradeoff parameter
gamma_ = 0.2           # supervised loss tradeoff parameter
eta_ = 0.01            # time constant for z update 
batch_ = 10            # number of images in a batch
num_steps_ = 5.0       # number of time steps for enoding
num_trials_ = 1000000  # number of batches to learn weights
train_display_ = 10    # How often to update training stats outputs
val_display_ = 100     # How often to update validation stats outputs
learning_rate_ = 0.01  # Learning rate for SGD

# Interactive session allows us to enter IPython for analysis
sess = tf.InteractiveSession()

## Input data
dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

## Placeholders
x = tf.placeholder(dtype=tf.float32, shape=[batch_, m_])  # Data
y = tf.placeholder(dtype=tf.float32, shape=[batch_, l_])  # Image labels
gamma = tf.placeholder(dtype=tf.float32) # Supervised loss tradeoff hyper-parameter

## Dynamic variables
z = tf.Variable(np.zeros([batch_, n_], dtype=np.float32), trainable=False)
norm_z = tf.truediv(z, tf.sqrt(tf.reduce_sum(tf.pow(z, 2.0))))

## Weight vectors
"""
Need to normalize weight matrices
per Rolfe & Lecun (2013)
 Magnitude of rows of E are bounded by 1.25/num_steps_
 Magnitude of columns of D are bounded by 1
 Magnitude of rows of C are bounded by 5
"""
# Encoding weights
E = tf.Variable(hf.normalize_rows(tf.truncated_normal([n_, m_], mean=0.0, stddev=np.sqrt(1.0),
    dtype=tf.float32), 1.25/num_steps_), trainable=True) 
# Decoding weights
D = tf.Variable(hf.normalize_cols(tf.truncated_normal([m_, n_], mean=0.0, stddev=np.sqrt(1.0), 
    dtype=tf.float32), 1.0), trainable=True) 
# Explaining away weights
S = tf.Variable(tf.truncated_normal([n_, n_], mean=0.0, stddev=np.sqrt(1.0),
    dtype=tf.float32), trainable=True) 
# Classification matrix
C = tf.Variable(hf.normalize_rows(tf.truncated_normal([l_, n_], mean=0.0, stddev=np.sqrt(1.0),
    dtype=tf.float32), 5.0), trainable=True) 
# Bias
b = tf.Variable(tf.truncated_normal([1, n_], mean=0.0, stddev=np.sqrt(1.0),
    dtype=tf.float32), trainable=True)  

## Discretized update rule: z(t+1) = eta * (x E^T + z(t) S^T - b)
z_update = eta_ * (\
    tf.matmul(x, tf.transpose(E)) + \
    tf.matmul(z, S) - \
    tf.matmul(tf.constant(np.ones([batch_, 1]), dtype=tf.float32), b))
zeros = tf.constant(np.zeros([int(shp) for shp in z.get_shape()], dtype=np.float32))  
dz = tf.select(tf.greater_equal(zeros, z_update), zeros, z_update) # dz = max(0, z_update)
step_z = tf.group(z.assign(dz))

## Network outputs
y_ = tf.nn.softmax(tf.matmul(norm_z, tf.transpose(C))) # label output
x_ = tf.matmul(z, tf.transpose(D))  # reconstruction

## Loss fucntions
cross_entropy_loss = -tf.reduce_sum(y * tf.log(y_)) # reduce_sum sums across all dim (images in minibatch, classes)
euclidean_loss = 0.5 * tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(x, x_), 2.0)))
sparse_loss = tf.reduce_sum(tf.abs(z))
unsupervised_loss = euclidean_loss + lambda_ * sparse_loss
supervised_loss = gamma_ * cross_entropy_loss
total_loss = unsupervised_loss + supervised_loss

## Weight update method
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(total_loss)

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
#with tf.Session() as sess:
sess.run(init_op)
for batch_idx in range(num_trials_):
    batch = dataset.train.next_batch(batch_)
    ## First find image code, z
    for t in range(int(num_steps_)):
        step_z.run({x:hf.normalize_image(batch[0])})
        if np.any(np.isnan(z.eval())):
            print("ERROR: inner loop: batch number %g, time step %g - some z values are nan."%(batch_idx, t))
            IPython.embed()

    ## Use converged z to compute loss & update weights
    train_step.run({x:hf.normalize_image(batch[0]), y:batch[1], gamma:0.0})
     
    """
    Need to re-normalize weight matrices
    per Rolfe & Lecun (2013)
     Magnitude of rows of E are bounded by 1.25/num_steps_
     Magnitude of columns of D are bounded by 1
     Magnitude of rows of C are bounded by 5
    """
    E = hf.normalize_rows(E, 1.25/num_steps_)
    D = hf.normalize_cols(D, 1.0)
    C = hf.normalize_rows(C, 5.0)
    
    if np.any(np.isnan(z.eval())):
        print("ERROR: outer loop: batch number %g, time step %g - some z values are nan."%(batch_idx, t))
        IPython.embed()

    z.assign(zeros)

    if batch_idx % train_display_ == 0:
        train_accuracy = accuracy.eval({x:hf.normalize_image(batch[0]), y:batch[1], z:z.eval().astype(np.float32)})
        print("Batch number %g out of %g"%(batch_idx, num_trials_))
        print("\ttrain accuracy:\t\t%g"%(train_accuracy))
        print("\teuclidean loss:\t\t%g"%(euclidean_loss.eval({x:hf.normalize_image(batch[0])})))
        print("\tsparse loss:\t\t%g"%(sparse_loss.eval({z:z.eval()})))
        print("\tcross-entropy loss:\t%g"%(cross_entropy_loss.eval({y:batch[1]})))

    if batch_idx % val_display_ == 0:
        e_prev_fig = hf.display_data(E.eval().reshape(n_, int(np.sqrt(m_)), int(np.sqrt(m_))),
            title='Encoding matrix at trial number '+str(batch_idx), prev_fig=e_prev_fig)
        d_prev_fig = hf.display_data(tf.transpose(D).eval().reshape(n_, int(np.sqrt(m_)), int(np.sqrt(m_))),
            title='Decoding matrix at trial number '+str(batch_idx), prev_fig=d_prev_fig)
        ## TODO: Should plot S - Identity
        s_prev_fig = hf.display_data(S.eval(),
            title='Explaining-away matrix at trial number '+str(batch_idx), prev_fig=s_prev_fig)
        c_prev_fig = hf.display_data(C.eval().reshape(l_, int(np.sqrt(n_)), int(np.sqrt(n_))),
            title='Classification matrix at trial number '+str(batch_idx), prev_fig=c_prev_fig)
        b_prev_fig = hf.display_data(b.eval().reshape(int(np.sqrt(n_)), int(np.sqrt(n_))),
            title='Bias at trial number '+str(batch_idx)+'\nEach pixel represents the bias for a neuron',
            prev_fig=b_prev_fig)
        batch = dataset.validation.next_batch(batch_)
        val_accuracy = accuracy.eval({x:hf.normalize_image(batch[0]), y:batch[1], z:z.eval().astype(np.float32)})
        print("---validation accuracy %g"%(val_accuracy))
        
print("Model has finished learning.")
IPython.embed()
