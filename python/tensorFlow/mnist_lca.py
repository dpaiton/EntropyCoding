import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython

from tensorflow.examples.tutorials.mnist import input_data

## User-defined parameters
m_    = 1568        # num_elements
n_    = 784         # num_pixels
lamb_ = 0.025       # threshold potential
tau_  = 0.001       # [s] LCA time constant
batch_ = 1          # number of images in a batch
num_steps_ = 100    # number of steps to run LCA
tf.set_random_seed(1234567890)

## Helper functions
def normalize(phi):
    """
    Normalize the phi matrix
    """
    norm_mat = tf.diag(1.0/tf.sqrt(tf.reduce_sum(tf.pow(phi, 2.0), reduction_indices=0)))
    return tf.matmul(phi, norm_mat)

def b(s, phi):
    """
    Driving input for LCA model
        b_m = <phi_m, s(t)>
    """
    return tf.matmul(s, tf.transpose(phi))

def G(phi):
    """
    Lateral inhibition
        G_m,n = <phi_m, phi_n> - I
    where I is the identity matrix and prevents a neuron from inhibiting itself
    """
    return tf.matmul(phi, tf.transpose(phi)) - tf.constant(np.identity(int(phi.get_shape()[0])), dtype=tf.float32)

def T(u, lamb):
    """
    Soft threshold function
    T(u) = 0        for u <  lambda
           u-lambda for u >= lambda
    """
    # tf.select(condition, a, e) generates a new variable with subsets of a and e, based on condition
    # if condition is true, select from a; if condition is false, select from e
    # here I assign a to be u-lambda and e to be a tensor of zeros, this will perform thresholding function
    return tf.select(tf.less(u, 0), u-lamb, tf.constant(np.zeros([int(dim) for dim in u.get_shape()]), dtype=tf.float32))

## Interactive session allows us to enter IPython for analysis
sess = tf.InteractiveSession()

## Setup data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## Setup constants & placeholders
s = tf.placeholder(tf.float32, shape=[None, n_]) # Placeholder for data
tau = tf.placeholder(tf.float32, shape=())       # Placeholder for time constant
lamb = tf.placeholder(tf.float32, shape=())      # Placeholder for threshold parameter lambda

## Initialize dictionary
# Truncated normal distribution is a standard normal distribution with specified mean and standard
# deviation, except that values whose magnitude is more than 2 standard deviations from the mean
# are dropped and re-picked.
phi_init_mean = 0.0
phi_init_var = 1.0
phi = normalize(tf.Variable(tf.truncated_normal([m_, n_], mean=phi_init_mean,
    stddev=np.sqrt(phi_init_var), dtype=tf.float32)))

## Initialize membrane potential
u = tf.Variable(np.zeros([batch_, m_], dtype=np.float32))

## Discritized membrane update rule
ut = (1.0/tau) * (b(s, phi) - u - tf.matmul(T(u,lamb), G(phi)))

## Operation to update the state
step = tf.group(u.assign(ut))

tf.initialize_all_variables().run()

## Select images TODO: Should happen in loop
batch = mnist.train.next_batch(batch_)

(figNo, subAxes) = plt.subplots(1)
figNo.show()

## Converge network
for t in range(num_steps_):
    # Step simulation
    step.run({s: batch[0], tau: tau_, lamb: lamb_})
    if t % 5 == 0:
        print("time step %g"%t)
        subAxes.plot(u.eval().T, color='b')
        figNo.canvas.draw()

activity = u.eval()

IPython.embed()
