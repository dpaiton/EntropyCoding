import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
import IPython
from tensorflow.examples.tutorials.mnist import input_data

## User-defined parameters
n_    = 784         # Num_pixels
m_    = 484         # Num_elements
lamb_ = 0.10        # Threshold potential (sparseness penalty scaling)
dt_   = 0.001       # [s] discrete time constant
tau_  = 0.01        # [s] LCA time constant
lr_   = 0.10        # Learning rate for weight updates (will be divided by batch_)
batch_ = 60         # Number of images in a batch
num_steps_ = 20     # Number of steps to run LCA
num_trials_ = 5000  # Number of batches to learn weights
display_ = 10       # How often to display status updates
thresh_ = 'soft'    # Thresholding type for LCA -> can be 'hard' or 'soft'

tf.set_random_seed(1234567890)

## Helper functions
"""
Driving input for LCA model
    b_m = <s(t), phi_m>
"""
def b(s, phi):
    return tf.matmul(s, tf.transpose(phi))

"""
Lateral inhibition
    G_m,n = <phi_m, phi_n> - I
where I is the identity matrix and prevents a neuron from inhibiting itself
"""
def G(phi):
    return tf.matmul(phi, tf.transpose(phi)) - tf.constant(np.identity(int(phi.get_shape()[0])), dtype=tf.float32)

"""
Soft threshold function
T(u) = 0        for u <  lambda
       u-lambda for u >= lambda

Hard threshold function
T(u) = 0 for u <  lambda
       u for u >= lambda

TODO: Define as in eq 3.5 of Rozell et al. 2008 for generality
"""
def T(u, lamb, thresh_type='soft'):
    # tf.select(condition, a, e) generates a new variable with subsets of a and e, based on condition
    # select from a if condition is true; select from e if condition is false
    # here I assign a to be u-lambda and e to be a tensor of zeros, this will perform thresholding function
    if thresh_type is 'soft':
        return tf.select(tf.greater_equal(u, lamb), u-lamb, tf.constant(np.zeros([int(dim) for dim in u.get_shape()]), dtype=tf.float32))
    else:
        return tf.select(tf.greater_equal(u, lamb), u, tf.constant(np.zeros([int(dim) for dim in u.get_shape()]), dtype=tf.float32))

"""
Reconstruction of the input is a weighted sum of basis vectors
recon = <a, phi>
"""
def compute_recon(a, phi):
    return tf.matmul(a, phi)

## Interactive session allows us to enter IPython for analysis
sess = tf.InteractiveSession()

## Setup data
dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

## Setup constants & placeholders
s = tf.placeholder(tf.float32, shape=[None, n_]) # Placeholder for data
r = tf.placeholder(tf.float32, shape=[None, n_]) # Placeholder for reconstruction
eta = tf.placeholder(tf.float32, shape=())       # Placeholder for LCA update rate
weight_lr = tf.placeholder(tf.float32, shape=()) # Placeholder for Phi update rule

## Initialize membrane potential
u = tf.Variable(np.zeros([batch_, m_], dtype=np.float32))

## Initialize dictionary
# Truncated normal distribution is a standard normal distribution with specified mean and standard
# deviation, except that values whose magnitude is more than 2 standard deviations from the mean
# are dropped and re-picked.
phi_init_mean = 0.0
phi_init_var = 1.0
phi = tf.Variable(hf.l2_normalize_rows(tf.truncated_normal([m_, n_], mean=phi_init_mean,
    stddev=np.sqrt(phi_init_var), dtype=tf.float32)))

## Discritized membrane update rule
du = (1-eta) * u + eta * (b(s, phi) - tf.matmul(T(u, lamb_, thresh_type=thresh_), G(phi)))

## Discritized weight update rule
dphi = weight_lr * tf.matmul(tf.transpose(T(u, lamb_, thresh_type=thresh_)), 
    s - compute_recon(T(u, lamb_, thresh_type=thresh_), phi))

## Operation to update the state
step_lca = tf.group(u.assign(du))
step_phi = tf.group(phi.assign_add(dphi))

## Loss functions (for analysis)
s_ = compute_recon(T(u, lamb_, thresh_type=thresh_), phi)
euclidean_loss = 0.5 * tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(s, s_), 2.0)))
sparse_loss = lamb_ * tf.reduce_sum(tf.abs(T(u, lamb_, thresh_type=thresh_)))
unsupervised_loss = euclidean_loss + sparse_loss

saver = tf.train.Saver(var_list=[phi], max_to_keep=5, keep_checkpoint_every_n_hours=1, restore_sequentially=True)

tf.initialize_all_variables().run()

phi_prev_fig = None
recon_prev_fig = None

## Select images
for trial in range(num_trials_):
    norm_input = hf.normalize_image(dataset.train.next_batch(batch_)[0], divide_l2=False) # don't need labels

    ## Converge network
    for t in range(num_steps_):
        ## Step simulation
        step_lca.run({s:norm_input, eta:dt_/tau_})

    step_phi.run({s:norm_input, weight_lr:float(lr_)/float(batch_)})

    ## Renormalize weights after update
    phi = hf.l2_normalize_rows(phi)

    if trial % display_ == 0:
        sparsity = 100*np.count_nonzero(T(u, lamb_, thresh_).eval())/np.float32(np.size(T(u, lamb_, thresh_).eval()))
        print('Finished trial %g, max val of u is %g, num active of a was %g percent'%(trial, u.eval().max(), sparsity))
        print("\teuclidean loss:\t%g"%(euclidean_loss.eval({s:norm_input, u:u.eval(), phi:phi.eval()})))
        print("\tsparse loss:\t%g"%(sparse_loss.eval({u:u.eval()})))
        print("\ttotal loss:\t%g"%(unsupervised_loss.eval({s:norm_input, u:u.eval(), phi:phi.eval()})))
        r = compute_recon(T(u, lamb_, thresh_type=thresh_), phi)
        recon_prev_fig = hf.display_data(
            r.eval({u:u.eval()}).reshape(batch_, int(np.sqrt(n_)), int(np.sqrt(n_))),
            title='Reconstructions for time step '+str(t)+' in trial '+str(trial), prev_fig=recon_prev_fig)
        phi_prev_fig = hf.display_data(phi.eval().reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
            title='Dictionary for trial number '+str(trial), prev_fig=phi_prev_fig)
    if trial % 100 == 0:
        saver.save(sess, 'checkpoints/lca_checkpoint', global_step=trial)

IPython.embed()
sess.close()
