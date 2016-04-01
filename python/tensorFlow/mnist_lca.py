import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython

from tensorflow.examples.tutorials.mnist import input_data

## User-defined parameters
n_    = 784            # num_pixels
m_    = 500            # num_elements
lamb_ = 0.1            # threshold potential
dt_   = 0.001          # [s] discrete time constant
tau_  = 0.01           # [s] LCA time constant
lr_   = 0.001          # Learning rate for weight updates
batch_ = 60            # number of images in a batch
num_steps_ = 50        # number of steps to run LCA
num_trials_ = 5000     # number of batches to learn weights
thresh_ = 'hard'       # type of thresholding for LCA -> can be 'hard' or 'soft'
tf.set_random_seed(1234567890)

## Helper functions
def b(s, phi):
    """
    Driving input for LCA model
        b_m = <s(t), phi_m>
    """
    return tf.matmul(s, tf.transpose(phi))

def G(phi):
    """
    Lateral inhibition
        G_m,n = <phi_m, phi_n> - I
    where I is the identity matrix and prevents a neuron from inhibiting itself
    """
    return tf.matmul(phi, tf.transpose(phi)) - tf.constant(np.identity(int(phi.get_shape()[0])), dtype=tf.float32)

def T(u, lamb, thresh_type='soft'):
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
    if thresh_type is 'soft':
        return tf.select(tf.greater_equal(u, lamb), u-lamb, tf.constant(np.zeros([int(dim) for dim in u.get_shape()]), dtype=tf.float32))
    else:
        return tf.select(tf.greater_equal(u, lamb), u, tf.constant(np.zeros([int(dim) for dim in u.get_shape()]), dtype=tf.float32))

def normalize_image(img):
    """
    Normalize input image to have mean 0 and std 1

    expects input to be numpy ndarray
    """
    return np.vstack([(img[idx,:]-np.mean(img[idx,:]))/np.std(img[idx,:]) for idx in range(img.shape[0])])

def normalize_weights(weights):
    """
    Normalize the weights matrix
    """
    norm_mat = tf.diag(1.0/tf.sqrt(tf.reduce_sum(tf.pow(weights, 2.0), reduction_indices=0)))
    return tf.matmul(weights, norm_mat)

def compute_recon(a, phi):
    """
    Reconstruction of the input is a weighted sum of basis vectors
    recon = <a, phi>
    """
    return tf.matmul(a, phi)

def display_data(data, title='', prev_fig=None):
    """
    Display input data as an image

    data should be of shape (n, height, width) or (n, height, width, 3)
    """
    # normalize input
    data = (data - data.min()) / (data.max() - data.min())

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (1, 1), (1, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)

    data = np.pad(data, padding, mode='constant', constant_values=1)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    if prev_fig is None:
        fig_no, sub_axis = plt.subplots(1)
        axis_image = sub_axis.imshow(data, cmap='Greys', interpolation='nearest')
    else:
        (fig_no, sub_axis, axis_image) = prev_fig
        axis_image.set_data(data)

    for axis in fig_no.axes:
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)

    fig_no.suptitle(title)
    if prev_fig is None:
        fig_no.show()
    else:
        fig_no.canvas.draw()

    return (fig_no, sub_axis, axis_image)

## TODO: It would be good to write out some more analysis functions
#def compute_recon_error(s, recon):
#    """
#    Returns the l_2 distance between s and recon
#    inputs:
#        s - numpy ndarray scaled from 0 to 1 of dim p_h x p_w
#        recon - numpy ndarray scaled from 0 to 1 of dim p_h x p_w
#    """
#    return np.sqrt(np.sum(s
#
#def compute_energy(phi, a, p=0):
#    """
#    LCA network seeks to minimize the following energy function:
#        E = 1/2 ||s - <a,phi>||_2^2 + lambda * |a|_p
#        where s is the input image, <a,phi> is the reconstruction,
#        lambda is the sparsity tradeoff parameter, and |.|_p represents
#        the l_p norm
#
#    current allowed values for p are p={0, 1}
#    """
#    return tf.matmul(a, phi)


## Interactive session allows us to enter IPython for analysis
sess = tf.InteractiveSession()

## Setup data
dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

## Setup constants & placeholders
s = tf.placeholder(tf.float32, shape=[None, n_]) # Placeholder for data
r = tf.placeholder(tf.float32, shape=[None, n_]) # Placeholder for reconstruction
eta = tf.placeholder(tf.float32, shape=())       # Placeholder for LCA update rate
lamb = tf.placeholder(tf.float32, shape=())      # Placeholder for threshold parameter lambda
thresh = tf.placeholder(tf.string, shape=())    # Placeholder for threshold type
lr = tf.placeholder(tf.float32, shape=())        # Placeholder for weight learning rate

## Initialize membrane potential
u = tf.Variable(np.zeros([batch_, m_], dtype=np.float32))

## Initialize dictionary
# Truncated normal distribution is a standard normal distribution with specified mean and standard
# deviation, except that values whose magnitude is more than 2 standard deviations from the mean
# are dropped and re-picked.
phi_init_mean = 0.0
phi_init_var = 1.0
phi = tf.Variable(normalize_weights(tf.truncated_normal([m_, n_], mean=phi_init_mean,
    stddev=np.sqrt(phi_init_var), dtype=tf.float32)))

## Discritized membrane update rule
du = (1 - eta) * u + eta * (b(s, phi) - tf.matmul(T(u, lamb, thresh_type=thresh), G(phi)))

## Discritized weight update rule
dphi = normalize_weights(phi +
    lr * tf.matmul(tf.transpose(T(u, lamb, thresh_type=thresh)),
    s - compute_recon(T(u, lamb, thresh_type=thresh), phi)))

## Operation to update the state
step_lca = tf.group(u.assign(du))
step_phi = tf.group(phi.assign(dphi))

tf.initialize_all_variables().run()

phi_prev_fig = None
recon_prev_fig = None
## Select images
for trial in range(num_trials_):
    batch = dataset.train.next_batch(batch_)

    ## Converge network
    for t in range(num_steps_):
        ## Step simulation
        step_lca.run({s:normalize_image(batch[0]), eta:dt_/tau_, lamb:lamb_, thresh:thresh_})

    step_phi.run({s:normalize_image(batch[0]), lr:lr_, lamb:lamb_, thresh:thresh_})
    if trial % 2 == 0:
        sparsity = 100*np.count_nonzero(T(u, lamb_, thresh_).eval())/np.float32(np.size(T(u, lamb_, thresh_).eval()))
        print('Finished trial %g, max val of u is %g, num active of a was %g percent'%(trial, u.eval().max(), sparsity))
        r = compute_recon(T(u, lamb, thresh_type=thresh), phi)
        recon_prev_fig = display_data(
            r.eval({u:u.eval(), lamb:lamb_, thresh:thresh_}).reshape(batch_, int(np.sqrt(n_)), int(np.sqrt(n_))),
            title='Reconstructions for time step '+str(t)+' in trial '+str(trial), prev_fig=recon_prev_fig)
        phi_prev_fig = display_data(phi.eval().reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
            title='Dictionary for trial number '+str(trial), prev_fig=phi_prev_fig)
    #IPython.embed()

#activity = u.eval()
IPython.embed()
