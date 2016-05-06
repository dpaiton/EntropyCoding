import os
import numpy as np
import helper_functions as hf
import IPython

import tensorflow as tf # always import tensorflow after other imports
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

# ADAM parameters
beta_1_ = 0.9
beta_2_ = 0.999
epsilon_ = 1e-7

# Display & Checkpointing
checkpoint_ = 1000
train_display_ = 2  # How often to display status updates
val_display_ = 100
display_plots_ = False
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
                u-lamb, tf.constant(np.zeros([int(dim) for dim in u.get_shape()]), dtype=tf.float32))
    else:
        return tf.select(tf.greater_equal(u, lamb),
                u, tf.constant(np.zeros([int(dim) for dim in u.get_shape()]), dtype=tf.float32))

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
    u = tf.Variable(np.zeros([m_, batch_], dtype=np.float32), trainable=False, name="membrane_potential")

with tf.name_scope("weights") as scope:
    ## Initialize dictionary
    # Truncated normal distribution is a standard normal distribution with specified mean and standard
    # deviation, except that values whose magnitude is more than 2 standard deviations from the mean
    # are dropped and re-picked.
    weight_init_mean = 0.0
    weight_init_var = 1.0
    phi = tf.Variable(tf.truncated_normal([n_, m_], mean=weight_init_mean,
        stddev=np.sqrt(weight_init_var), dtype=tf.float32, name="phi_init"), trainable=True, name="phi")

    C = tf.Variable(tf.truncated_normal([l_, m_], mean=weight_init_mean,
        stddev=np.sqrt(weight_init_var), dtype=tf.float32, name="C_init"), trainable=True, name="C")

with tf.name_scope("normalize_weights") as scope:
    norm_phi = phi.assign(tf.nn.l2_normalize(phi, dim=1, epsilon=eps, name="row_l2_norm"))
    norm_C = C.assign(tf.nn.l2_normalize(C, dim=1, epsilon=eps, name="row_l2_norm"))
    normalize_weights = tf.group(norm_phi, norm_C, name="do_normalization")

with tf.name_scope("output") as scope:
    with tf.name_scope("image_estimate"):
        s_ = compute_recon(phi, T(u, lamb, thresh_type=thresh_))
    with tf.name_scope("label_estimate"):
        y_ = tf.nn.softmax(tf.matmul(C, tf.nn.l2_normalize(u, dim=0, epsilon=1e-12,
            name="col_l2_norm"), name="classify"), name="softmax")

with tf.name_scope("update_u") as scope:
    ## Discritized membrane update rule
    du = (1 - eta) * u + eta * tf.sub(b(phi, s), tf.matmul(G(phi), T(u, lamb, thresh_type=thresh_)))

    ## Operation to update the state
    step_lca = tf.group(u.assign(du), name="do_update_u")

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
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Load in scheduler
schedules = scheduler.schedule().blocks

## Weight update method
var_lists = [[phi] if sch["prefix"] == "unsupervised" else [C] if sch["prefix"] == "supervised" else [phi, C] for sch in schedules]
train_weights = [tf.train.AdamOptimizer(lr, beta_1_, beta_2_, epsilon_,
    name="adam_optimizer").minimize(total_loss, var_list=var_lists[sch_no],
    name="adam_minimzer") for sch_no in range(len(schedules))]

## Checkpointing & graph output
if checkpoint_ != -1:
    var_list={"phi":phi, "C":C}
    saver = tf.train.Saver(var_list)
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

## Initialization
init_op = tf.initialize_all_variables()

c_prev_fig = None
phi_prev_fig = None
recon_prev_fig = None
with tf.Session() as sess:
    with tf.device(device_):
        sess.run(init_op)

        global_batch_timer = 0
        for sched_no, schedule in enumerate(schedules):
            if checkpoint_ != -1 and global_batch_timer == 0:
                tf.train.write_graph(sess.graph_def, "checkpoints", "lca_gradient_graph.pb", as_text=False)
                tf.train.SummaryWriter("checkpoints", sess.graph)

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

                ## Normalize weights
                normalize_weights.run()

                ## Converge network
                for t in range(num_steps_):
                    step_lca.run({s:input_image, eta:dt_/tau_, lamb:lambda_})

                train_weights[sched_no].run({\
                    s:input_image,
                    y:input_label,
                    lr:learning_rate_,
                    lamb:lambda_,
                    gamma:gamma_})

                if trial % train_display_ == 0:
                    sparsity = 100 * np.count_nonzero(T(u, lamb, thresh_).eval({lamb:lambda_})) / (m_ * batch_)
                    train_accuracy = accuracy.eval({s:input_image, y:input_label})
                    print("\nGlobal batch number is %g"%global_batch_timer)
                    print("Finished trial %g out of %g, max val of u is %g, num active of a was %g percent"%(trial, 
                        num_batches_, u.eval().max(), sparsity))
                    print("\teuclidean loss:\t\t%g"%(euclidean_loss.eval({s:input_image, lamb:lambda_})))
                    print("\tsparse loss:\t\t%g"%(sparse_loss.eval({s:input_image, lamb:lambda_})))
                    print("\tunsupervised loss:\t%g"%(unsupervised_loss.eval({s:input_image, lamb:lambda_})))
                    print("\tsupervised loss:\t%g"%(supervised_loss.eval({s:input_image,
                        y:input_label, gamma:gamma_, lamb:lambda_})))
                    print("\ttrain accuracy:\t\t%g"%(train_accuracy))
                    if display_plots_: #TODO: plot weight gradients
                        c_prev_fig = hf.display_data_tiled(C.eval().reshape(l_, int(np.sqrt(m_)), int(np.sqrt(m_))),
                            title="Classification matrix at trial number "+str(global_batch_timer), prev_fig=c_prev_fig)
                        recon_prev_fig = hf.display_data_tiled(
                            tf.transpose(s_).eval({lamb:lambda_}).reshape(batch_, int(np.sqrt(n_)), int(np.sqrt(n_))),
                            title="Reconstructions in trial "+str(global_batch_timer), prev_fig=recon_prev_fig)
                        phi_prev_fig = hf.display_data_tiled(tf.transpose(phi).eval().reshape(m_, int(np.sqrt(n_)), int(np.sqrt(n_))),
                            title="Dictionary for trial "+str(global_batch_timer), prev_fig=phi_prev_fig)
                    if val_display_ != -1 and global_batch_timer % val_display_ == 0:
                        val_batch = dataset.validation.next_batch(5000) # Full validation set
                        val_image = hf.normalize_image(val_batch[0]).T
                        val_label = val_batch[1].T
                        val_accuracy = accuracy.eval({s:val_image, y:val_label})
                        print("\t---validation accuracy: %g"%(val_accuracy))
                if trial % checkpoint_ == 0:
                    saver.save(sess, "./checkpoints/lca_checkpoint", global_step=global_batch_timer)

                global_batch_timer += 1


        IPython.embed()
