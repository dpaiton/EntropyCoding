import numpy as np
import tensorflow as tf
import helper_functions as hf

# Get data first
# mnist.train has 55,000 examples
# mnist.test has 10,000 examples
# mnist.validation has 5,000 examples
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Input is a symbolic variable of type float32
# first dim is any length, second dim is num_pixels
x = tf.placeholder(tf.float32, [None, 784])

# Initialize weights & biases as a tensor of zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Perform softmax regression
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Placeholder for the correct labels
y_ = tf.placeholder(tf.float32, [None, 10])

# Cross-entropy loss function
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Train using gradient descent with a learning rate of 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Check all precictions against labels
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Determine training accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init = tf.initialize_all_variables()

w_prev_fig = None
with tf.Session() as sess: # this way we don't have to close the session
    with tf.device('/cpu:0'): # specify hardware, could be cpu:0, gpu:0, gpu:1, etc
        sess.run(init)

        # Read in all training data & train model
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
                print("step %d, training accuracy %g"%(i, train_accuracy))
                w_prev_fig = hf.display_data_tiled(tf.transpose(W).eval().reshape(10, np.sqrt(784), np.sqrt(784)), title="W", prev_fig=w_prev_fig)

        # Determine test accuracy
        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})

        print("Test accuracy = %g"%test_accuracy)

        import IPython
        IPython.embed()
