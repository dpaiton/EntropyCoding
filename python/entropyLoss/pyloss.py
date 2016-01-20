import caffe
import numpy as np
import IPython

class ActivityLossLayer(caffe.Layer):
   def setup(self, bottom, top):
      #self.beta_ = layer_params['beta']
      self.beta_ = 1.0

   def reshape(self, bottom, top):
      self.q_dist = np.zeros_like(bottom[0].data, dtype=np.float32)
      top[0].reshape(1)

   def forward(self, bottom, top):
      num_batch = np.float(bottom[0].shape[0])
      num_nodes = np.float(bottom[0].shape[1])
      self.q_dist = np.exp(-self.beta_ * bottom[0].data) / np.sum(np.exp(-self.beta_ * bottom[0].data))
      top[0].data[...] = np.sum(np.sum(np.log(num_nodes / num_batch * self.q_dist)))

   def backward(self, top, propagate_down, bottom):
      num_batch = np.float(bottom[0].shape[0])
      num_nodes = np.float(bottom[0].shape[1])
      bottom[0].diff[...] = np.sum(self.beta_ * (self.q_dist - 1.0), axis=0)

class EntropyLossLayer(caffe.Layer):
   def setup(self, bottom, top):
      #self.beta_ = layer_params['beta']
      self.beta_ = 1.0

   def reshape(self, bottom, top):
      # loss output is scalar
      # variables for storing distributions
      self.q_dist = np.zeros_like(bottom[0].data, dtype=np.float32)
      top[0].reshape(1)

   def forward(self, bottom, top):
      # First we compute the output distribution from input data
      self.q_dist = np.exp(-self.beta_ * bottom[0].data) / np.sum(np.exp(-self.beta_ * bottom[0].data))
      # Now we compute entropy over q distribution
      top[0].data[...] = -np.sum(np.sum(np.multiply(self.q_dist, np.log(self.q_dist)), axis=1), axis=0)

   def backward(self, top, propagate_down, bottom):
      num_nodes = np.int(bottom[0].shape[1])
      temp = np.array([np.sum(np.multiply(bottom[0].data, self.q_dist), axis=1),]*num_nodes).transpose() - bottom[0].data
      bottom[0].diff[...] = np.power(self.beta_, 2.0) * np.multiply(self.q_dist, temp)
