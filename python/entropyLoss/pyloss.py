import caffe
import numpy as np

class ActivityLossLayer(caffe.Layer):
   def setup(self, bottom, top):
      if len(bottom) != 1:
         raise Exception("Need one input to compute information conveyed.")
      self.beta_ = 1.0

   def reshape(self, bottom, top):
      top[0].reshape(1)
      self.q_dist = np.zeros_like(bottom[0].data, dtype=np.float32)

   def forward(self, bottom, top):
      self.q_dist = np.exp(-self.beta_ * bottom[0].data) / np.sum(np.exp(-self.beta_*bottom[0].data))
      num_batch = bottom[0].shape[0]
      num_nodes = bottom[0].shape[1]
      top[0] = np.sum(np.pow(np.sum(self.q_dist / num_batch, axis=0) - (1.0 / num_nodes), 2.0), axis=0)

   def backward(self, top, propagate_down, bottom):
      num_nodes = bottom[0].shape[1]
      num_batch = bottom[0].shape[0]
      temp1 = 2.0 * np.sum(np.sum(self.q_dist / num_batch, axis=0) - 1.0/num_nodes, axis=0)
      temp2 = -np.sum(np.exp(-self.beta_ * bottom[0].data), axis=0)
      bottom[0].diff[...] = np.mult(temp1, temp2)

class EntropyLossLayer(caffe.Layer):
   def setup(self, bottom, top):
      if len(bottom) != 1:
         raise Exception("Need one input to compute entropy.")
      #self.beta_ = layer_params['beta']
      self.beta_ = 1.0

   def reshape(self, bottom, top):
      # loss output is scalar
      # variables for storing distributions
      self.q_dist = np.zeros_like(bottom[0].data, dtype=np.float32)
      top[0].reshape(1)

   def forward(self, bottom, top):
      # First we compute the output distribution from input data
      self.q_dist = np.exp(-self.beta_ * bottom[0].data) / np.sum(np.exp(-self.beta_*bottom[0].data))

      # Now we compute entropy over q distribution
      top[0] = np.sum(np.mult(self.q_dist, np.log(self.q_dist)))

   def backward(self, top, propagate_down, bottom):
      temp = np.sum(np.mult(bottom[0].data, self.q_dist)) - bottom[0].data
      bottom[0].diff[...] = np.pow(self.beta_, 2.0) * np.mult(self.q_dist, temp)
