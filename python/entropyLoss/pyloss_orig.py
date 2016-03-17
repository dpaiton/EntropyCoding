import caffe
import numpy as np
import theano
import theano.tensor as T
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
      num_batch = np.int(bottom[0].shape[0])
      bottom[0].diff[...] = np.array([np.sum(self.beta_ * (self.q_dist - 1.0), axis=0),]*num_batch)

class BatchEntropyLossLayer(caffe.Layer):
   def setup(self, bottom, top):
      #self.beta_ = layer_params['beta']
      self.beta_ = -1.0

   def reshape(self, bottom, top):
      # loss output is scalar
      # variables for storing distributions
      self.p_dist = np.zeros_like(bottom[0].data, dtype=np.float32)
      self.fixed_bottom = np.zeros_like(bottom[0].data, dtype=np.float32)
      top[0].reshape(1)

   def forward(self, bottom, top):
      self.fixed_bottom = bottom[0].data - np.max(bottom[0].data)
      self.p_dist = np.multiply(np.exp(-self.beta_ * self.fixed_bottom), 1.0/np.array([np.sum(np.exp(-self.beta_ * self.fixed_bottom), axis=0),]*bottom[0].data.shape[0]))
      top[0].data[...] = np.sum(np.sum(np.multiply(self.p_dist, np.log(self.p_dist)), axis=0))
      print 'batch loss: ' + str(top[0].data[...])

   def backward(self, top, propagate_down, bottom):
      bottom[0].diff[...] = -np.power(self.beta_,2.0) * np.dot(self.p_dist, np.dot(bottom[0].data.transpose(), self.p_dist)) - np.multiply(self.p_dist, bottom[0].data)
      #print 'batch grad mean: ' + str(np.mean(bottom[0].diff[...]))
      #print 'batch grad min: ' + str(np.min(bottom[0].diff[...]))
      #print 'batch grad max: ' + str(np.max(bottom[0].diff[...]))

class NodeEntropyLossLayer(caffe.Layer):
   def setup(self, bottom, top):
      #self.beta_ = layer_params['beta']
      self.beta_ = -1.0

   def reshape(self, bottom, top):
      # loss output is scalar
      # variables for storing distributions
      self.q_dist = np.zeros_like(bottom[0].data, dtype=np.float32)
      self.fixed_bottom = np.zeros_like(bottom[0].data, dtype=np.float32)
      top[0].reshape(1)

   def forward(self, bottom, top):
      self.fixed_bottom = bottom[0].data - np.max(bottom[0].data)
      self.q_dist = np.multiply(np.exp(-self.beta_ * self.fixed_bottom), 1.0/np.array([np.sum(np.exp(-self.beta_ * self.fixed_bottom), axis=1),]*bottom[0].data.shape[1]).transpose())
      #self.q_dist = np.multiply(np.exp(-self.beta_ * bottom[0].data), 1.0/np.array([np.sum(np.exp(-self.beta_ * bottom[0].data), axis=1),]*bottom[0].data.shape[1]).transpose())
      top[0].data[...] = np.sum(-np.sum(np.multiply(self.q_dist, np.log(self.q_dist)), axis=1))
      print 'node loss: ' + str(top[0].data[...])
      print 'bottom max: ' + str(np.max(bottom[0].data))
      print 'bottom min: ' + str(np.min(bottom[0].data))

   def backward(self, top, propagate_down, bottom):
      bottom[0].diff[...] = np.power(self.beta_, 2.0) * np.dot(self.q_dist, np.dot(bottom[0].data.transpose(), self.q_dist)) - np.multiply(self.q_dist, bottom[0].data)
      #print 'node grad mean: ' + str(np.mean(bottom[0].diff[...]))
      #print 'node grad min: ' + str(np.min(bottom[0].diff[...]))
      #print 'node grad max: ' + str(np.max(bottom[0].diff[...]))
