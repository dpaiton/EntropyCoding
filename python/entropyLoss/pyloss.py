import caffe
import numpy as np
import theano
import theano.tensor as T
import IPython

class BatchEntropyLossLayer(caffe.Layer):
   def setup(self, bottom, top):
      #self.beta_ = layer_params['beta']
      self.beta_ = -1.0

   def reshape(self, bottom, top):
      # loss output is scalar
      # variables for storing distributions
      self.stable_bottom = np.zeros_like(bottom[0].data, dtype=np.float32)
      self.x = T.matrix('float32')
      self.beta = T.scalar('float32')
      self.p = T.nnet.softmax(-self.beta * self.x.T)
      self.cost = (self.p * T.log(self.p)).sum()
      self.forward_fn = theano.function(inputs=[self.x, self.beta], outputs=self.cost)
      self.dcost_dx = T.grad(self.cost, wrt=self.x)
      self.grad_fn = theano.function(inputs=[self.x, self.beta], outputs=self.dcost_dx)
      top[0].reshape(1)

   def forward(self, bottom, top):
      self.stable_bottom = bottom[0].data - np.max(bottom[0].data)
      top[0].data[...] = self.forward_fn(self.stable_bottom, self.beta_)[()] # have to cast 0-d array to scalar
      print 'batch loss: ' + str(top[0].data[...])

   def backward(self, top, propagate_down, bottom):
      bottom[0].diff[...] = self.grad_fn(bottom[0].data, self.beta_)
      #print 'batch grad mean: ' + str(np.mean(bottom[0].diff[...]))
      print 'batch grad max: ' + str(np.max(np.abs(bottom[0].diff[...])))

class NodeEntropyLossLayer(caffe.Layer):
   def setup(self, bottom, top):
      #self.beta_ = layer_params['beta']
      self.beta_ = -1.0

   def reshape(self, bottom, top):
      # loss output is scalar
      # variables for storing distributions
      self.stable_bottom = np.zeros_like(bottom[0].data, dtype=np.float32)
      self.x = T.matrix('float32')
      self.beta = T.scalar('float32')
      self.q = T.nnet.softmax(-self.beta * self.x)
      self.cost = -(self.q * T.log(self.q)).sum()
      self.forward_fn = theano.function(inputs=[self.x, self.beta], outputs=self.cost)
      self.dcost_dx = T.grad(self.cost, wrt=self.x)
      self.grad_fn = theano.function(inputs=[self.x, self.beta], outputs=self.dcost_dx)
      top[0].reshape(1)

   def forward(self, bottom, top):
      self.stable_bottom = bottom[0].data - np.max(bottom[0].data)
      top[0].data[...] = self.forward_fn(self.stable_bottom, self.beta_)[()] # have to cast 0-d array to scalar
      print 'node loss: ' + str(top[0].data[...])

   def backward(self, top, propagate_down, bottom):
      bottom[0].diff[...] = self.grad_fn(bottom[0].data, self.beta_)
      #print 'node grad mean: ' + str(np.mean(bottom[0].diff[...]))
      print 'node grad max: ' + str(np.max(np.abs(bottom[0].diff[...])))
