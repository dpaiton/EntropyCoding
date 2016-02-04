import caffe
import numpy as np
import theano
import theano.tensor as T
import yaml
import IPython

class entropyLossLayer(caffe.Layer):
   def setup(self, bottom, top):
      layer_params = yaml.load(self.param_str)
      self.beta_ = np.float(layer_params['beta'])
      self.lamb_ = np.float(layer_params['lamb'])
      self.alpha_ = 1.0 # for debugging

   def reshape(self, bottom, top):
      #inputs
      self.x = T.matrix('float32')
      self.beta = T.scalar('float32')
      self.lamb = T.scalar('float32')
      self.alpha = T.scalar('float32') # for debugging

      #probabilities
      self.p = T.nnet.softmax(-self.beta * (self.x.T - self.x.max()))
      self.q = T.nnet.softmax(-self.beta * (self.x - self.x.max()))

      #forward cost function
      self.cost = -self.alpha * (self.q * T.log(self.q)).sum() + self.lamb * (self.p * T.log(self.p)).sum()
      self.forward_fn = theano.function(inputs=[self.x, self.beta, self.lamb, self.alpha], outputs=self.cost)

      #gradient function
      self.dcost_dx = T.grad(self.cost, wrt=self.x)
      self.grad_fn = theano.function(inputs=[self.x, self.beta, self.lamb, self.alpha], outputs=self.dcost_dx)

      # loss output is scalar
      top[0].reshape(1)

   def forward(self, bottom, top):
      top[0].data[...] = self.forward_fn(bottom[0].data, self.beta_, self.lamb_, self.alpha_)[()] # have to cast 0-d array to scalar
      print '-------\ntot loss: ' + str(top[0].data[...])
      print 'q loss: ' + str(self.forward_fn(bottom[0].data, self.beta_, 0.0, 1.0)[()])
      print 'p loss: ' + str(self.forward_fn(bottom[0].data, self.beta_, 1.0, 0.0)[()])

   def backward(self, top, propagate_down, bottom):
      bottom[0].diff[...] = self.grad_fn(bottom[0].data, self.beta_, self.lamb_, self.alpha_)
      print 'grad max: ' + str(np.max(np.abs(bottom[0].diff[...])))
