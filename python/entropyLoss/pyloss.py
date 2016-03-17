import yaml
import caffe
import IPython
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

def plotIndiv(data, title_sup, num_imgs, prevFig=None):
    if prevFig[0] is None:
        figNo, figSubAxes = plt.subplots(np.ceil(np.sqrt(data.shape[1])).astype(np.int32), np.floor(np.sqrt(data.shape[1])).astype(np.int32))
        plt.suptitle(title_sup+' individual node activations')
        barIds = []
        for i in range(len(figSubAxes.flat)):
            if i in range(data.shape[1]):
                barIds.append(figSubAxes.flat[i].bar(range(0, num_imgs), data[0:num_imgs,i]))
            figSubAxes.flat[i].tick_params(
                axis='both',       # changes apply to the x-axis and y-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                left='off',        # ticks along the left edge are off
                right='off',       # ticks along the righ tedge are off
                labelbottom='off', # labels along the bottom edge are off
                labelleft='off')   # labels along the left edge are off
            figSubAxes.flat[i].tick_params(
                axis='y',          # changes apply to the x-axis and y-axis
                labelleft='on',    # labels along the left edge are off
                labelsize='8')     # font size of labels
    else:
        (figNo, figSubAxes, barIds) = prevFig
        for i in range(len(figSubAxes.flat)):
            if i in range(data.shape[1]):
                for barId, h in zip(barIds[i], data[0:num_imgs, i]):
                    barId.set_height(h)
    if prevFig[0] is None:
        figNo.show()
    else:
        figNo.canvas.draw()
    return (figNo, figSubAxes, barIds)

def plotBatch(data, title_sup, prevFig=None):
    if prevFig[0] is None:
        figNo, figSubAxes = plt.subplots(1,1)
    else:
        (figNo, figSubAxes) = prevFig
    figSubAxes.bar(range(0, data.shape[1]), np.mean(data, axis=0))
    figSubAxes.set_title(title_sup+' batch average activation')
    if prevFig[0] is None:
        figNo.show()
    else:
        figNo.canvas.draw()
    return (figNo, figSubAxes)

class entropyLossLayer(caffe.Layer):
   def setup(self, bottom, top):
      layer_params = yaml.load(self.param_str)
      self.beta_ = np.float(layer_params['beta'])
      self.alpha_ = np.float(layer_params['alpha'])
      self.lamb_ = np.float(layer_params['lambda'])
      self.figNo = [None, None, None] # for debugging
      self.figSubAxes = [None, None, None] # for debugging
      self.barId = [None, None] # for debugging

   def reshape(self, bottom, top):
      #inputs
      self.x = T.matrix('float32')
      self.beta = T.scalar('float32')
      self.alpha = T.scalar('float32') # for debugging
      self.lamb = T.scalar('float32')

      #probabilities
      self.q = T.nnet.softmax(-self.beta * (self.x - self.x.max()))
      self.p = T.nnet.softmax(-self.beta * (self.x.T - self.x.max())).T

      #forward cost function
      self.cost = -self.alpha * (self.q * T.log(self.q)).sum() + self.lamb * (self.p * T.log(self.p)).sum()
      self.forward_fn = theano.function(inputs=[self.x, self.beta, self.alpha, self.lamb], outputs=[self.cost, self.q, self.p])

      #gradient function
      self.dcost_dx = T.grad(self.cost, wrt=self.x)
      self.grad_fn = theano.function(inputs=[self.x, self.beta, self.alpha, self.lamb], outputs=self.dcost_dx)

      # loss output is scalar
      top[0].reshape(1)

   def forward(self, bottom, top):
      fwd_out = self.forward_fn(bottom[0].data, self.beta_, self.alpha_, self.lamb_) 
      top[0].data[...] = fwd_out[0]
      #(self.figNo[0], self.figSubAxes[0], self.barId[0]) = plotIndiv(bottom[0].data, 'data', 15, (self.figNo[0], self.figSubAxes[0], self.barId[0]))
      #(self.figNo[1], self.figSubAxes[1], self.barId[1]) = plotIndiv(fwd_out[1], 'q prob', 15, (self.figNo[1], self.figSubAxes[1], self.barId[1]))
      #(self.figNo[2], self.figSubAxes[2]) = plotBatch(fwd_out[2], 'p prob', (self.figNo[2], self.figSubAxes[2]))
      #print '-------\ntot loss: ' + str(fwd_out[0])
      #print 'q loss: ' + str(self.forward_fn(bottom[0].data, self.beta_, self.alpha_*1.0, 0.0)[0])
      #print 'p loss: ' + str(self.forward_fn(bottom[0].data, self.beta_, 0.0, self.lamb_*1.0)[0])

   def backward(self, top, propagate_down, bottom):
      bottom[0].diff[...] = self.grad_fn(bottom[0].data, self.beta_, self.lamb_, self.alpha_)
      #print 'grad max: ' + str(np.max(np.abs(bottom[0].diff[...])))
