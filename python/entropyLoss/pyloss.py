import caffe
import numpy as np
import IPython

class EntropyCode(caffe.Layer):
   def setup(self, bottom, top):
      if len(bottom) != 1:
         raise Exception("Need one input to compute annealing.")
      if len(top) != 1:
         raise Exception("Need one output to compute annealing.")
      #self.beta_ = layer_params['beta']

   def reshape(self, bottom, top):
      if(len(np.shape(bottom[0].data)) != 2):
         raise Exception("Bottom data must have a heigth and width of 1")
      top[0].reshape(*bottom[0].data.shape)

   def forward(self, bottom, top):
      #top[0].data[...] = np.copy(bottom[0].data)
      #IPython.embed()
      #Summing over batches, batchSums is a vector of num batches
      #batchSums = np.sum(np.exp(bottom[0].data), 1)

      #Maximize peak
      top[0].data[...] = (1/np.exp(-bottom[0].data))

      #numBatch = np.shape(bottom[0].data)[0]
      #for i in range(numBatch):
      #   #Set value to push each other away
      #   top[0].data[i, ...] = peakVal[i, ...] * (bottom[0].data[i, ...] - np.log(batchSums[i]))
      #   #Normalize peak val
      #   #top[0].data[i, ... ]= (outVal - np.min(bottom[0].data))/(np.max(bottom[0].data) - np.min(bottom[0].data))

   def backward(self, top, propagate_down, bottom):
      bottom[0].diff[...] = np.copy(top[0].diff)
      pass
