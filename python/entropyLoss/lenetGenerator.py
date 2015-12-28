import caffe
import numpy as np
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

def create_lenet(lmdb, batch_size):
   # Caffe's version of LeNet: http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb
   net = caffe.NetSpec()
   net.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, \
                                   transform_param=dict(scale=1./255), ntop=2)
   net.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
   net.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
   net.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
   net.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
   net.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
   net.relu1 = L.ReLU(n.ip1, in_place=True)
   net.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
   net.loss = L.SoftmaxWithLoss(n.ip2, n.label)
   return net

source = "/Users/slundquist/workspace/caffe/examples/mnist/mnist_train_lmdb"
net = create_lenet(source, 64)
