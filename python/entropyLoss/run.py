import caffe
import numpy as np
import matplotlib.pyplot as plt
import IPython
import sys

sys.path.insert(0,"./python/entropyLoss/")

#caffeDir = "/Users/slundquist/workspace/caffe/"
#modelDIr = "/Users/slundquist/workspace/caffe/"

caffeDir = "/Users/dpaiton/Work/Libraries/caffe/"
modelDir = "./models/entropy/"

#net = caffe.Net("caffenet.prototxt", caffe.TRAIN)
#solver = caffe.SGDSolver(modelDir + "solver.prototxt")
solver = caffe.SGDSolver(modelDir + "drsae_solver.prototxt")
net = solver.net
#solver.net.copy_from(modelDir  + "models/entropy/output/lenet_iter_10000.caffemodel")

solver.solve()

IPython.embed()
