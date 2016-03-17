import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import caffe
import lmdb
import sys
import os
import IPython

def load_lmdb(input_file):
    env = lmdb.open(input_file, readonly=True)
    datum = caffe.proto.caffe_pb2.Datum()
    x = []
    y = []
    with env.begin() as txn:
        for key, value in txn.cursor():
            datum.ParseFromString(value)
            flat_x = np.array(datum.float_data)
            if flat_x.size == 0:
                x.append(np.fromstring(datum.data, dtype=np.uint8).reshape(datum.channels, datum.height, datum.width))
                y.append(datum.label)
            else:
                x.append(flat_x.reshape(datum.channels, datum.height, datum.width))
                y.append(datum.label)
    x_array = np.vstack([x[i][np.newaxis,:] for i in range(len(x))])
    y_array = np.array(y) # column array
    return {'data':x_array, 'label':y_array}

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # generate plot
    plt.imshow(data); plt.axis('off')

base_path = '/Users/dpaiton/Google_Drive/School/UCB/Research/Redwood/EntropyCoding/'
model_deploy = os.path.join(base_path, 'models/entropy/drsae_deploy.prototxt')
pretrained_weights = os.path.join(base_path, 'snapshot/drsae/drsae_v00_iter_1000000.caffemodel')
test_dataset = '/Users/dpaiton/Work/Datasets/MNIST/examples/mnist_test_lmdb'

sys.path.insert(0,os.path.join(base_path,'python/entropyLoss/'))

image = {'cmap' : 'gray'}
matplotlib.rc('image', **image)

net = caffe.Net(model_deploy, pretrained_weights, caffe.TEST)

#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_raw_scale('data', 0.00390625) # rescale to [0, 1]
rescale_param = 0.00390625

img_set = load_lmdb(test_dataset)

#for img_idx in range(100):
#    #transformed_img = transformer.preprocess('data', img_set['data'][img_idx,...])
#    #net.blobs['data'].data[...] = transformed_img
#    net.blobs['data'].data[...] = img_set['data'][img_idx,...] * rescale_param
#    output = net.forward()
#    max_idx = np.argmax(output['prob'][0])
#    print str(img_set['label'][img_idx]) + " : " + str(max_idx)

img_idx = 0
net.blobs['data'].data[...] = img_set['data'][img_idx,...] * rescale_param
output = net.forward()

plt.figure()
param_shape = net.params['Ex'][0].data.shape
filters = net.params['Ex'][0].data.reshape(param_shape[0], int(np.sqrt(param_shape[1])), int(np.sqrt(param_shape[1])))
vis_square(filters)
plt.suptitle('E')
plt.show(block=False)

plt.figure()
param_shape = net.params['Sz_010'][0].data.shape
filters = net.params['Sz_010'][0].data.reshape(param_shape[0], int(np.sqrt(param_shape[1])), int(np.sqrt(param_shape[1])))
plt.suptitle('S')
vis_square(filters)

plt.figure()
param_shape = net.params['recon_010'][0].data.shape
filters = net.params['recon_010'][0].data.reshape(param_shape[0], int(np.sqrt(param_shape[1])), int(np.sqrt(param_shape[1])))
vis_square(filters)
plt.suptitle('D')
plt.show(block=False)

plt.figure()
param_shape = net.params['ip3'][0].data.shape
filters = net.params['ip3'][0].data.reshape(param_shape[0], int(np.sqrt(param_shape[1])), int(np.sqrt(param_shape[1])))
vis_square(filters)
plt.suptitle('C')
plt.show(block=False)

IPython.embed()
