import re
import numpy as np
import matplotlib.pyplot as plt
import IPython

log_file = 'logfiles/drsae_ent_v0.log'

with open(log_file, 'r') as f:
    log_text = f.read()
    max_iter = float(re.findall("max_iter: (\d+)",log_text)[0])

    time_start = 0
    time_step = float(re.findall("display: (\d+)", log_text)[0])
    time_list = np.arange(time_start, max_iter, time_step)

    train_accuracy_vals = np.array([float(val) for val in re.findall('train_accuracy \= (\d+\.?\d*)', log_text)])
    test_accuracy_vals = np.array([float(val) for val in re.findall('test_accuracy \= (\d+\.?\d*)', log_text)])
    softmax_loss = np.array([float(val) for val in re.findall('softmax_loss \= (\d+\.?\d*)', log_text)])
    euclidean_loss = np.array([float(val) for val in re.findall('euclidean_loss_010 \= (\d+\.?\d*)', log_text)])
    sparse_loss = np.array([float(val) for val in re.findall('sparse_loss_010 \= (\d+\.?\d*)', log_text)])
    entropy_loss = np.array([float(val) for val in re.findall('en \= (\d+\.?\d*)', log_text)])

#Convert time step to epoch
batch_size = 64
num_images_per_epoch = 10000
epoch_list = np.multiply(time_list, np.float(batch_size)/num_images_per_epoch)

x_step = len(time_list)/10

fig = plt.figure()

ax1 = fig.add_subplot(3,2,1)
#line1 = ax1.plot(epoch_list[0:len(train_accuracy_vals)], train_accuracy_vals[0:len(epoch_list)], 'r', label='train_accuracy')
line1 = ax1.plot(time_list[0:len(train_accuracy_vals)], train_accuracy_vals[0:len(time_list)], 'r', label='train_accuracy')
ax1.set_ylabel('Train Accuracy')
ax1.set_ylim([0, 1])

ax2 = fig.add_subplot(3,2,2)
#line2 = ax2.plot(epoch_list[0:len(test_accuracy_vals)], test_accuracy_vals[0:len(epoch_list)], 'r', label='test_accuracy')
line2 = ax2.plot(time_list[0:len(test_accuracy_vals)], test_accuracy_vals[0:len(time_list)], 'r', label='test_accuracy')
ax2.set_ylabel('Test Accuracy')
ax2.set_ylim([0, 1])

ax3 = fig.add_subplot(3,2,3)
#line3 = ax3.plot(epoch_list[0:len(softmax_loss)], softmax_loss[0:len(epoch_list)], 'r', label='logistic_loss')
line3 = ax3.plot(time_list[0:len(softmax_loss)], softmax_loss[0:len(time_list)], 'r', label='logistic_loss')
ax3.set_ylabel('Logistic Loss')
#ax3.set_xlabel('Number of epochs')
#ax3.set_xlabel('Number of iterations')
ax3.set_ylim([0, 1])

ax4 = fig.add_subplot(3,2,4)
#line4 = ax4.plot(epoch_list[0:len(euclidean_loss)], euclidean_loss[0:len(epoch_list)], 'r', label='euclidean_loss')
line4 = ax4.plot(time_list[0:len(euclidean_loss)], euclidean_loss[0:len(time_list)], 'r', label='euclidean_loss')
ax4.set_ylabel('Euclidean Loss')
#ax4.set_xlabel('Number of epochs')
ax4.set_xlabel('Number of iterations')

ax5 = fig.add_subplot(3,2,5)
#line5 = ax5.plot(epoch_list[0:len(sparse_loss)], sparse_loss[0:len(epoch_list)], 'r', label='sparse_loss')
line5 = ax5.plot(time_list[0:len(sparse_loss)], sparse_loss[0:len(time_list)], 'r', label='sparse_loss')
ax5.set_ylabel('Sparse Loss')
#ax5.set_xlabel('Number of epochs')
ax5.set_xlabel('Number of iterations')

ax6 = fig.add_subplot(3,2,6)
#line5 = ax5.plot(epoch_list[0:len(entropy_loss)], entropy_loss[0:len(epoch_list)], 'r', label='sparse_loss')
line6 = ax6.plot(time_list[0:len(entropy_loss)], entropy_loss[0:len(time_list)], 'r', label='sparse_loss')
ax6.set_ylabel('Entropy Loss')
#ax6.set_xlabel('Number of epochs')
ax6.set_xlabel('Number of iterations')

plt.show(block=False)

IPython.embed()
