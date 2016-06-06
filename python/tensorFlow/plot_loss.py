import numpy as np
import os
import io
import matplotlib.pyplot as plt
import re
import IPython

"""
Plot log values over time

Output: arrays of values

Input: log file location
"""
def get_log_outputs(log_file):
    with open(log_file, "r") as f:
        log_text = f.read()

    #max_iter = float(re.findall("out of (\d+)",log_text)[0])

    #time_start = 0 
    #time_step = float(re.findall("batch number (\d+)", log_text)[0])
    #time_list = np.arange(time_start, max_iter, time_step)

    #train_accuracy_vals = np.array([float(val) for val in re.findall("train_accuracy:\s+(\d+\.?\d*)", log_text)])
    #alpha_1_vals = np.array([float(val) for val in re.findall("alpha_1 value:\s+(\d+\.?\d*)", log_text)])
    #alpha_2_vals = np.array([float(val) for val in re.findall("alpha_2 value:\s+(\d+\.?\d*)", log_text)])
    #euclidean_loss = np.array([float(val) for val in re.findall("euclidean loss:\s+(\d+\.?\d*)", log_text)])
    #sparse_loss = np.array([float(val) for val in re.findall("sparse loss:\s+(\d+\.?\d*)", log_text)])
    #cross_entropy_loss = np.array([float(val) for val in re.findall("cross-entropy loss:\s+(\d+\.?\d*)", log_text)])
    #supervised_loss = np.array([float(val) for val in re.findall("supervised loss:\s+(\d+\.?\d*)", log_text)])
    #validation_accuracy = np.array([float(val) for val in re.findall("validation accuracy (\d+\.?\d*)", log_text)])

    return log_text

"""
Create activation plots similar to those in
JT Rolfe, Y Lecun (2013) - Discriminative Recurrent Sparse Auto-Encoders
"""
#def plot_connection_summaries(enc_w, dec_w, rec_w


log_file = "logfiles/drsae_full_schedule.log"

log_text = get_log_outputs(log_file)

IPython.embed()

