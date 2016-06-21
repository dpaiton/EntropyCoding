import os
import numpy as np
import helper_functions as hf
import IPython

import tensorflow as tf # always import tensorflow after primary imports
from tensorflow.examples.tutorials.mnist import input_data

## Load in scheduler
#import lca_schedule as scheduler
#schedules = scheduler.schedule().blocks

## Load in parameters
import lca_params
params = lca_params.parameters().user_params
schedules = lca_params.parameters().model_schedule

for sch_idx, sch in enumerate(schedules):
  assert(len(sch["lr"]) == len(sch["weights"]))
  assert(len(sch["lr"]) == len(sch["decay_steps"]))
  assert(len(sch["lr"]) == len(sch["decay_rate"]))
  assert(len(sch["lr"]) == len(sch["staircase"]))

tf.set_random_seed(params["rand_seed"])

## Setup data
dataset = input_data.read_data_sets("MNIST_data", one_hot=True)

## Setup constants, placeholders & step counter
with tf.name_scope("constants") as scope:
  s = tf.placeholder(tf.float32, shape=[params["n"], None], name="input_data")  # Placeholder for data (column vector)
  y = tf.placeholder(tf.float32, shape=[params["l"], None], name="input_label") # Placeholder for ground truth

with tf.name_scope("parameters") as scope:
  eta = tf.placeholder(tf.float32, shape=(), name="LCA_update_rate")       # Placeholder for LCA update rate (dt/tau)
  lamb = tf.placeholder(tf.float32, shape=(), name="sparsity_tradeoff")    # Placeholder for sparsity loss tradeoff
  gamma = tf.placeholder(tf.float32, shape=(), name="supervised_tradeoff") # Placeholder for supervised loss tradeoff
  psi = tf.placeholder(tf.float32, shape=(), name="feedback_strength")     # Placeholder for feedback strength
  lr = tf.placeholder(tf.float32, shape=(), name="weight_learning_rate")   # Placeholder for Phi update rule

with tf.name_scope("step_counter") as scope:
  global_step = tf.Variable(0, trainable=False, name="global_step")

## Initialize membrane potential
with tf.name_scope("dynamic_variables") as scope:
  u = tf.Variable(tf.zeros(shape=tf.pack([params["m"], tf.shape(s)[1]]), dtype=tf.float32, name="u_init"),
    trainable=False, validate_shape=False, name="u")
  Tu = tf.select(tf.greater(u, lamb),
    u-lamb, tf.zeros(shape=tf.shape(u), dtype=tf.float32, name="zeros"))

with tf.variable_scope("weights") as scope:
  ## Initialize dictionary
  # Truncated normal distribution is a standard normal distribution with specified mean and standard
  # deviation, except that values whose magnitude is more than 2 standard deviations from the mean
  # are dropped and re-picked.
  weight_init_mean = 0.0
  weight_init_var = 1.0
  phi = tf.get_variable(name="phi", dtype=tf.float32,
    initializer=tf.truncated_normal([params["n"], params["m"]], mean=weight_init_mean,
    stddev=np.sqrt(weight_init_var), dtype=tf.float32, name="phi_init"), trainable=True)
  w = tf.get_variable(name="w", dtype=tf.float32,
    initializer=tf.truncated_normal([params["l"], params["m"]], mean=weight_init_mean,
    stddev=np.sqrt(weight_init_var), dtype=tf.float32, name="w_init"), trainable=True)
  #phi = tf.get_variable(tf.truncated_normal([params["n"], params["m"]], mean=weight_init_mean,
  #  stddev=np.sqrt(weight_init_var), dtype=tf.float32, name="phi_init"),
  #  trainable=True, name="phi")
  #w = tf.Variable(tf.truncated_normal([params["l"], params["m"]], mean=weight_init_mean,
  #  stddev=np.sqrt(weight_init_var), dtype=tf.float32, name="w_init"),
  #  trainable=True, name="w")

with tf.name_scope("normalize_weights") as scope:
  norm_phi = phi.assign(tf.nn.l2_normalize(phi, dim=1, epsilon=params["eps"], name="row_l2_norm"))
  norm_w = w.assign(tf.nn.l2_normalize(w, dim=0, epsilon=params["eps"], name="col_l2_norm"))
  normalize_weights = tf.group(norm_phi, norm_w, name="do_normalization")

with tf.name_scope("output") as scope:
  with tf.name_scope("image_estimate"):
    s_ = tf.matmul(phi, Tu, name="reconstruction")
  with tf.name_scope("label_estimate"):
    #y_ = tf.nn.softmax(tf.matmul(w, tf.nn.l2_normalize(Tu,
    #  dim=0, epsilon=1e-12, name="col_l2_norm"), name="classify"), name="softmax")
    y_ = tf.nn.softmax(tf.matmul(w, Tu,
      name="classify"), name="softmax")

with tf.name_scope("update_u") as scope:
  ## Discritized membrane update rule
  du = (tf.matmul(tf.transpose(phi), s) -
    tf.matmul(tf.matmul(tf.transpose(phi), phi, name="G") -
    tf.constant(np.identity(int(phi.get_shape()[1])),
    dtype=tf.float32, name="identity_matrix"), Tu,
    name="explaining_away") - u)
  #  psi * tf.matmul(tf.transpose(w), tf.mul(y, y_))))

  ## Operation to update the state
  step_lca = tf.group(u.assign_add(eta * du), name="do_update_u")

  ## Operation to clear u
  clear_u = tf.group(u.assign(tf.zeros(shape=tf.pack([params["m"], tf.shape(s)[1]]), dtype=tf.float32, name="zeros")))

with tf.name_scope("loss") as scope:
  with tf.name_scope("unsupervised"):
    euclidean_loss = 0.5 * tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(s, s_), 2.0)))
    sparse_loss = lamb * tf.reduce_sum(tf.abs(Tu))
    #entropy_loss = psi * -tf.reduce_sum(tf.clip_by_value(y_, 1e-10, 1.0) *\
    #  tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
    unsupervised_loss = euclidean_loss + sparse_loss
  with tf.name_scope("supervised"):
    with tf.name_scope("cross_entropy_loss"):
      cross_entropy_loss = gamma * -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
    supervised_loss = cross_entropy_loss
  total_loss = unsupervised_loss + supervised_loss

with tf.name_scope("accuracy_calculation") as scope:
  with tf.name_scope("prediction_bools"):
    correct_prediction = tf.equal(tf.argmax(y_, 0), tf.argmax(y, 0), name="individual_accuracy")
  with tf.name_scope("accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="avg_accuracy")

## Weight update method
with tf.name_scope("optimizers") as scope:
  grads_and_vars = list()
  apply_grads = list()
  for sch_idx, sch in enumerate(schedules):
    for w_idx, weight in enumerate(sch["weights"]):
      with tf.variable_scope("weights", reuse=True) as scope:
        var = [tf.get_variable(weight)]

      learning_rates = tf.train.exponential_decay(learning_rate=sch["lr"][w_idx],
        global_step=global_step, decay_steps=sch["decay_steps"][w_idx], decay_rate=sch["decay_rate"][w_idx],
        staircase=sch["staircase"][w_idx], name="annealing_schedule_"+weight)

      if params["opt_algorithm"] == "annealed_sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rates,
          name="grad_optimizer_"+weight)
      elif params["opt_algorithm"] == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rates,
          beta1=0.9, beta2=0.99, epsilon=1e-07, name="adam_optimizer_"+weight)

      grads_and_vars.append(optimizer.compute_gradients(total_loss, var_list=var))
      apply_grads.append(optimizer.apply_gradients(grads_and_vars[sch_idx*w_idx+w_idx],
        global_step=global_step))

## Checkpointing & graph output
if params["checkpoint"] > 0:
  if not os.path.exists(params["checkpoint_base_path"]+"/checkpoints"):
    os.makedirs(params["checkpoint_base_path"]+"/checkpoints")
  saver = tf.train.Saver()
  saver_def = saver.as_saver_def()
  with open(params["checkpoint_base_path"]+"/checkpoints/lca_gradient_saver_v"+params["version"]+".def", 'wb') as f:
    f.write(saver_def.SerializeToString())

## Initialization
init_op = tf.initialize_all_variables()

if params["display_plots"]:
  w_prev_fig = None
  phi_prev_fig = None
  recon_prev_fig = None

with tf.Session() as sess:
  with tf.device(params["device"]):
    ## Run session, passing empty arrays to set up network size
    sess.run(init_op,
      feed_dict={s:np.zeros((params["n"], params["batch"]), dtype=np.float32),
      y:np.zeros((params["l"], params["batch"]), dtype=np.float32)})

    for sch_idx, schedule in enumerate(schedules):
      print("\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
      print("Beginning schedule:")
      print("\n".join([key+"\t"+str(schedule[key]) for key in schedule.keys()]).expandtabs(16))
      print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
      lambda_ = schedule["lambda"]           # Sparsity tradeoff
      gamma_ = schedule["gamma"]             # Supervised loss tradeoff
      psi_ = schedule["psi"]
      num_steps_ = schedule["num_steps"]     # Number of time steps for enoding
      num_batches_ = schedule["num_batches"] # Number of batches to learn weights

      for step in range(num_batches_):
        if step == 0 and sch_idx == 0:
          tf.train.write_graph(sess.graph_def, params["checkpoint_base_path"]+"/checkpoints",
            "lca_gradient_graph_v"+params["version"]+".pb", as_text=False)

        ## Load in data
        batch = dataset.train.next_batch(params["batch"])
        input_image = hf.normalize_image(batch[0]).T
        input_label = batch[1].T

        ## Normalize weights
        normalize_weights.run()

        ## Perform inference
        clear_u.run({s:input_image})
        for t in range(num_steps_):
          step_lca.run({s:input_image, y:input_label, eta:params["dt"]/params["tau"], lamb:lambda_, gamma:gamma_, psi:psi_})

        ## Run update method - auto updates global_step
        for weight_idx in range(len(schedule["weights"])):
          apply_grads[sch_idx*weight_idx+weight_idx].run({\
            s:input_image,
            y:input_label,
            lamb:lambda_,
            gamma:gamma_})

        current_step = global_step.eval()
        ## Print statistics about run to stdout
        if current_step % params["stats_display"] == 0 and params["stats_display"] > 0:
          sparsity = 100 * np.count_nonzero(Tu.eval({lamb:lambda_})) / (params["m"] * params["batch"])
          train_accuracy = accuracy.eval({s:input_image, y:input_label, lamb:lambda_})
          print("\nGlobal batch index is %g"%current_step)
          print("Finished step %g out of %g, max val of u is %g, num active of a was %g percent"%(step+1,
            num_batches_, u.eval().max(), sparsity))
          print("\teuclidean loss:\t\t%g"%(euclidean_loss.eval({s:input_image, lamb:lambda_})))
          print("\tsparse loss:\t\t%g"%(sparse_loss.eval({s:input_image, lamb:lambda_})))
          print("\tunsupervised loss:\t%g"%(unsupervised_loss.eval({s:input_image, lamb:lambda_})))
          print("\tsupervised loss:\t%g"%(supervised_loss.eval({s:input_image,
            y:input_label, gamma:gamma_, lamb:lambda_})))
          print("\ttrain accuracy:\t\t%g"%(train_accuracy))

        ## Create plots for visualizing network
        if current_step % params["generate_plots"] == 0 and params["generate_plots"] > 0:
          if params["display_plots"]:
            w_prev_fig = hf.display_data_tiled(w.eval().reshape(params["l"], int(np.sqrt(params["m"])), int(np.sqrt(params["m"]))),
              title="Classification matrix at step number "+str(current_step), prev_fig=w_prev_fig)
            recon_prev_fig = hf.display_data_tiled(
              tf.transpose(s_).eval({lamb:lambda_}).reshape(params["batch"], int(np.sqrt(params["n"])), int(np.sqrt(params["n"]))),
              title="Reconstructions in step "+str(current_step), prev_fig=recon_prev_fig)
            phi_prev_fig = hf.display_data_tiled(tf.transpose(phi).eval().reshape(params["m"], int(np.sqrt(params["n"])), int(np.sqrt(params["n"]))),
              title="Dictionary for step "+str(current_step), prev_fig=phi_prev_fig)
          if params["save_plots"]:
            plot_out_dir = params["checkpoint_base_path"]+"/vis/"
            if not os.path.exists(plot_out_dir):
              os.makedirs(plot_out_dir)
            w_status = hf.save_data_tiled(
              w.eval().reshape(params["l"], int(np.sqrt(params["m"])), int(np.sqrt(params["m"]))),
              title="Classification matrix at step number "+str(current_step),
              save_filename=plot_out_dir+"class_v"+params["version"]+"-"+str(current_step).zfill(5)+".pdf")
            s_status = hf.save_data_tiled(
              tf.transpose(s_).eval({lamb:lambda_}).reshape(params["batch"], int(np.sqrt(params["n"])), int(np.sqrt(params["n"]))),
              title="Reconstructions in step "+str(current_step),
              save_filename=plot_out_dir+"recon_v"+params["version"]+"-"+str(current_step).zfill(5)+".pdf")
            phi_status = hf.save_data_tiled(
              tf.transpose(phi).eval().reshape(params["m"], int(np.sqrt(params["n"])), int(np.sqrt(params["n"]))),
              title="Dictionary for step "+str(current_step),
              save_filename=plot_out_dir+"phi_v"+params["version"]+"-"+str(current_step).zfill(5)+".pdf")

        ## Test network on validation dataset
        if current_step % params["val_test"] == 0 and params["val_test"] > 0:
          val_image = hf.normalize_image(dataset.validation.images).T
          val_label = dataset.validation.labels.T
          with tf.Session() as temp_sess:
            temp_sess.run(init_op, feed_dict={s:val_image, y:val_label})
            for t in range(num_steps_):
              temp_sess.run(step_lca, feed_dict={s:val_image, y:val_label, eta:params["dt"]/params["tau"], lamb:lambda_, gamma:gamma_, psi:0})
            val_accuracy = temp_sess.run(accuracy, feed_dict={s:val_image, y:val_label, lamb:lambda_})
            print("\t---validation accuracy: %g"%(val_accuracy))

        ## Write checkpoint to disc
        if current_step % params["checkpoint"] == 0 and params["checkpoint"] > 0:
          output_path = params["checkpoint_base_path"]+\
            "/checkpoints/lca_checkpoint_v"+params["version"]+"_s"+str(sch_idx)
          save_path = saver.save(sess, save_path=output_path, global_step=global_step)
          print("\tModel saved in file %s"%save_path)

    ## Write final checkpoint regardless of specified interval
    if params["checkpoint"] > 0:
      save_path = saver.save(sess, params["checkpoint_base_path"]+"/checkpoints/lca_checkpoint_v"+params["version"]+"_FINAL", global_step=global_step)
      print("\tFinal version of model saved in file %s"%save_path)

    with tf.Session() as temp_sess:
      test_images = dataset.test.images.T
      test_labels = dataset.test.labels.T
      temp_sess.run(init_op, feed_dict={s:test_images, y:test_labels})
      for t in range(num_steps_):
        temp_sess.run(step_lca, feed_dict={s:test_images, y:test_labels, eta:params["dt"]/params["tau"], lamb:0.1, gamma:1.0, psi:0})
      test_accuracy = temp_sess.run(accuracy, feed_dict={s:test_images, y:test_labels, lamb:0.1})
      print("Final accuracy: %g"%test_accuracy)

    #IPython.embed()
