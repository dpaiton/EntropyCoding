import os
class parameters(object):
  def __init__(self):
    self.user_params = {\
      "version"              : "0",            # [str] Version for output
      "n"                    : 784,            # [int] Number of pixels
      "m"                    : 400,            # [int] Number of layer 1 elements
      "l"                    : 10,             # [int] Number of layer 2 elements (categories)
      "batch"                : 60,             # [int] Number of images in a batch
      "dt"                   : 0.001,          # [float] discrete global time constant
      "tau"                  : 0.01,           # [float] LCA time constant
      "eps"                  : 1e-12,          # [float] Small value to avoid division by zero
      "opt_algorithm"        : "annealed_sgd", # [str] Which optimization algorithm to use
      "checkpoint"           : -1,             # [int] How often to checkpoint
      "checkpoint_base_path" : os.path.expanduser("~")+"/Work/Projects/lca_output",
      "stats_display"        : 100,            # [int] How often to print updates to stdout
      "val_test"             : -1,             # [int] How often to run the validation test
      "generate_plots"       : 200,            # [int] How often to generate plots
      "display_plots"        : False,          # [bool] Whether to display plots
      "save_plots"           : True,           # [bool] Whether to save plots to file
      "device"               : "/cpu:0",       # [str] Which device to run on
      "rand_seed"            : 1234567890}     # [int] Random seed

    self.model_schedule= [ \
      {"weights"     : ["phi"], # [list of str] Which weights to update
      "lambda"      : 0.7,      # [float] Sparsity tradeoff
      "gamma"       : 0.0,      # [float] Supervised loss tradeoff
      "psi"         : 0.0,      # [float] Feedback strength
      "num_steps"   : 20,       # [int] Number of iterations of LCA update equation
      "lr"          : [0.1],    # [list of float] Learning rates for weight updates
      "decay_steps" : [3000],   # [list of int] How often to decay for SGD annealing
      "decay_rate"  : [0.5],    # [list of float] Rate to decay for SGD annealing
      "staircase"   : [True],   # [list of bool] Whether SGD annealing should be step (T) or exponential (F)
      "num_batches" : 6000},    # [int] Number of batches to run for this schedule
    \
      {"weights"     : ["w"],
      "lambda"      : 0.7,
      "gamma"       : 0.0,
      "psi"         : 0.0,
      "num_steps"   : 20,
      "lr"          : [0.1],
      "decay_steps" : [3000],
      "decay_rate"  : [0.5],
      "staircase"   : [True],
      "num_batches" : 6000},
    \
      {"weights"     : ["phi", "w"],
      "lambda"      : 0.7,
      "gamma"       : 0.0,
      "psi"         : 0.0,
      "num_steps"   : 20,
      "lr"          : [0.01, 0.1],
      "decay_steps" : [3000,]*2,
      "decay_rate"  : [0.5,]*2,
      "staircase"   : [True,]*2,
      "num_batches" : 6000}]
