"""
lambda         # Sparsity tradeoff
gamma          # Supervised loss tradeoff
learning_rate  # Learning rate for optimizer
num_steps      # Number of time steps for enoding
num_batches    # Number of batches to learn weights
"""
class schedule(object):
  def __init__(self):
    self.blocks = [ \
      {"prefix":"unsupervised",
      "lambda":0.1,
      "gamma":0.0,
      "learning_rate":0.001,
      "num_steps":50,
      "num_batches":10001},
    \
      {"prefix":"supervised",
      "lambda":0.1,
      "gamma":0.1,
      "learning_rate":0.001,
      "num_steps":50,
      "num_batches":10001},
    \
      {"prefix":"both",
      "lambda":0.1,
      "gamma":1.0,
      "learning_rate":0.001,
      "num_steps":50,
      "num_batches":20001},
    \
      {"prefix":"both",
      "lambda":0.1,
      "gamma":1.0,
      "learning_rate":0.0001,
      "num_steps":50,
      "num_batches":20001}]
