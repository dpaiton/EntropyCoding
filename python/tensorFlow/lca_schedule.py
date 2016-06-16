"""
lambda         # Sparsity tradeoff
gamma          # Supervised loss tradeoff
psi            # Feedback strength
learning_rate  # Learning rate for optimizer
num_steps      # Number of time steps for enoding
num_batches    # Number of batches to learn weights
"""
class schedule(object):
  def __init__(self):
    self.blocks = [ \
      {"prefix":"unsupervised",
      "lambda":0.5,
      "gamma":0.0,
      "psi":0.0,
      "learning_rate":0.001,
      "num_steps":20,
      "num_batches":20000},
    \
      {"prefix":"supervised",
      "lambda":0.5,
      "gamma":0.1,
      "psi":0.0,
      "learning_rate":0.001,
      "num_steps":20,
      "num_batches":20000},
    \
      {"prefix":"both",
      "lambda":0.5,
      "gamma":1.0,
      "psi":0.0,
      "learning_rate":0.001,
      "num_steps":20,
      "num_batches":20000},
    \
      {"prefix":"both",
      "lambda":0.5,
      "gamma":1.0,
      "psi":1.0,
      "learning_rate":0.001,
      "num_steps":20,
      "num_batches":40000}]
