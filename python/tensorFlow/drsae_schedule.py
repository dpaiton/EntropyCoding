"""
lambda         # Sparsity tradeoff
gamma          # Supervised loss tradeoff
learning_rate  # Learning rate for SGD
num_steps      # Number of time steps for enoding
num_batches    # Number of batches to learn weights
"""
class schedule(object):
    def __init__(self):
        self.blocks = [ \
            {"prefix":"",
            "lambda":0.001,
            "gamma":0.0,
            "learning_rate":0.001,
            "num_steps":20,
            "num_batches":1001},
        \
            {"prefix":"",
            "lambda":0.01,
            "gamma":0.0,
            "learning_rate":0.001,
            "num_steps":20,
            "num_batches":5001},
        \
            {"prefix":"",
            "lambda":0.1,
            "gamma":0.0,
            "learning_rate":0.001,
            "num_steps":20,
            "num_batches":20001},
        \
            {"prefix":"",
            "lambda":0.01,
            "gamma":0.01,
            "learning_rate":0.001,
            "num_steps":20,
            "num_batches":10001},
        \
            {"prefix":"",
            "lambda":0.01,
            "gamma":0.1,
            "learning_rate":0.001,
            "num_steps":20,
            "num_batches":10001}]
