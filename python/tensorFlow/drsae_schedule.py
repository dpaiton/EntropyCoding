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
            {"prefix":"",
            "lambda":0.001,
            "gamma":0.0,
            "learning_rate":0.001,
            "num_steps":10,
            "num_batches":1000},
        \
            {"prefix":"",
            "lambda":0.01,
            "gamma":0.0,
            "learning_rate":0.001,
            "num_steps":10,
            "num_batches":5000},
        \
            {"prefix":"",
            "lambda":0.1,
            "gamma":0.0,
            "learning_rate":0.001,
            "num_steps":10,
            "num_batches":40000},
        \
            {"prefix":"",
            "lambda":0.1,
            "gamma":0.01,
            "learning_rate":0.001,
            "num_steps":10,
            "num_batches":20000},
        \
            {"prefix":"",
            "lambda":0.1,
            "gamma":0.1,
            "learning_rate":0.001,
            "num_steps":10,
            "num_batches":40000}]
