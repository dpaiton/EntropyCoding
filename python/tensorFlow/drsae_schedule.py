"""
lambda         # Sparsity tradeoff
gamma          # Supervised loss tradeoff
eta            # Time constant for z update
learning_rate  # Learning rate for SGD
num_steps      # Number of time steps for enoding
num_batches    # Number of batches to learn weights
"""
class schedule(object):
    def __init__(self):
        self.blocks = [ \
            {"prefix":"",
            "lambda":0.0,
            "gamma":0.0,
            "eta":1.0,
            "learning_rate":0.01,
            "num_steps":20,
            "num_batches":30001},
        \
            {"prefix":"",
            "lambda":0.001,
            "gamma":0.0,
            "eta":1.0,
            "learning_rate":0.01,
            "num_steps":20,
            "num_batches":30001},
        \
            {"prefix":"",
            "lambda":0.01,
            "gamma":0.0,
            "eta":1.0,
            "learning_rate":0.01,
            "num_steps":20,
            "num_batches":100001},
        \
            {"prefix":"",
            "lambda":0.01,
            "gamma":0.01,
            "eta":1.0,
            "learning_rate":0.01,
            "num_steps":20,
            "num_batches":50001},
        \
            {"prefix":"",
            "lambda":0.01,
            "gamma":0.1,
            "eta":1.0,
            "learning_rate":0.001,
            "num_steps":20,
            "num_batches":50001},
        \
            {"prefix":"",
            "lambda":0.01,
            "gamma":0.3,
            "eta":1.0,
            "learning_rate":0.001,
            "num_steps":20,
            "num_batches":50001},
        \
            {"prefix":"",
            "lambda":0.01,
            "gamma":0.3,
            "eta":1.0,
            "learning_rate":0.001,
            "num_steps":10,
            "num_batches":50001},
        \
            {"prefix":"",
            "lambda":0.01,
            "gamma":0.3,
            "eta":1.0,
            "learning_rate":0.0001,
            "num_steps":10,
            "num_batches":50001}]
