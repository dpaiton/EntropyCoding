import IPython
import numpy as np
from scipy.optimize import minimize

def qDist(x, beta):
    return np.exp(-beta * x) / np.array([np.sum(np.exp(-beta * x), axis=1),]*x.shape[1]).transpose()

def pDist(x, beta):
    return np.exp(-beta * x) / np.array([np.sum(np.exp(-beta * x), axis=0),]*x.shape[0])

def entropy(prob, ax):
    return - np.sum(np.multiply(prob, np.log(prob)), axis=ax)

def optfn(x, num_batch, num_nodes, lamb, beta):
    '''
    Optimization callback for scipy.optimize.minimize()

    Determines the optimial solution when ignoring generators of x
    '''

    assert(x.shape[0] == num_batch * num_nodes)
    x = x.reshape(num_batch, num_nodes)

    loss = np.sum(entropy(qDist(x, beta), 1)) - lamb * np.sum(entropy(pDist(x, beta), 0))

    qdiff = np.dot(qDist(x, beta), np.dot(x.transpose(), qDist(x, beta))) - np.multiply(qDist(x, beta), x)
    pdiff = np.dot(pDist(x, beta), np.dot(x.transpose(), pDist(x, beta))) - np.multiply(pDist(x, beta), x)
    grad = np.power(beta,2.0) * (qdiff - lamb * pdiff)

    return loss, grad.flatten().astype(np.float64)

def main(args):
    #jac:True means that the optfn returns gradient as well as loss
    # requried for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg
    minfn_args = {
        "args": (args["num_batch"], args["num_nodes"], args["lamb"], args["beta"]),
        "method": "L-BFGS-B", "jac": True,
        "options": {"maxcor": 8, "maxiter": args["n_iter"], "disp": args["verbose"]}
    }

    results = []
    for example in range(args["x0"].shape[0]):
        results.append(minimize(optfn, args["x0"][example,:,:], **minfn_args).nit)

    IPython.embed()

if __name__ == "__main__":
    verbose = True
    n_iter = 1000

    num_batch = 50
    num_nodes = 10

    lamb = 0.1
    beta = 1.0

    num_examples = 2

    norm_mean = 0
    norm_var = 1
    x0 = np.zeros((num_examples, num_batch, num_nodes))
    for example in range(num_examples):
        x0[example,:,:] = np.random.normal(norm_mean, np.sqrt(norm_var), (num_batch, num_nodes))

    args = {
        "x0":x0,
        "num_batch":num_batch,
        "num_nodes":num_nodes,
        "lamb":lamb,
        "beta":beta,
        "n_iter":n_iter,
        "verbose":verbose}

    main(args)
