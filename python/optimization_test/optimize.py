import IPython
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def qDist(x, beta):
    return np.multiply(np.exp(-beta * x), 1.0/np.array([np.sum(np.exp(-beta * x), axis=1),]*x.shape[1]).transpose())

def pDist(x, beta):
    return np.multiply(np.exp(-beta * x), 1.0/np.array([np.sum(np.exp(-beta * x), axis=0),]*x.shape[0]))

def entropy(prob, ax):
    return -np.sum(np.multiply(prob, np.log(prob)), axis=ax)

def fwdfn(x, beta, node_weight, batch_weight):
    return node_weight * np.sum(entropy(qDist(x, beta), 1)) - batch_weight * np.sum(entropy(pDist(x, beta), 0))

def gradfn(x, beta, node_weight, batch_weight):
    qdiff = np.multiply(qDist(x, beta), np.array([np.sum(np.multiply(x, qDist(x, beta)),axis=0),]*x.shape[0])) - np.multiply(qDist(x, beta), x)
    pdiff = np.multiply(pDist(x, beta), np.array([np.sum(np.multiply(x, pDist(x, beta)),axis=1),]*x.shape[1]).transpose()) - np.multiply(pDist(x, beta), x)

    #qdiff = np.dot(qDist(x, beta), np.dot(x.transpose(), qDist(x, beta)).transpose()) - np.multiply(qDist(x, beta), x)
    #pdiff = np.dot(pDist(x, beta), np.dot(x.transpose(), pDist(x, beta)).transpose()) - np.multiply(pDist(x, beta), x)

    return np.power(beta,2.0) * (node_weight * qdiff - batch_weight * pdiff)

def optfn(x, num_batch, num_nodes, beta, node_weight, batch_weight):
    '''
    Optimization callback for scipy.optimize.minimize()

    Determines the optimial solution when ignoring generators of x
    '''
    assert(x.shape[0] == num_batch * num_nodes)
    x = x.reshape(num_batch, num_nodes)

    loss = fwdfn(x, beta, node_weight, batch_weight)
    grad = gradfn(x, beta, node_weight, batch_weight)

    #print "loss: " + str(loss)
    #print "grad: " + str(np.mean(grad))

    return loss, grad.flatten().astype(np.float64)

def gradCheck(x, eps, node_weight, batch_weight):
    act_grad = gradfn(x, beta, node_weight, batch_weight).flatten().astype(np.float64)

    eps_mat = np.zeros_like(x)
    est_grad = np.zeros_like(x.flatten())
    for i in range(len(x.flatten())):
        eps_mat.flat[i] = 1.
        eps_mat *= eps
        est_grad[i] = (fwdfn(x+eps_mat, beta, node_weight, batch_weight) - fwdfn(x-eps_mat, beta, node_weight, batch_weight)) / (2 * eps)
        eps_mat.flat[i] = 0.

    x_ = T.matrix('float64')
    beta_ = T.scalar('float64')
    node_weight_ = T.scalar('float64')
    batch_weight_ = T.scalar('float64')
    q = T.nnet.softmax(- beta_ * x_)
    p = T.nnet.softmax(- beta_ * x_.T)
    cost = -node_weight_ * (q * T.log(q)).sum() + batch_weight_ * (p * T.log(p)).sum()
    forward_fn = theano.function(inputs=[x_, beta_, node_weight_, batch_weight_], outputs=cost)
    dcost_dx = T.grad(cost, wrt=x_)
    grad_fn = theano.function(inputs=[x_, beta_, node_weight_, batch_weight_], outputs=dcost_dx)
    thn_grad = grad_fn(x, -1.0, node_weight, batch_weight).flatten().astype(np.float64)

    error1 = np.abs(est_grad - thn_grad)
    error2 = np.abs(act_grad - est_grad)
    error3 = np.abs(act_grad - thn_grad)

    IPython.embed()

    error = np.abs(act_grad - est_grad)
    return error

def main(args):
    #jac:True means that the optfn returns gradient as well as loss
    # requried for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg
    minfn_args = {
        "args": (args["num_batch"], args["num_nodes"], args["beta"], args["node_weight"], args["batch_weight"]),
        "method": "L-BFGS-B", "jac": True,
        "options": {"maxcor": 8, "maxiter": args["n_iter"], "disp": args["verbose"]}
    }

    results = []
    for example in range(args["x0"].shape[0]):
        #print "\nEXAMPLE # " + str(example)
        results.append(minimize(optfn, args["x0"][example,:,:], **minfn_args))

    if args["plot"]:
        for result in results:
            plt.figure()
            plt.hist(result.x)
            plt.show(block=False)

    if args["grad_check"]:
        #for example in range(args["x0"].shape[0]):
        error = gradCheck(args["x0"][0,:,:], np.power(10.,-4.), args["node_weight"], args["batch_weight"])

    IPython.embed()

if __name__ == "__main__":
    plot_figs  = False
    verbose    = False
    grad_check = True

    n_iter = 1000

    num_batch = 100
    num_nodes = 10

    # Only the left term (minimize entropy per image)
    #batch_weight = 0.0000000000001
    #node_weight = 1.0

    # Only the right term (maximize entropy per batch)
    #batch_weight = 1.0
    #node_weight = 0.0000000000001

    # Both terms
    batch_weight = 0.2
    node_weight= 0.8

    beta = -1.0

    num_examples = 10

    norm_mean = 0
    norm_var = 1
    x0 = np.zeros((num_examples, num_batch, num_nodes))
    for example in range(num_examples):
        x0[example,:,:] = np.random.normal(norm_mean, np.sqrt(norm_var), (num_batch, num_nodes))

    args = {
        "x0":x0,
        "num_batch":num_batch,
        "num_nodes":num_nodes,
        "beta":beta,
        "batch_weight":batch_weight,
        "node_weight":node_weight,
        "n_iter":n_iter,
        "grad_check":grad_check,
        "verbose":verbose,
        "plot":plot_figs}

    main(args)
