import IPython
import numpy as np
from scipy.optimize import minimize

def optfn(x, num_batch, num_nodes, lamb, beta):
    '''
    Optimization callback for scipy.optimize.minimize()

    Determines the optimial solution when ignoring generators of x
    '''

    assert(x.shape[0] == num_batch * num_nodes)

    x = x.reshape(num_batch, num_nodes)

    q_dist = np.exp(-beta * x) / np.sum(np.exp(-beta * x))

    ent_fwd = -np.sum(np.sum(np.multiply(q_dist, np.log(q_dist)), axis=1), axis=0)
    act_fwd = np.sum(np.power(np.sum(q_dist / num_batch, axis=0) - (1.0 / num_nodes), 2.0), axis=0)

    loss = ent_fwd + lamb * act_fwd

    ent_tmp = np.array([np.sum(np.multiply(x, q_dist), axis=1),]*num_nodes).transpose() - x
    ent_grad = np.sum(np.power(beta, 2.0) * np.multiply(q_dist, ent_tmp))

    act_tmp1 = 2.0 * np.sum(np.sum(q_dist / num_batch, axis=0) - (1.0 / num_nodes), axis=0)
    act_tmp2 = (-beta / num_batch) * np.sum(np.exp(-beta * x), axis=0)
    act_grad = np.multiply(act_tmp1, act_tmp2)

    IPython.embed()

    grad = ent_grad + lamb * act_grad

    return loss, grad

def main(args):
    #jac:True means that the optfn returns gradient as well as loss
    # requried for CG, BFGS< Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg
    minfn_args = {
        "args": (args["num_batch"], args["num_nodes"], args["lamb"], args["beta"]),
        "method": "L-BFGS-B", "jac": True,
        "options": {"maxcor":8, "maxiter":args["n_iter"], "disp":args["verbose"]}
    }

    res = []
    for attempt in range(args["x0"].shape[0]):
        res.append(minimize(optfn, args["x0"][attempt,:,:], **minfn_args).nit)

    IPython.embed()

if __name__ == "__main__":
    n_iter = 1000
    verbose = True

    num_batch = 10
    num_nodes = 10

    lamb = 0.1
    beta = 1.0

    num_attempts = 1

    x0 = np.zeros((num_attempts, num_batch, num_nodes))
    x0[0,:,:] = np.identity(num_nodes)

    args = {
        "x0":x0,
        "num_batch":num_batch,
        "num_nodes":num_nodes,
        "lamb":lamb,
        "beta":beta,
        "n_iter":n_iter,
        "verbose":verbose}

    main(args)


