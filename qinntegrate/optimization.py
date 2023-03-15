"""
    Optimization routines
"""
import numpy as np
from qibo.optimizers import optimize


def launch_optimization(
    predictor,
    target,
    xmin=[0.0],
    xmax=[1.0],
    npoints=int(5e2),
    max_iterations=100,
    max_evals=int(1e5),
    tol_error=1e-5,
    padding=False,
):
    """Receives a predictor (can be a circuit, NN, etc... which inherits from quanting.BaseVariationalObservable)
    and a target function (which inherits from target.TargetFunction) and performs the training
    """
    # Generate a set of random points within the integration limits
    xmin = np.array(xmin)
    xmax = np.array(xmax)
    if padding:
        xdelta = xmax - xmin
        xmin -= 0.1 * xdelta
        xmax += 0.1 * xdelta
    xrand = np.random.rand(npoints, target.ndim) * (xmax - xmin) + xmin

    # Now generate the target values
    ytrue = []
    for xarr in xrand:
        ytrue.append(target(xarr))
    ytrue = np.array(ytrue)

    # If we don't normalize... we are (sort of) doing importance "learning"...
    ytrue_abs = np.abs(ytrue) + 1e-6

    # Build a simple loss function, we will improve later on
    def loss(parameters):
        # Set the parameters
        predictor.set_parameters(parameters)

        # Compute the prediction for the points in x
        pred_y = []
        for xarr in xrand:
            pred_y.append(predictor.forward_pass(xarr))
        pred_y = np.array(pred_y)

        # Now compute MSE
        mse = np.mean((pred_y - ytrue) ** 2 / ytrue_abs)
        return mse

    options = {
        "verbose": -1,
        "tolfun": 1e-12,
        "ftarget": tol_error,  # Target error
        "maxiter": max_iterations,  # Maximum number of iterations
        "maxfeval": max_evals,  # Maximum number of function evaluations
    }

    # And... optimize!
    initial_p = predictor.parameters
    result = optimize(loss, initial_p, method="cma", options=options)

    # Set the final set of parameters
    best_p = result[1]
    predictor.set_parameters(best_p)
    print(f"Best set of parameters: {best_p=}")
    return best_p
