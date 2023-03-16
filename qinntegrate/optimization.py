"""
    Optimization routines
"""
import numpy as np
from qibo.optimizers import optimize


def mse(y, p, norm=1.0):
    return np.mean((y - p) ** 2 / norm)


class Loss:
    def __init__(self, xarr, target, predictor):
        self._target = target
        self._predictor = predictor
        self._xarr = xarr

        self._ytrue = np.array([target(i) for i in xarr])
        self._ynorm = np.abs(self._ytrue) + 1e-7

    def __call__(self, parameters):
        """Set the parameters in the predictor
        and compare the results with ytrue"""
        self._predictor.set_parameters(parameters)

        # Compute the prediction for the points in x
        pred_y = []
        for xarr in self._xarr:
            pred_y.append(self._predictor.forward_pass(xarr))
        pred_y = np.array(pred_y)

        return mse(pred_y, self._ytrue, norm=self._ynorm)


def launch_optimization(
    predictor,
    target,
    xmin=(0.0,),
    xmax=(1.0,),
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

    loss = Loss(xrand, target, predictor)

    options = {
        "verbose": -1,
        "tolfun": 1e-12,
        "ftarget": tol_error,  # Target error
        "maxiter": max_iterations,  # Maximum number of iterations
        "maxfeval": max_evals,  # Maximum number of function evaluations
    }

    # And... optimize!
    # Use whatever is the current value of the parameters as the initial point
    initial_p = predictor.parameters

    if max_iterations == 0:
        print("Skipping the optimization phase since max_iterations=0")
        result = (None, initial_p)
    else:
        result = optimize(loss, initial_p, method="cma", options=options)

    # Set the final set of parameters
    best_p = result[1]
    predictor.set_parameters(best_p)
    print(f"Best set of parameters: {best_p=}")
    return best_p
