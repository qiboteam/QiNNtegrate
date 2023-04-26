"""
    Optimization routines
"""
from abc import abstractmethod
import random
import time
import numpy as np
from qibo.optimizers import optimize
from scipy.optimize import basinhopping


def mse(y, p, norm=1.0):
    return np.mean((y - p) ** 2 / norm)


class Optimizer:
    """
    The optimizer.optimize method uses the same convention as qibo, and it returns
    an object for which element 1 is the best set of parameters

    The value of nbatch is the number of points that are used in each evaluation of the loss
    if randomize_batch is `True` the batch will be randomly sampled for each evaluation of the loss
    unless they correspond to the same step
    """

    _method = None

    def __init__(self, xarr, target, predictor, normalize=True, nbatch=100, randomize_batch=True):
        self._target = target
        self._predictor = predictor
        self._xarr = xarr
        self._options = {}

        # Ensure that the batch is not bigger than the number of points we have
        ntotal = xarr.shape[0]
        self._nbatch = np.minimum(ntotal, nbatch)

        # Now prepare the first batch
        self._arange = np.arange(0, ntotal)
        self._current_subset = np.random.choice(self._arange, size=self._nbatch, replace=False)
        self._random_batch = randomize_batch

        self._ytrue = np.array([target(i) for i in xarr])
        self._ynorm = np.ones_like(self._ytrue)
        if normalize:
            self._ynorm = np.abs(self._ytrue) + 1e-7

    def loss(self, parameters, same_step=False, **kwargs):
        """Set the parameters in the predictor
        and compare the results with ytrue in a MSE way
        if same_step is `False` a new set of points will be drawn, otherwise the previous one will be use
        This is necessary for numerical gradient descent or other algorithms which needs to evaluate the same points
        """
        self._predictor.set_parameters(parameters)

        if self._random_batch and not same_step:
            # Draw a new batch
            idx_subset = np.random.choice(self._arange, size=self._nbatch, replace=False)
            self._current_subset = idx_subset
        else:
            idx_subset = self._current_subset

        xarr = self._xarr[idx_subset]
        ytrue = self._ytrue[idx_subset]
        ynorm = self._ynorm[idx_subset]

        # Compute the prediction for the points in x
        pred_y = np.array([self._predictor.forward_pass(xx) for xx in xarr])

        return mse(pred_y, ytrue, norm=ynorm)

    @abstractmethod
    def set_options(self, **kwargs):
        """Cast the options passed into the format expected by the optimizer"""
        pass

    def _callback(self, *args, **kwargs):
        pass

    def optimize(self, initial_p):
        if self._method is None:
            raise ValueError(
                f"The optimizer {self.__class__.__name__} does not implement any methods"
            )
        return optimize(
            self.loss,
            initial_p,
            method=self._method,
            options=self._options,
            callback=self._callback,
        )


class CMA(Optimizer):
    _method = "cma"

    def set_options(self, **kwargs):
        self._options = {
            "verbose": -1,
            "tolfun": 1e-12,
            "ftarget": kwargs["tol_error"],  # target error
            "maxiter": kwargs["max_iterations"],  # maximum number of iterations
            "maxfeval": kwargs["max_evals"],  # maximum number of function evaluations
        }


class BFGS(Optimizer):
    _method = "BFGS"

    def __init__(self, *args, randomize_batch=False, **kwargs):
        # The BFGS needs to set the randomize_batch option to False
        # a new set of points will be drawn in the callback option
        super().__init__(*args, randomize_batch=False, **kwargs)

    #     def _callback(self, params):
    #         self._current_subset = np.random.choice(self._arange, size=self._nbatch, replace=False)

    def set_options(self, **kwargs):
        self._options = {"disp": True, "return_all": True}
        print(f"Initial parameters: {self._predictor.parameters}")


class LBFGS(Optimizer):
    _method = "L-BFGS-B"


class SGD(Optimizer):
    _method = "sgd"

    def set_options(self, **kwargs):
        self._options = {
            "optimizer": "Adam",
            "nmessage": 1,
            "learning_rate": 0.025,
            "nepochs": kwargs["max_iterations"],  # maximum number of iterations
        }


class BasinHopping(Optimizer):
    _method = "basinhopping"

    def set_options(self, **kwargs):
        self._niter = kwargs.get("max_iterations", 5)
        self._disp = kwargs.get("disp", True)

    def optimize(self, initial_p):
        print(f"Initial parameters: {self._predictor.parameters}")
        res = basinhopping(
            func=self.loss, x0=initial_p, niter=self._niter, disp=self._disp, niter_success=2
        )
        return None, res["x"]


class SimAnnealer(Optimizer):
    """Simulated annealing implementation for VQCs model optimization"""

    def set_options(self, **kwargs):
        self._betai = kwargs.get("betai", 1)
        self._betaf = kwargs.get("betaf", 1000)
        self._maxiter = kwargs.get("max_iterations", 500)
        self._dbeta = (self._betaf - self._betai) / self._maxiter
        self._delta = kwargs.get("delta", 0.5)

        self._nprint = 1
        print(
            f"Simulated annealing settings: beta_i={self._betai}, beta_f={self._betaf}, maxiter={self._maxiter}, delta={self._delta}."
        )

    def optimize(self, initial_p):
        """Performs the cooling of the system searching for minimum of the energy, where the energy is the loss function"""
        energies = []
        acc_rate = self._maxiter

        parameters = initial_p
        nparams = len(parameters)

        beta = self._betai
        for step in range(self._maxiter):
            # Energy before
            ene1 = self.loss(parameters)
            deltas = np.random.uniform(-self._delta, self._delta, nparams)
            parameters += deltas
            ene2 = self.loss(parameters, same_step=True)
            # evaluating Boltzmann energies
            p = min(1.0, np.exp(-beta * (ene2 - ene1)))
            r = random.uniform(0, 1)

            if r >= p:
                parameters -= deltas
                energies.append(ene1)
                acc_rate -= 1
            else:
                energies.append(ene2)

            if (nstep := step + 1) % self._nprint == 0:
                print(
                    f"Obtained E at step {nstep} with T={round(1/beta, 5)} is {round(energies[-1], 5)}"
                )

            beta += self._dbeta

        print(f"\nAnnealing finishes here with acceptance rate AR={acc_rate/self._maxiter}")
        best_p = parameters
        return None, best_p


def launch_optimization(
    xarr,
    predictor,
    target,
    optimizer_class,
    max_iterations=100,
    max_evals=int(1e5),
    tol_error=1e-5,
    normalize=True,
):
    """Receives a predictor (can be a circuit, NN, etc... which inherits from quanting.BaseVariationalObservable)
    and a target function (which inherits from target.TargetFunction) and performs the training
    """
    optimizer = optimizer_class(xarr, target, predictor, normalize=normalize)

    # And... optimize!
    # Use whatever is the current value of the parameters as the initial point
    initial_p = predictor.parameters

    if max_iterations == 0:
        print("Skipping the optimization phase since max_iterations=0")
        best_p = initial_p

    else:
        # tracking required time
        start = time.time()

        optimizer.set_options(
            max_iterations=max_iterations, max_evals=max_evals, tol_error=tol_error
        )
        results = optimizer.optimize(initial_p)
        best_p = results[1]

        # end of the time tracking
        end = time.time()
        print(f"Total time required for the optimization: {round(end-start, 5)} sec.")

    predictor.set_parameters(best_p)
    print(f"Best set of parameters: {best_p=}")
    return best_p


available_optimizers = {
    "cma": CMA,
    "bfgs": BFGS,
    "sgd": SGD,
    "lbfgs": LBFGS,
    "annealing": SimAnnealer,
    "basinhopping": BasinHopping,
}
