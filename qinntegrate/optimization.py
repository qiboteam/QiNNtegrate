"""
    Optimization routines
"""
import numpy as np
from qibo.optimizers import optimize
import time


def mse(y, p, norm=1.0):
    return np.mean((y - p) ** 2 / norm)

class Loss:
    def __init__(self, xarr, target, predictor, normalize=True):
        self._target = target
        self._predictor = predictor
        self._xarr = xarr

        self._ytrue = np.array([target(i) for i in xarr])
        self._ynorm = 1.0
        if normalize:
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
    

class SimAnnealer():

    def __init__(self, predictor, betai=1, betaf=500, nsteps=500, delta=0.5):
        """Simulated annealing implementation for VQCs model optimization"""

        from copy import deepcopy

        self._betai = betai
        self._betaf = betaf
        self._nsteps = nsteps
        self._predictor = predictor
        self._params = predictor.parameters
        self._nparams = len(self._params) 
        self._delta = delta

        print(f'Simulated annealing settings: beta_i={betai}, beta_f={betaf}, nsteps={nsteps}, delta={delta}.')

    def cooling(self, xrand, target, normalize, nprint=5):

        import random
        import matplotlib.pyplot as plt
        
        energy = Loss(xrand, target, self._predictor, normalize=normalize)
        energies = []

        beta = self._betai
        for _ in range(self._nsteps):
            beta += _
            # energy before
            ene1 = energy(self._params)
            deltas = np.random.uniform(-self._delta, self._delta, self._nparams)
            self._params += deltas
            self._predictor.set_parameters(self._params)
            ene2 = energy(self._params)
            energies.append(ene2)
            # evaluating Boltzmann energies
            p = min(1., np.exp(-beta*(ene2-ene1)))

            r = random.uniform(0,1)

            if(r >= p):
                self._params -= deltas
                self._predictor.set_parameters(self._params)
                energies[-1] = ene1

            if(_ % nprint == 0):
                print(f"Obtained E at step {_+1} with T={round(1/beta, 5)} is {round(energies[-1], 5)}")

        plt.title('Energy evolution during the annealing')
        plt.plot(energies, c='purple', lw=2, alpha=0.6)
        plt.xlabel('Step')
        plt.ylabel('E')
        plt.show()
            
        return self._predictor.parameters


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
    normalize=True,
    method="cma",
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

    loss = Loss(xrand, target, predictor, normalize=normalize)

    # And... optimize!
    # Use whatever is the current value of the parameters as the initial point
    initial_p = predictor.parameters

    if max_iterations == 0:
        print("Skipping the optimization phase since max_iterations=0")
        result = (None, initial_p)

    else:
        # tracking required time
        start = time.time()

        if method == 'cma':

            options = {
                "verbose": -1,
                "tolfun": 1e-12,
                "ftarget": tol_error,  # Target error
                "maxiter": max_iterations,  # Maximum number of iterations
                "maxfeval": max_evals,  # Maximum number of function evaluations
            }

            result = optimize(loss, initial_p, method="cma", options=options)

        elif method == 'BFGS':

            print('initial parameters: ', initial_p)

            options = {
                "disp" : True,
                "return_all" : True,
            }

            result = optimize(loss, initial_p, method=method, options=options)

        # this one is not working right now
        elif method == 'sgd':

            options = {
                "optimizer" : "Adam",
                "nepochs" : 500,
                "nmessage" : 1,
                "learning_rate" : 0.025,
            }

            result = optimize(loss, initial_p, method="sgd", options=options)

        
        elif method == 'annealing':
            
            simann = SimAnnealer(predictor)
            params = simann.cooling(xrand, target, normalize, nprint=1)

        # end of the time tracking
        end = time.time()


    # Set the final set of parameters
    if method == 'annealing':
        best_p = params
    else:
        best_p = result[1]
    
    predictor.set_parameters(best_p)
    print(f"Best set of parameters: {best_p=}")
    print(f"Total time required for the optimization: {round(end-start, 5)} sec.")
    return best_p
