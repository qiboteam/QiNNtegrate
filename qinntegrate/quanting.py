"""Qibo interface of the integrator"""
import numpy as np
from qibo import models, gates, hamiltonians, set_backend

set_backend("numpy")

GEN_EIGENVAL = 0.5  # Eigenvalue for the parameter shift rule of rotations
SHIFT = np.pi / (4.0 * GEN_EIGENVAL)


class BaseVariationalObservable:
    """Generate a variational circuit following the example at
    https://qibo.science/qibo/stable/code-examples/advancedexamples.html#how-to-write-a-custom-variational-circuit-optimization

    In this case the inputs of the function are injected in the `ndim` first parameters.

    """

    def __init__(self, nqubits=3, nlayers=3, ndim=1, initial_state=None):
        self._ndim = ndim
        self._nqubits = nqubits
        self._nlayers = nlayers
        self._circuit = None
        self._observable = None
        self._variational_params = []
        self._initial_state = initial_state

        self.build_circuit()
        self.build_observable()

    def build_circuit(self):
        """Build step of the circuit"""
        circuit = models.Circuit(self._nqubits)
        for _ in range(self._nlayers):
            circuit.add((gates.RY(q, theta=0) for q in range(self._nqubits)))
            circuit.add((gates.CZ(q, q + 1) for q in range(0, self._nqubits - 1, 2)))
            circuit.add((gates.RY(q, theta=0) for q in range(self._nqubits)))
            circuit.add((gates.CZ(q, q + 1) for q in range(1, self._nqubits - 2, 2)))
            circuit.add(gates.CZ(0, self._nqubits - 1))
        circuit.add((gates.RY(q, theta=0) for q in range(self._nqubits)))
        self._circuit = circuit

        # Get the initial parameters
        pp = circuit.get_parameters()
        self._variational_params = np.concatenate(pp[self._ndim :])

    def build_observable(self):
        """Build step of the observable"""
        m0 = (1 / self._nqubits) * hamiltonians.Z(self._nqubits).matrix
        ham = hamiltonians.Hamiltonian(self._nqubits, m0)
        self._observable = ham

    def execute(self, xarr):
        """Obtain the value of the observable for the given xarr
        by evaluating the circuit in xarr and computing the expectation value of the observable
        """
        bexec = self._observable.backend.execute_circuit  # is this needed?
        circ_parameters = np.concatenate([xarr, self.parameters])
        self._circuit.set_parameters(circ_parameters)
        state = bexec(circuit=self._circuit, initial_state=self._initial_state).state()
        return self._observable.expectation(state)

    @property
    def parameters(self):
        return self._variational_params

    @property
    def nparams(self):
        return len(self._variational_params)

    def set_parameters(self, new_parameters):
        """Save the new set of parameters for the circuit
        They only get burned into the circuit when the forward pass is called
        """
        if len(new_parameters) != self.nparams:
            raise ValueError("Trying to set more parameters than those allowed by the circuit")
        self._variational_params = new_parameters

    def forward_pass(self, xarr):
        """Forward pass of the variational observable.
        This entails a parameter shift rule shift around the parameters xarr
        """
        xarr = np.array(xarr)

        result_plus = self.execute(xarr + SHIFT)
        result_minus = self.execute(xarr - SHIFT)

        res = GEN_EIGENVAL * (result_plus - result_minus)
        return res


if __name__ == "__main__":
    cc = BaseVariationalObservable()
    # initialize with random parameters
    aa = np.random.rand(cc.nparams)
    # and try to evaluate for a value in x
    cc.set_parameters(aa)
    res = cc.forward_pass([0.5])
