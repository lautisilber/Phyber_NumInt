import unittest
import numpy as np

from phyber_numint import (
    ODEIntegrator,
    IntegrationMethod
)
from phyber_numint import (
    damped_oscillator,
    damped_oscillator_analytical
)

def test_damped_oscillator(method: IntegrationMethod, dt: float):
    X0 = (0., 1.)
    ti, tf, dt = 0., 25., dt
    k, nu, m = 1., .5, 1.

    integ = ODEIntegrator(damped_oscillator, X0, ti, tf, dt, F_args=(k, nu, m))
    ts, Xs_numerical = integ.solve(method)

    Xs_analytical = damped_oscillator_analytical(ts, X0, k, nu, m)

    Xs_error = np.abs(Xs_numerical[:,0] - Xs_analytical)
    return np.max(Xs_error)

class TestDampedOscillator(unittest.TestCase):
    def test_euler_forward(self):
        max_error = test_damped_oscillator(IntegrationMethod.EULER_FORWARD, 1e-3)
        self.assertLess(max_error, 0.0231)
    
    def test_euler_backward(self):
        max_error = test_damped_oscillator(IntegrationMethod.EULER_BACKWARD, 1e-3)
        self.assertLess(max_error, 0.023)

    def test_euler_improved(self):
        max_error = test_damped_oscillator(IntegrationMethod.EULER_IMPROVED, 1e-3)
        self.assertLess(max_error, 0.023)
    
    def test_euler_rk2(self):
        max_error = test_damped_oscillator(IntegrationMethod.RUNGE_KUTTA_2, 1e-3)
        self.assertLess(max_error, 0.357)
    
    def test_euler_rk4(self):
        max_error = test_damped_oscillator(IntegrationMethod.RUNGE_KUTTA_4, 1e-3)
        self.assertLess(max_error, 0.023)

if __name__ == '__main__':
    unittest.main()

